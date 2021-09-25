from joblib import Parallel, delayed
import pandas as pd
from pandas.core.common import flatten
from collections import OrderedDict
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")
pd.set_option("max_columns", 300)


CONFIG = {
    "root_dir": "../input/optiver-realized-volatility-prediction/",
    "ckpt_dir": "../ckpts",
    "kfold_seed": 42,
    "n_splits": 5,
    "n_clusters": 7,
    "shuffle": True,
    "shuffle_seed": 1996
}


def read_train_test():
    train = pd.read_csv("../input/optiver-realized-volatility-prediction/train.csv")
    test = pd.read_csv("../input/optiver-realized-volatility-prediction/test.csv")

    # Create a key to merge with book and trade data
    train["row_id"] = train["stock_id"].astype(str) + "-" + train["time_id"].astype(str)
    test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str)

    print("Our training set has {} rows".format(train.shape[0]))

    return train, test


def activity_counts(df):
    activity_counts_ = df.groupby(["time_id"])["seconds_in_bucket"].agg("count").reset_index()
    activity_counts_ = activity_counts_.rename(columns={"seconds_in_bucket": "activity_counts"})
    return activity_counts_


def calc_wap(df, pos=1):
    wap = (df["bid_price{}".format(pos)] * df["ask_size{}".format(pos)] + df["ask_price{}".format(pos)] * df[
        "bid_size{}".format(pos)]) / (df["bid_size{}".format(pos)] + df["ask_size{}".format(pos)])
    return wap


def calc_wap2(df, pos=1):
    wap = (df["bid_price{}".format(pos)] * df["bid_size{}".format(pos)] + df["ask_price{}".format(pos)] * df[
        "ask_size{}".format(pos)]) / (df["bid_size{}".format(pos)] + df["ask_size{}".format(pos)])
    return wap


def wp(df):
    wp_ = (df["bid_price1"] * df["bid_size1"] + df["ask_price1"] * df["ask_size1"] + df["bid_price2"] * df[
        "bid_size2"] + df["ask_price2"] * df["ask_size2"]) / (
                  df["bid_size1"] + df["ask_size1"] + df["bid_size2"] + df["ask_size2"])
    return wp_


def maximum_drawdown(series, window=600):
    # window for 10 minutes, use min_periods=1 if you want to allow the expanding window
    roll_max = series.rolling(window, min_periods=1).max()
    second_drawdown = series / roll_max - 1.0
    max_drawdown = second_drawdown.rolling(window, min_periods=1).min()

    return max_drawdown


def log_return(series):
    return np.log(series).diff().fillna(0)


def rolling_log_return(series, rolling=60):
    return np.log(series.rolling(rolling)).diff().fillna(0)


def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))


def diff(series):
    return series.diff().fillna(0)


def time_diff(series):
    return series.diff().fillna(series)


def order_flow_imbalance(df, pos=1):
    df["bid_price{}_diff".format(pos)] = df.groupby(["time_id"])["bid_price{}".format(pos)].apply(diff)
    df["bid_size{}_diff".format(pos)] = df.groupby(["time_id"])["bid_price{}".format(pos)].apply(diff)
    df["bid_order_flow{}".format(pos)] = df["bid_size{}".format(pos)].copy(deep=True)
    df["bid_order_flow{}".format(pos)].loc[df["bid_price{}_diff".format(pos)] < 0] *= -1
    df["bid_order_flow{}".format(pos)].loc[df["bid_price{}_diff".format(pos)] == 0] = \
        df["bid_size{}_diff".format(pos)].loc[df["bid_price{}_diff".format(pos)] == 0]

    df["ask_price{}_diff".format(pos)] = df.groupby(["time_id"])["ask_price{}".format(pos)].apply(diff)
    df["ask_size{}_diff".format(pos)] = df.groupby(["time_id"])["ask_price{}".format(pos)].apply(diff)
    df["ask_order_flow{}".format(pos)] = df["ask_size{}".format(pos)].copy(deep=True)
    df["ask_order_flow{}".format(pos)].loc[df["ask_price{}_diff".format(pos)] < 0] *= -1
    df["ask_order_flow{}".format(pos)].loc[df["ask_price{}_diff".format(pos)] == 0] = \
        df["ask_size{}_diff".format(pos)].loc[df["ask_price{}_diff".format(pos)] == 0]

    order_flow_imbalance_ = df["bid_order_flow{}".format(pos)] - df["ask_order_flow{}".format(pos)]

    df.drop(["bid_price{}_diff".format(pos), "bid_size{}_diff".format(pos), "bid_order_flow{}".format(pos),
             "ask_price{}_diff".format(pos), "ask_size{}_diff".format(pos), "ask_order_flow{}".format(pos)], axis=1,
            inplace=True)

    return order_flow_imbalance_ + 1e-8


def order_book_slope(df):

    df["mid_point"] = (df["bid_price1"] + df["ask_price1"]) / 2
    best_mid_point_ = df.groupby(["time_id"])["mid_point"].agg("max").reset_index()
    best_mid_point_ = best_mid_point_.rename(columns={"mid_point": "best_mid_point"})
    df = df.merge(best_mid_point_, how="left", on="time_id")

    best_mid_point = df["best_mid_point"].copy()
    df.drop(["mid_point", "best_mid_point"], axis=1, inplace=True)

    def ratio(series):
        ratio_ = series / series.shift()
        return ratio_

    bid_price1_ratio = df.groupby(["time_id"])["bid_price1"].apply(ratio)
    bid_price1_mid_point_ratio = df["bid_price1"] / best_mid_point
    bid_price1_ratio = abs(bid_price1_ratio.fillna(bid_price1_mid_point_ratio) - 1)

    bid_size1_ratio = df.groupby(["time_id"])["bid_size1"].apply(ratio) - 1
    bid_size1_ratio = bid_size1_ratio.fillna(df["bid_size1"])
    df["DE"] = (bid_size1_ratio / bid_price1_ratio).replace([np.inf, -np.inf], np.nan).fillna(0)

    ask_price1_ratio = df.groupby(["time_id"])["ask_price1"].apply(ratio)
    ask_price1_mid_point_ratio = df["ask_price1"] / best_mid_point
    ask_price1_ratio = abs(ask_price1_ratio.fillna(ask_price1_mid_point_ratio) - 1)

    ask_size1_ratio = df.groupby(["time_id"])["ask_size1"].apply(ratio) - 1
    ask_size1_ratio = ask_size1_ratio.fillna(df["ask_size1"])
    df["SE"] = (ask_size1_ratio / ask_price1_ratio).replace([np.inf, -np.inf], np.nan).fillna(0)

    df["order_book_slope"] = (df["DE"] + df["SE"]) / 2
    order_book_slope_ = df.groupby(["time_id"])["order_book_slope"].agg("mean").reset_index()
    df.drop(["order_book_slope", "DE", "SE"], axis=1, inplace=True)

    return order_book_slope_


def ldispersion(df):
    LDispersion = 1 / 2 * (
            df["bid_size1"] / (df["bid_size1"] + df["bid_size2"]) * abs(df["bid_price1"] - df["bid_price2"]) + df[
        "ask_size1"] / (df["ask_size1"] + df["ask_size2"]) * abs(df["ask_price1"] - df["ask_price2"]))
    return LDispersion


def depth_imbalance(df, pos=1):
    depth_imbalance_ = (df["bid_size{}".format(pos)] - df["ask_size{}".format(pos)]) / (
            df["bid_size{}".format(pos)] + df["ask_size{}".format(pos)])

    return depth_imbalance_


def height_imbalance(df, pos=1):
    height_imbalance_ = (df["bid_price{}".format(pos)] - df["ask_price{}".format(pos)]) / (
            df["bid_price{}".format(pos)] + df["ask_price{}".format(pos)])

    return height_imbalance_


def pressure_imbalance(df):
    mid_price = (df["bid_price1"] + df["ask_price1"]) / 2

    weight_buy = mid_price / (mid_price - df["bid_price1"]) + mid_price / (mid_price - df["bid_price2"])
    pressure_buy = df["bid_size1"] * (mid_price / (mid_price - df["bid_price1"])) / weight_buy + df["bid_size2"] * (
            mid_price / (mid_price - df["bid_price2"])) / weight_buy

    weight_sell = mid_price / (df["ask_price1"] - mid_price) + mid_price / (df["ask_price2"] - mid_price)
    pressure_sell = df["ask_size1"] * (mid_price / (df["ask_price1"] - mid_price)) / weight_sell + df["ask_size2"] * (
            mid_price / (df["ask_price2"] - mid_price)) / weight_sell

    pressure_imbalance_ = np.log(pressure_buy) - np.log(pressure_sell)

    return pressure_imbalance_


def relative_spread(df, pos=1):
    relative_spread_ = 2 * (df["ask_price{}".format(pos)] - df["bid_price{}".format(pos)]) / (
            df["ask_price{}".format(pos)] + df["bid_price{}".format(pos)])

    return relative_spread_


def count_unique(series):
    return len(np.unique(series))


# Function to fill missing seconds (for each stock id)
def fill_missing_seconds(df):
    time_ids = list(OrderedDict.fromkeys(df["time_id"]))

    filled_df = pd.DataFrame(index=range(len(time_ids) * 600), columns=df.columns)

    all_seconds_in_bucket = list(flatten([range(600) for i in range(len(time_ids))]))
    filled_df["seconds_in_bucket"] = all_seconds_in_bucket

    all_time_ids = list(flatten([[time_ids[i]] * 600 for i in range(len(time_ids))]))
    filled_df["time_id"] = all_time_ids

    filled_df = pd.merge(filled_df, df, on=["time_id", "seconds_in_bucket"], how="left", suffixes=("_to_move", ""))

    to_remove_columns = [column for column in filled_df.columns if "to_move" in column]
    filled_df = filled_df.drop(to_remove_columns, axis=1)

    filled_df = filled_df.fillna(method="ffill")

    return filled_df


# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path):
    df = pd.read_parquet(file_path)

    # float 64 to float 32
    float_cols = df.select_dtypes(include=[np.float64]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # int 64 to int 32
    int_cols = df.select_dtypes(include=[np.int64]).columns
    df[int_cols] = df[int_cols].astype(np.int32)

    # rebase seconds_in_bucket
    df["seconds_in_bucket"] = df["seconds_in_bucket"] - df["seconds_in_bucket"].min()

    # Calculate seconds gap
    df["seconds_gap"] = df.groupby(["time_id"])["seconds_in_bucket"].apply(time_diff)

    # Calculate Wap
    df["wap1"] = calc_wap(df, pos=1)
    df["wap2"] = calc_wap(df, pos=2)

    # Calculate wap balance
    df["wap_balance"] = abs(df["wap1"] - df["wap2"])

    # Calculate log returns
    df["log_return1"] = df.groupby(["time_id"])["wap1"].apply(log_return)
    df["log_return2"] = df.groupby(["time_id"])["wap2"].apply(log_return)

    # Calculate spread
    df["bid_ask_spread1"] = df["ask_price1"] / df["bid_price1"] - 1
    df["bid_ask_spread2"] = df["ask_price2"] / df["bid_price2"] - 1

    # order flow imbalance
    df["order_flow_imbalance1"] = order_flow_imbalance(df, 1)
    df["order_flow_imbalance2"] = order_flow_imbalance(df, 2)

    # order book slope
    order_slope_ = order_book_slope(df)
    df = df.merge(order_slope_, how="left", on="time_id")

    # depth imbalance
    df["depth_imbalance1"] = depth_imbalance(df, pos=1)
    df["depth_imbalance2"] = depth_imbalance(df, pos=2)

    # height imbalance
    df["height_imbalance1"] = height_imbalance(df, pos=1)
    df["height_imbalance2"] = height_imbalance(df, pos=2)

    # pressure imbalance
    df["pressure_imbalance"] = pressure_imbalance(df)

    # total volume
    df["total_volume"] = (df["ask_size1"] + df["ask_size2"]) + (df["bid_size1"] + df["bid_size2"])

    # Dict for aggregations
    create_feature_dict = {
        "wap1": [np.sum, np.std],
        "wap2": [np.sum, np.std],
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility],
        "wap_balance": [np.sum, np.max, np.min, np.std],
        "bid_ask_spread1": [np.sum, np.max, np.min, np.std],
        "bid_ask_spread2": [np.sum, np.max, np.min, np.std],
        "order_flow_imbalance1": [np.sum, np.max, np.min, np.std],
        "order_flow_imbalance2": [np.sum, np.max, np.min, np.std],
        "order_book_slope": [np.mean, np.max],
        "depth_imbalance1": [np.sum, np.max, np.std],
        "depth_imbalance2": [np.sum, np.max, np.std],
        "height_imbalance1": [np.sum, np.max, np.std],
        "height_imbalance2": [np.sum, np.max, np.std],
        "pressure_imbalance": [np.sum, np.max, np.std],
        "total_volume": [np.sum],
        "seconds_gap": [np.mean]
    }
    create_feature_dict_time = {
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility],
        "wap_balance": [np.sum, np.max, np.min, np.std],
        "bid_ask_spread1": [np.sum, np.max, np.min, np.std],
        "bid_ask_spread2": [np.sum, np.max, np.min, np.std],
        "order_flow_imbalance1": [np.sum, np.max, np.min, np.std],
        "order_flow_imbalance2": [np.sum, np.max, np.min, np.std],
        "total_volume": [np.sum],
        "seconds_gap": [np.mean]
    }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(feature_dict, seconds_in_bucket, add_suffix=False):
        # Group by the window
        df_feature_ = df[df["seconds_in_bucket"] >= seconds_in_bucket].groupby(["time_id"]).agg(
            feature_dict).reset_index()
        # Rename columns joining suffix
        df_feature_.columns = ["_".join(col) for col in df_feature_.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature_ = df_feature_.add_suffix("_" + str(seconds_in_bucket))
        return df_feature_

    # Get the stats for different windows
    windows = [0, 150, 300, 450]
    add_suffixes = [False, True, True, True]
    df_feature = None

    for window, add_suffix in zip(windows, add_suffixes):
        if df_feature is None:
            df_feature = get_stats_window(feature_dict=create_feature_dict, seconds_in_bucket=window,
                                          add_suffix=add_suffix)
        else:
            new_df_feature = get_stats_window(feature_dict=create_feature_dict_time, seconds_in_bucket=window,
                                              add_suffix=add_suffix)
            df_feature = df_feature.merge(new_df_feature, how="left", left_on="time_id_",
                                          right_on="time_id__{}".format(window))

            # Drop unnecesary time_ids
            df_feature.drop(["time_id__{}".format(window)], axis=1, inplace=True)

    # Create row_id so we can merge
    stock_id = file_path.split("=")[1]
    df_feature["row_id"] = df_feature["time_id_"].apply(lambda x: f"{stock_id}-{x}")
    df_feature.drop(["time_id_"], axis=1, inplace=True)

    return df_feature


# Function to preprocess trade data (for each stock id)
def trade_preprocessor(file_path):
    df = pd.read_parquet(file_path)

    # float 64 to float 32
    float_cols = df.select_dtypes(include=[np.float64]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # int 64 to int 32
    int_cols = df.select_dtypes(include=[np.int64]).columns
    df[int_cols] = df[int_cols].astype(np.int32)

    # rebase seconds_in_bucket
    df["seconds_in_bucket"] = df["seconds_in_bucket"] - df["seconds_in_bucket"].min()

    # Calculate seconds gap
    df["seconds_gap"] = df.groupby(["time_id"])["seconds_in_bucket"].apply(time_diff)

    # Calculate log return
    df["price_log_return"] = df.groupby("time_id")["price"].apply(log_return)

    # Calculate volumes
    df["volumes"] = df["price"] * df["size"]

    # Dict for aggregations
    create_feature_dict = {
        "price_log_return": [realized_volatility],
        "volumes": [np.sum, np.max, np.std],
        "order_count": [np.sum],
        "seconds_gap": [np.mean]
    }
    create_feature_dict_time = {
        "price_log_return": [realized_volatility],
        "volumes": [np.sum, np.max, np.std],
        "order_count": [np.sum],
        "seconds_gap": [np.mean]
    }

    # Function to get group stats for different windows (seconds in bucket)
    def get_stats_window(feature_dict, seconds_in_bucket, add_suffix=False):
        # Group by the window
        df_feature_ = df[df["seconds_in_bucket"] >= seconds_in_bucket].groupby(["time_id"]).agg(
            feature_dict).reset_index()
        # Rename columns joining suffix
        df_feature_.columns = ["_".join(col) for col in df_feature_.columns]
        # Add a suffix to differentiate windows
        if add_suffix:
            df_feature_ = df_feature_.add_suffix("_" + str(seconds_in_bucket))
        return df_feature_

    # Get the stats for different windows
    windows = [0, 150, 300, 450]
    add_suffixes = [False, True, True, True]
    df_feature = None

    for window, add_suffix in zip(windows, add_suffixes):
        if df_feature is None:
            df_feature = get_stats_window(feature_dict=create_feature_dict, seconds_in_bucket=window,
                                          add_suffix=add_suffix)
        else:
            new_df_feature = get_stats_window(feature_dict=create_feature_dict_time, seconds_in_bucket=window,
                                              add_suffix=add_suffix)
            df_feature = df_feature.merge(new_df_feature, how="left", left_on="time_id_",
                                          right_on="time_id__{}".format(window))

            # Drop unnecesary time_ids
            df_feature.drop(["time_id__{}".format(window)], axis=1, inplace=True)

    def tendency(price, vol):
        df_diff = np.diff(price)
        val = (df_diff / price[1:]) * 100
        power = np.sum(val * vol[1:])
        return (power)

    lis = []
    for n_time_id in df["time_id"].unique():
        df_id = df[df["time_id"] == n_time_id]

        tendencyV = tendency(df_id["price"].values, df_id["size"].values)
        energy = np.mean(df_id["price"].values ** 2)

        lis.append(
            {
                "time_id": n_time_id,
                "tendency": tendencyV,
                "energy": energy,
            }
        )

    df_lr = pd.DataFrame(lis)
    df_feature = df_feature.merge(df_lr, how="left", left_on="time_id_", right_on="time_id")

    # Create row_id so we can merge
    df_feature = df_feature.add_prefix("trade_")
    stock_id = file_path.split("=")[1]
    df_feature["row_id"] = df_feature["trade_time_id_"].apply(lambda x: f"{stock_id}-{x}")
    df_feature.drop(["trade_time_id_", "trade_time_id"], axis=1, inplace=True)
    return df_feature


# Funtion to make preprocessing function in parallel (for each stock id)
def preprocessor(list_stock_ids, is_train=True):
    # Parrallel for loop
    def for_joblib(stock_id):
        # Train
        if is_train:
            file_path_book = CONFIG["root_dir"] + "book_train.parquet/stock_id=" + str(stock_id)
            file_path_trade = CONFIG["root_dir"] + "trade_train.parquet/stock_id=" + str(stock_id)
        # Test
        else:
            file_path_book = CONFIG["root_dir"] + "book_test.parquet/stock_id=" + str(stock_id)
            file_path_trade = CONFIG["root_dir"] + "trade_test.parquet/stock_id=" + str(stock_id)

        # Preprocess book and trade data and merge them
        df_tmp = pd.merge(book_preprocessor(file_path_book), trade_preprocessor(file_path_trade), on="row_id",
                          how="left")

        # Return the merge dataframe
        return df_tmp

    # Use parallel api to call paralle for loop
    df = Parallel(n_jobs=-1, verbose=1)(delayed(for_joblib)(stock_id) for stock_id in list_stock_ids)

    # Concatenate all the dataframes that return from Parallel
    df = pd.concat(df, ignore_index=True)

    return df


# Process agg by kmeans
def get_kmeans_idx(n_clusters=7):
    train_p = pd.read_csv("../input/optiver-realized-volatility-prediction/train.csv")
    train_p = train_p.pivot(index="time_id", columns="stock_id", values="target")

    corr = train_p.corr()

    ids = corr.index

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(corr.values)

    kmeans_clusters = []
    for n in range(n_clusters):
        kmeans_clusters.append([(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0])

    return kmeans_clusters


def agg_stat_features_by_clusters(df, n_clusters=7, function=np.nanmean, post_fix="_cluster_mean"):
    kmeans_clusters = get_kmeans_idx(n_clusters=n_clusters)

    clusters = []
    agg_columns = [
        "time_id",
        "stock_id",
        "log_return1_realized_volatility",
        "log_return2_realized_volatility",
        "order_flow_imbalance1_sum",
        "order_flow_imbalance2_sum",
        "order_book_slope_mean",
        "depth_imbalance1_std",
        "depth_imbalance2_std",
        "height_imbalance1_sum",
        "height_imbalance2_sum",
        "pressure_imbalance_std",
        "total_volume_sum",
        "seconds_gap_mean",
        "trade_price_log_return_realized_volatility",
        "trade_volumes_sum",
        "trade_order_count_sum",
        "trade_seconds_gap_mean",
        "trade_tendency",
        "trade_energy"
    ]

    for cluster_idx, ind in enumerate(kmeans_clusters):
        cluster_df = df.loc[df["stock_id"].isin(ind), agg_columns].groupby(["time_id"]).agg(function)
        cluster_df.loc[:, "stock_id"] = str(cluster_idx) + post_fix
        clusters.append(cluster_df)

    clusters_df = pd.concat(clusters).reset_index()
    # multi index (column, c1)
    clusters_df = clusters_df.pivot(index="time_id", columns="stock_id")
    # ravel multi index to list of tuple [(target, c1), ...]
    clusters_df.columns = ["_".join(x) for x in clusters_df.columns.ravel()]
    clusters_df.reset_index(inplace=True)

    postfixes = [
        "0" + post_fix,
        "1" + post_fix,
        "3" + post_fix,
        "4" + post_fix,
        "6" + post_fix,
    ]
    merge_columns = []
    for column in agg_columns:
        if column == "time_id":
            merge_columns.append(column)
        elif column == "stock_id":
            continue
        else:
            for postfix in postfixes:
                merge_columns.append(column + "_" + postfix)

    not_exist_columns = [column for column in merge_columns if column not in clusters_df.columns]
    clusters_df[not_exist_columns] = 0

    df = pd.merge(df, clusters_df[merge_columns], how="left", on="time_id")

    return df


# Function to get group stats for the time_id
def agg_stat_features_by_market(df, operations=None, operations_names=None):
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = "percentile_%s" % n
        return percentile_

    if operations is None:
        operations = [
            np.nanmean,
        ]
        operations_names = [
            "mean",
        ]

    # Get realized volatility columns
    vol_cols = [
        "log_return1_realized_volatility",
        "log_return1_realized_volatility_150",
        "log_return1_realized_volatility_300",
        "log_return1_realized_volatility_450",
    ]

    # Group by the stock id
    df_stock_id = df.groupby(["stock_id"])[vol_cols].agg(operations).reset_index()
    # Rename columns joining suffix
    df_stock_id.columns = ["_".join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix("_" + "stock")

    # Group by the stock id
    df_time_id = df.groupby(["time_id"])[vol_cols].agg(operations).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ["_".join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix("_" + "time")

    # Merge with original dataframe
    df = df.merge(df_stock_id, how="left", left_on=["stock_id"], right_on=["stock_id__stock"])
    df.drop("stock_id__stock", axis=1, inplace=True)

    df = df.merge(df_time_id, how="left", left_on=["time_id"], right_on=["time_id__time"])
    df.drop("time_id__time", axis=1, inplace=True)

    return df


def generate_interval_feature(df):

    # Get unique stock ids
    df_stock_ids = df["stock_id"].unique()

    # Preprocess them using Parallel and our single stock id functions
    df_ = preprocessor(df_stock_ids, is_train=True)
    df = df.merge(df_, on=["row_id"], how="left")

    # abs log columns
    abs_log_columns = [column for column in df.columns if
                       "order_flow_imbalance" in column or
                       "order_book_slope" in column or
                       "depth_imbalance" in column or
                       "pressure_imbalance" in column or
                       "total_volume" in column or
                       "seconds_gap" in column or
                       "trade_volumes" in column or
                       "trade_order_count" in column or
                       "trade_seconds_gap" in column or
                       "trade_tendency" in column
                       ]

    # apply abs + log1p
    df[abs_log_columns] = (df[abs_log_columns].apply(np.abs)).apply(np.log1p)

    return df
