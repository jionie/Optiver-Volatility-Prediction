import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore")
pd.set_option("max_columns", 300)


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


# Function to preprocess book data (for each stock id)
def book_preprocessor(file_path):
    df = pd.read_parquet(file_path)

    # float 64 to float 32
    float_cols = df.select_dtypes(include=[np.float64]).columns
    df[float_cols] = df[float_cols].astype(np.float32)

    # int 64 to int 32
    int_cols = df.select_dtypes(include=[np.int64]).columns
    df[int_cols] = df[int_cols].astype(np.int32)

    # Calculate seconds gap
    df["seconds_gap"] = df.groupby(["time_id"])["seconds_in_bucket"].apply(time_diff)

    # Calculate Wap
    df["wap1"] = calc_wap(df, pos=1)
    df["wap2"] = calc_wap(df, pos=2)
    df["wap3"] = calc_wap2(df, pos=1)
    df["wap4"] = calc_wap2(df, pos=2)

    # Calculate wap balance
    df["wap_balance"] = abs(df["wap1"] - df["wap2"])

    # Calculate log returns
    df["log_return1"] = df.groupby(["time_id"])["wap1"].apply(log_return)
    df["log_return2"] = df.groupby(["time_id"])["wap2"].apply(log_return)
    df["log_return3"] = df.groupby(["time_id"])["wap3"].apply(log_return)
    df["log_return4"] = df.groupby(["time_id"])["wap4"].apply(log_return)

    # Calculate spread
    df["bid_spread"] = df["bid_price1"] / df["bid_price2"] - 1
    df["ask_spread"] = df["ask_price2"] / df["ask_price1"] - 1
    df["bid_ask_spread1"] = df["ask_price1"] / df["bid_price1"] - 1
    df["bid_ask_spread2"] = df["ask_price2"] / df["bid_price2"] - 1

    # bid ask spread log return
    df["bid_ask_spread_log_return1"] = df.groupby(["time_id"])["bid_ask_spread1"].apply(log_return)
    df["bid_ask_spread_log_return2"] = df.groupby(["time_id"])["bid_ask_spread2"].apply(log_return)

    # order flow imbalance
    df["order_flow_imbalance1"] = order_flow_imbalance(df, 1)
    df["order_flow_imbalance2"] = order_flow_imbalance(df, 2)

    # order flow imbalance log return
    df["order_flow_imbalance_log_return1"] = df.groupby(["time_id"])["order_flow_imbalance1"].apply(log_return)
    df["order_flow_imbalance_log_return2"] = df.groupby(["time_id"])["order_flow_imbalance2"].apply(log_return)

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
        "wap3": [np.sum, np.std],
        "wap4": [np.sum, np.std],
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility],
        "log_return3": [realized_volatility],
        "log_return4": [realized_volatility],
        "wap_balance": [np.sum, np.max, np.min],
        "bid_spread": [np.sum, np.max, np.min],
        "ask_spread": [np.sum, np.max, np.min],
        "bid_ask_spread1": [np.sum, np.max, np.min],
        "bid_ask_spread2": [np.sum, np.max, np.min],
        "bid_ask_spread_log_return1": [realized_volatility],
        "bid_ask_spread_log_return2": [realized_volatility],
        "order_flow_imbalance1": [np.sum, np.max, np.min],
        "order_flow_imbalance2": [np.sum, np.max, np.min],
        "order_flow_imbalance_log_return1": [realized_volatility],
        "order_flow_imbalance_log_return2": [realized_volatility],
        "order_book_slope": [np.mean, np.max, np.min],
        "depth_imbalance1": [np.sum, np.max, np.min],
        "depth_imbalance2": [np.sum, np.max, np.min],
        "height_imbalance1": [np.sum, np.max, np.min],
        "height_imbalance2": [np.sum, np.max, np.min],
        "pressure_imbalance": [np.sum, np.max, np.min, np.std],
        "total_volume": [np.sum, np.max, np.min],
        "seconds_in_bucket": [count_unique],
        "seconds_gap": [np.mean, np.max, np.min]
    }
    create_feature_dict_time = {
        "log_return1": [realized_volatility],
        "log_return2": [realized_volatility],
        "log_return3": [realized_volatility],
        "log_return4": [realized_volatility],
        "bid_ask_spread_log_return1": [realized_volatility],
        "bid_ask_spread_log_return2": [realized_volatility],
        "order_flow_imbalance_log_return1": [realized_volatility],
        "order_flow_imbalance_log_return2": [realized_volatility],
        "total_volume": [np.sum, np.max, np.min],
        "seconds_gap": [np.mean, np.max, np.min]
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

    # Calculate seconds gap
    df["seconds_gap"] = df.groupby(["time_id"])["seconds_in_bucket"].apply(time_diff)

    # Calculate log return
    df["price_log_return"] = df.groupby("time_id")["price"].apply(log_return)

    # Calculate size log return
    df["size_log_return"] = df.groupby("time_id")["size"].apply(log_return)

    # Calculate volumes
    df["volumes"] = df["price"] * df["size"]
    df["volumes_log_return"] = df.groupby("time_id")["volumes"].apply(log_return)

    # Dict for aggregations
    create_feature_dict = {
        "price_log_return": [realized_volatility],
        "size_log_return": [realized_volatility],
        "volumes_log_return": [realized_volatility],
        "volumes": [np.sum, np.max, np.min],
        "size": [np.sum, np.max, np.min],
        "order_count": [np.sum, np.max, np.min],
        "seconds_in_bucket": [count_unique],
        "seconds_gap": [np.mean, np.max, np.min, np.std]
    }
    create_feature_dict_time = {
        "price_log_return": [realized_volatility],
        "size_log_return": [realized_volatility],
        "volumes_log_return": [realized_volatility],
        "size": [np.sum, np.max, np.min],
        "order_count": [np.sum, np.max, np.min],
        "seconds_in_bucket": [count_unique],
        "seconds_gap": [np.mean, np.max, np.min, np.std]
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
        f_max = np.sum(df_id["price"].values > np.mean(df_id["price"].values))
        f_min = np.sum(df_id["price"].values < np.mean(df_id["price"].values))
        df_max = np.sum(np.diff(df_id["price"].values) > 0)
        df_min = np.sum(np.diff(df_id["price"].values) < 0)
        # new
        abs_diff = np.median(np.abs(df_id["price"].values - np.mean(df_id["price"].values)))
        energy = np.mean(df_id["price"].values ** 2)
        iqr_p = np.percentile(df_id["price"].values, 75) - np.percentile(df_id["price"].values, 25)

        # vol vars
        abs_diff_v = np.median(np.abs(df_id["size"].values - np.mean(df_id["size"].values)))
        energy_v = np.sum(df_id["size"].values ** 2)
        iqr_p_v = np.percentile(df_id["size"].values, 75) - np.percentile(df_id["size"].values, 25)

        lis.append(
            {
                "time_id": n_time_id,
                "tendency": tendencyV,
                "f_max": f_max,
                "f_min": f_min,
                "df_max": df_max,
                "df_min": df_min,
                "abs_diff": abs_diff,
                "energy": energy,
                "iqr_p": iqr_p,
                "abs_diff_v": abs_diff_v,
                "energy_v": energy_v,
                "iqr_p_v": iqr_p_v
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


# Process size tau
def process_size_tau(df, windows=None):
    if windows is None:
        windows = [0, 150, 300, 450]

    for window in windows:
        if window == 0:
            df["size_tau"] = np.sqrt(1 / df["trade_seconds_in_bucket_count_unique"])
            df["size_tau2"] = np.sqrt(1 / df["trade_order_count_sum"])
        else:
            df["size_tau_{}".format(window)] = np.sqrt(1 / df["trade_seconds_in_bucket_count_unique_{}".format(window)])
            df["size_tau2_{}".format(window)] = np.sqrt(
                (1 - window / 600) / df["trade_order_count_sum_{}".format(window)])

    return df


# Process agg by kmeans
def get_kmeans_idx(n_clusters=7, train_path="../input/optiver-realized-volatility-prediction/train.csv"):
    train_p = pd.read_csv(train_path)
    train_p = train_p.pivot(index="time_id", columns="stock_id", values="target")

    corr = train_p.corr()

    ids = corr.index

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(corr.values)

    kmeans_clusters = []
    for n in range(n_clusters):
        kmeans_clusters.append([(x - 1) for x in ((ids + 1) * (kmeans.labels_ == n)) if x > 0])

    return kmeans_clusters


def agg_mean_features_by_clusters(df, n_clusters=7, post_fix="_cluster_mean"):

    kmeans_clusters = get_kmeans_idx(n_clusters=n_clusters)

    clusters = []
    agg_columns = [
        "time_id",
        "stock_id",
        "log_return1_realized_volatility",
        "log_return2_realized_volatility",
        "log_return3_realized_volatility",
        "log_return4_realized_volatility",
        "bid_ask_spread_log_return1_realized_volatility",
        "bid_ask_spread_log_return2_realized_volatility",
        "order_flow_imbalance_log_return1_realized_volatility",
        "order_flow_imbalance_log_return2_realized_volatility",
        "order_book_slope_mean",
        "order_book_slope_amax",
        "order_book_slope_amin",
        "total_volume_sum",
        "total_volume_amax",
        "total_volume_amin",
        "seconds_gap_mean",
        "seconds_gap_amax",
        "seconds_gap_amin",
        "trade_size_sum",
        "trade_size_amax",
        "trade_size_amin",
        "trade_order_count_sum",
        "trade_order_count_amax",
        "trade_order_count_amin",
        "trade_seconds_gap_mean",
        "trade_seconds_gap_amax",
        "trade_seconds_gap_amin",
        "size_tau2"
    ]

    for cluster_idx, ind in enumerate(kmeans_clusters):
        cluster_df = df.loc[df["stock_id"].isin(ind), agg_columns].groupby(["time_id"]).agg(np.nanmean)
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
    df = pd.merge(df, clusters_df[merge_columns], how="left", on="time_id")

    return df


def agg_rank_features_by_clusters(df, n_clusters=7, post_fix="_cluster_rank"):

    kmeans_clusters = get_kmeans_idx(n_clusters=n_clusters)
    agg_columns = [
        "log_return1_realized_volatility",
        "order_flow_imbalance_log_return1_realized_volatility",
        "total_volume_sum",
        "trade_size_sum",
        "trade_order_count_sum",
        "bid_ask_spread1_sum",
        "size_tau2"
    ]

    def get_clusters_id(value, kmeans_clusters):
        for cluster_idx, ind in enumerate(kmeans_clusters):
            if value in ind:
                return cluster_idx

    df["stock_cluster_idx"] = df["stock_id"].apply(get_clusters_id, args=(kmeans_clusters,))

    for cluster_idx, ind in enumerate(kmeans_clusters):
        agg_columns_with_postfix = [column + str(cluster_idx) + "_" + post_fix for column in agg_columns]
        df[agg_columns_with_postfix] = df.groupby(["time_id", "stock_cluster_idx"])[agg_columns].rank(pct=True)

    df.drop("stock_cluster_idx", axis=1, inplace=True)
    return df


# Function to get group stats for the stock_id and time_id
def get_time_stock(df, operations=None, operations_names=None):
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)

        percentile_.__name__ = "percentile_%s" % n
        return percentile_

    if operations is None:
        operations = [
            np.mean,
            np.std,
            np.min,
            np.max,
            # percentile(10),
            # percentile(30),
            # percentile(50),
            # percentile(70),
            # percentile(90),
        ]
        operations_names = [
            "mean",
            "std",
            "amin",
            "amax",
            # "percentile_10",
            # "percentile_30",
            # "percentile_50",
            # "percentile_70",
            # "percentile_90",
        ]

    # Get realized volatility columns
    vol_cols = [
        "log_return1_realized_volatility",
        "log_return1_realized_volatility_150",
        "log_return1_realized_volatility_300",
        "log_return1_realized_volatility_450",
        "order_flow_imbalance_log_return1_realized_volatility",
        "order_flow_imbalance_log_return1_realized_volatility_150",
        "order_flow_imbalance_log_return1_realized_volatility_300",
        "order_flow_imbalance_log_return1_realized_volatility_450",
        "bid_ask_spread_log_return1_realized_volatility",
        "bid_ask_spread_log_return1_realized_volatility_150",
        "bid_ask_spread_log_return1_realized_volatility_300",
        "bid_ask_spread_log_return1_realized_volatility_450",
        "seconds_gap_mean",
        "seconds_gap_mean_150",
        "seconds_gap_mean_300",
        "seconds_gap_mean_450",
        "trade_price_log_return_realized_volatility",
        "trade_price_log_return_realized_volatility_150",
        "trade_price_log_return_realized_volatility_300",
        "trade_price_log_return_realized_volatility_450",
        "total_volume_sum",
        "total_volume_sum_150",
        "total_volume_sum_300",
        "total_volume_sum_450",
        "trade_size_sum",
        "trade_size_sum_150",
        "trade_size_sum_300",
        "trade_size_sum_450",
        "trade_seconds_gap_mean",
        "trade_seconds_gap_mean_150",
        "trade_seconds_gap_mean_300",
        "trade_seconds_gap_mean_450",
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
