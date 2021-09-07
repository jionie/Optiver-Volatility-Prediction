from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import RFE
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")
pd.set_option("max_columns", 300)

from utils.fe import book_preprocessor, trade_preprocessor, get_time_stock, agg_mean_features_by_clusters, \
    process_size_tau


CONFIG = {
    "root_dir": "../input/optiver-realized-volatility-prediction/",
    "kfold_seed": 42,
    "n_splits": 5,
    "n_clusters": 7,
}

PARAMS = {
    "objective": "rmse",
    "boosting_type": "gbdt",
    "max_depth": -1,
    "max_bin": 100,
    "min_data_in_leaf": 500,
    "learning_rate": 0.05,
    "subsample": 0.72,
    "subsample_freq": 4,
    "feature_fraction": 0.5,
    "lambda_l1": 0.5,
    "lambda_l2": 1.0,
    "categorical_column": [0],
    "seed": 2021,
    "feature_fraction_seed": 2021,
    "bagging_seed": 2021,
    "drop_seed": 2021,
    "data_random_seed": 2021,
    "n_jobs": -1,
    "verbose": -1
}


# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


# Function to early stop with root mean squared percentage error
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return "RMSPE", rmspe(y_true, y_pred), False


# Function to early stop with root mean squared percentage error
def eval_rmspe(y_pred, y_true):
    return "RMSPE", rmspe(y_true, y_pred), False


def read_train_test():
    train = pd.read_csv("../input/optiver-realized-volatility-prediction/train.csv")
    test = pd.read_csv("../input/optiver-realized-volatility-prediction/test.csv")

    # Create a key to merge with book and trade data
    train["row_id"] = train["stock_id"].astype(str) + "-" + train["time_id"].astype(str)
    test["row_id"] = test["stock_id"].astype(str) + "-" + test["time_id"].astype(str)

    print("Our training set has {} rows".format(train.shape[0]))

    return train, test


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


def generate_fold_data(train):

    # Split features and target
    x = train.drop(["row_id", "target"], axis=1)
    y = train["target"]

    # Transform stock id to a numeric value
    x["stock_id"] = x["stock_id"].astype(int)

    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])

    # Create a KFold object
    kfold = GroupKFold(n_splits=CONFIG["n_splits"])
    x_ref = get_time_stock(x.copy(deep=True))
    x_ref = agg_mean_features_by_clusters(x_ref, n_clusters=CONFIG["n_clusters"])

    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x, groups=x["time_id"])):

        print(f"Training fold {fold + 1}")
        x_train = x.iloc[trn_ind].copy(deep=True)
        x_train = get_time_stock(x_train)
        x_train = agg_mean_features_by_clusters(x_train, n_clusters=CONFIG["n_clusters"])

        x_val = x_ref.iloc[val_ind]

        x_train.drop("time_id", axis=1, inplace=True)
        x_val.drop("time_id", axis=1, inplace=True)

        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

        # save fold data
        x_train.to_csv("x_train_fold_{}.csv".format(fold + 1), index=False)
        x_val.to_csv("x_val_fold_{}.csv".format(fold + 1), index=False)
        y_train.to_csv("y_train_fold_{}.csv".format(fold + 1), index=False)
        y_val.to_csv("y_val_fold_{}.csv".format(fold + 1), index=False)

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight=train_weights, categorical_feature=["stock_id"])
        val_dataset = lgb.Dataset(x_val, y_val, weight=val_weights, categorical_feature=["stock_id"])

        # Train
        model = lgb.train(params=PARAMS,
                          train_set=train_dataset,
                          valid_sets=[train_dataset, val_dataset],
                          num_boost_round=6000,
                          early_stopping_rounds=300,
                          verbose_eval=100,
                          feval=feval_rmspe
                          )

        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val)

    rmspe_score = rmspe(y, oof_predictions)
    print(f"Our out of folds RMSPE is {rmspe_score}")

    return


def rfe():

    # sort single feature and their score
    feature_score_map = {}

    for fold in range(CONFIG["n_splits"]):

        print("loading fold {} data".format(fold + 1))
        x_train = pd.read_csv("x_train_fold_{}.csv".format(fold + 1))
        x_val = pd.read_csv("x_val_fold_{}.csv".format(fold + 1))
        y_train = pd.read_csv("y_train_fold_{}.csv".format(fold + 1)).squeeze()
        y_val = pd.read_csv("y_val_fold_{}.csv".format(fold + 1)).squeeze()

        train_columns = x_train.columns

        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)

        model = lgb.LGBMRegressor(**PARAMS, num_boost_round=6000)
        model.fit(
            X=x_train,
            y=y_train,
            sample_weight=train_weights,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_sample_weight=[train_weights, val_weights],
            eval_metric=eval_rmspe,
            categorical_feature=["stock_id"],
            early_stopping_rounds=300,
            verbose=100,
        )
        selector = RFE(model, n_features_to_select=400, step=30).fit(x_val, y_val)
        ranking = selector.ranking_

        for idx, column in enumerate(train_columns):

            if column not in feature_score_map:
                feature_score_map[column] = ranking[idx]
            else:
                feature_score_map[column] += ranking[idx]

    # sort feature_score_map
    feature_score_map = {k: v for k, v in sorted(feature_score_map.items(), key=lambda item: item[1])}

    return feature_score_map


def main():

    # # Read train and test
    # train, _ = read_train_test()
    #
    # # Get unique stock ids
    # train_stock_ids = train["stock_id"].unique()
    #
    # # Preprocess them using Parallel and our single stock id functions
    # train_ = preprocessor(train_stock_ids, is_train=True)
    # train = train.merge(train_, on=["row_id"], how="left")
    #
    # # add tau features
    # windows = [0, 150, 300, 450]
    # train = process_size_tau(train, windows)
    #
    # # Scaler
    # except_columns = ["stock_id", "time_id", "target", "row_id"]
    # normalized_columns = [column for column in train.columns if column not in except_columns]
    #
    # # scaler = StandardScaler()
    # # train[normalized_columns] = scaler.fit_transform(train[normalized_columns])
    #
    # qt = QuantileTransformer(n_quantiles=2000, random_state=2021)
    # train[normalized_columns] = qt.fit_transform(train[normalized_columns])
    #
    # # generate fold data
    # generate_fold_data(train)

    # run permutation importance
    feature_score_map = rfe()
    print(feature_score_map)

    # save results
    with open("recursive_feature.json", "w") as f:
        json.dump(feature_score_map, f, ensure_ascii=False, indent=4)

    return


if __name__ == "__main__":
    main()

