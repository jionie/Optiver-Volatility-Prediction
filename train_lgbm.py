from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import GroupKFold
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


def train_and_evaluate(train, test):

    # Split features and target
    x = train.drop(["row_id", "target"], axis=1)
    y = train["target"]

    # x_test with train feature
    x_test_ = test.drop("row_id", axis=1)
    x_test_ref = pd.concat([x, x_test_], axis=0)
    x_test_ref = get_time_stock(x_test_ref)
    x_test_ref = agg_mean_features_by_clusters(x_test_ref, n_clusters=CONFIG["n_clusters"])
    # x_test_ref = agg_rank_features_by_clusters(x_test_ref, n_clusters=CONFIG["n_clusters"])
    x_test = x_test_ref.iloc[train.shape[0]:]
    x_test.drop("time_id", axis=1, inplace=True)

    # Transform stock id to a numeric value
    x["stock_id"] = x["stock_id"].astype(int)
    x_test["stock_id"] = x_test["stock_id"].astype(int)

    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])

    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])

    # Create a KFold object
    kfold = GroupKFold(n_splits=CONFIG["n_splits"])
    x_ref = get_time_stock(x.copy(deep=True))
    x_ref = agg_mean_features_by_clusters(x_ref, n_clusters=CONFIG["n_clusters"])
    # x_ref = agg_rank_features_by_clusters(x_ref, n_clusters=CONFIG["n_clusters"])

    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x, groups=x["time_id"])):
        print(f"Training fold {fold + 1}")
        x_train = x.iloc[trn_ind].copy(deep=True)
        x_train = get_time_stock(x_train)
        x_train = agg_mean_features_by_clusters(x_train, n_clusters=CONFIG["n_clusters"])
        # x_train = agg_rank_features_by_clusters(x_train, n_clusters=CONFIG["n_clusters"])

        x_val = x_ref.iloc[val_ind]

        x_train.drop("time_id", axis=1, inplace=True)
        x_val.drop("time_id", axis=1, inplace=True)

        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

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

        # Feature Importance
        fig, ax = plt.subplots(figsize=(12, 30))
        lgb.plot_importance(model, max_num_features=50, ax=ax)
        plt.title("Feature importance")
        plt.show()
        fig.savefig("fold_{}.png".format(fold + 1))
        plt.close(fig)

        # Add predictions to the out of folds array
        oof_predictions[val_ind] = model.predict(x_val)

        # Predict the test set
        test_predictions += model.predict(x_test) / CONFIG["n_splits"]

    rmspe_score = rmspe(y, oof_predictions)
    print(f"Our out of folds RMSPE is {rmspe_score}")

    # Return test predictions
    return test_predictions


def main():

    # Read train and test
    train, test = read_train_test()

    # Get unique stock ids
    train_stock_ids = train["stock_id"].unique()

    # Preprocess them using Parallel and our single stock id functions
    train_ = preprocessor(train_stock_ids, is_train=True)
    train = train.merge(train_, on=["row_id"], how="left")

    # Get unique stock ids
    test_stock_ids = test["stock_id"].unique()

    # Preprocess them using Parallel and our single stock id functions
    test_ = preprocessor(test_stock_ids, is_train=False)
    test = test.merge(test_, on=["row_id"], how="left")

    # add tau features
    windows = [0, 150, 300, 450]
    train = process_size_tau(train, windows)
    test = process_size_tau(test, windows)

    # save train test
    train.to_csv("processed_train.csv", index=False)
    test.to_csv("processed_test.csv", index=False)

    # Scaler
    except_columns = ["stock_id", "time_id", "target", "row_id"]
    normalized_columns = [column for column in train.columns if column not in except_columns]

    # scaler = StandardScaler()
    # train[normalized_columns] = scaler.fit_transform(train[normalized_columns])
    # test[normalized_columns] = scaler.fit_transform(test[normalized_columns])

    qt = QuantileTransformer(n_quantiles=2000, random_state=2021)
    train[normalized_columns] = qt.fit_transform(train[normalized_columns])
    test[normalized_columns] = qt.fit_transform(test[normalized_columns])

    test_predictions = train_and_evaluate(train, test)


if __name__ == "__main__":
    main()
