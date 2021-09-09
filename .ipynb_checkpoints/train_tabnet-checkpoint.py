import os
from joblib import Parallel, delayed, dump, load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, QuantileTransformer, LabelEncoder
from sklearn.model_selection import GroupKFold
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings

warnings.filterwarnings("ignore")
pd.set_option("max_columns", 300)

from utils.fe import book_preprocessor, trade_preprocessor, agg_stat_features_by_market, agg_stat_features_by_clusters

CONFIG = {
    "root_dir": "../input/optiver-realized-volatility-prediction/",
    "ckpt_path": "../../ckpts/",
    "kfold_seed": 42,
    "n_splits": 5,
    "n_clusters": 7,
}


# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))


class RMSPE(Metric):
    def __init__(self):
        self._name = "rmspe"
        self._maximize = False

    def __call__(self, y_true, y_score):
        return np.sqrt(np.mean(np.square((y_true - y_score) / y_true)))


def RMSPELoss(y_pred, y_true):
    return torch.sqrt(torch.mean(((y_true - y_pred) / y_true) ** 2)).clone()


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

    # label encoder
    cat_columns = ["stock_id"]
    label_encoder = LabelEncoder()
    train[cat_columns] = label_encoder.fit_transform(train[cat_columns].values)
    test[cat_columns] = label_encoder.transform(test[cat_columns].values)
    cat_dims = [len(label_encoder.classes_)]

    # scale
    # scaler = QuantileTransformer(n_quantiles=2000, random_state=2021)
    scaler = StandardScaler()

    # Split features and target
    x = train.drop(["row_id", "target"], axis=1)
    y = train["target"]

    # x_test with train feature
    x_test = test.drop("row_id", axis=1)
    x_test = agg_stat_features_by_market(x_test)
    x_test = agg_stat_features_by_clusters(x_test, n_clusters=CONFIG["n_clusters"], function=np.nanmean,
                                           post_fix="_cluster_mean")
    x_test = agg_stat_features_by_clusters(x_test, n_clusters=CONFIG["n_clusters"], function=np.nanmax,
                                           post_fix="_cluster_max")
    x_test = agg_stat_features_by_clusters(x_test, n_clusters=CONFIG["n_clusters"], function=np.nanmin,
                                           post_fix="_cluster_min")
    x_test = agg_stat_features_by_clusters(x_test, n_clusters=CONFIG["n_clusters"], function=np.nanstd,
                                           post_fix="_cluster_std")

    # define normalize columns
    except_columns = ["stock_id", "time_id", "target", "row_id"]
    normalized_columns = [column for column in x_test.columns if column not in except_columns]
    x_test.drop("time_id", axis=1, inplace=True)

    # Process categorical features and get params dict
    cat_idxs = [i for i, f in enumerate(x_test.columns.tolist()) if f in cat_columns]

    params = dict(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=1,
        n_d=16,
        n_a=16,
        n_steps=2,
        gamma=2,
        n_independent=2,
        n_shared=2,
        lambda_sparse=0,
        optimizer_fn=Adam,
        optimizer_params=dict(lr=(2e-2)),
        mask_type="entmax",
        scheduler_params=dict(T_0=200, T_mult=1, eta_min=1e-4, last_epoch=-1, verbose=False),
        scheduler_fn=CosineAnnealingWarmRestarts,
        seed=42,
        verbose=10
    )

    # Create out of folds array
    oof_predictions = np.zeros(x.shape[0])

    # Create test array to store predictions
    test_predictions = np.zeros(x_test.shape[0])

    # Statistics
    feature_importances = pd.DataFrame()
    feature_importances["feature"] = x_test.columns.tolist()
    stats = pd.DataFrame()
    explain_matrices = []
    masks_ = []

    # Create a KFold object
    kfold = GroupKFold(n_splits=CONFIG["n_splits"])

    # Iterate through each fold
    for fold, (trn_ind, val_ind) in enumerate(kfold.split(x, groups=x["time_id"])):

        print(f"Training fold {fold + 1}")
        x_train = x.iloc[trn_ind]
        x_train = agg_stat_features_by_market(x_train)
        x_train = agg_stat_features_by_clusters(x_train, n_clusters=CONFIG["n_clusters"], function=np.nanmean,
                                                post_fix="_cluster_mean")
        x_train = agg_stat_features_by_clusters(x_train, n_clusters=CONFIG["n_clusters"], function=np.nanmax,
                                                post_fix="_cluster_max")
        x_train = agg_stat_features_by_clusters(x_train, n_clusters=CONFIG["n_clusters"], function=np.nanmin,
                                                post_fix="_cluster_min")
        x_train = agg_stat_features_by_clusters(x_train, n_clusters=CONFIG["n_clusters"], function=np.nanstd,
                                                post_fix="_cluster_std")
        x_train.drop("time_id", axis=1, inplace=True)
        scaler = scaler.fit(x_train[normalized_columns])
        dump(scaler, os.path.join(CONFIG["ckpt_dir"], "std_scaler_fold_{}.bin".format(fold + 1)), compress=True)
        x_train[normalized_columns] = scaler.transform(x_train[normalized_columns])

        x_val = x.iloc[val_ind]
        x_val = agg_stat_features_by_market(x_val)
        x_val = agg_stat_features_by_clusters(x_val, n_clusters=CONFIG["n_clusters"], function=np.nanmean,
                                              post_fix="_cluster_mean")
        x_val = agg_stat_features_by_clusters(x_val, n_clusters=CONFIG["n_clusters"], function=np.nanmax,
                                              post_fix="_cluster_max")
        x_val = agg_stat_features_by_clusters(x_val, n_clusters=CONFIG["n_clusters"], function=np.nanmin,
                                              post_fix="_cluster_min")
        x_val = agg_stat_features_by_clusters(x_val, n_clusters=CONFIG["n_clusters"], function=np.nanstd,
                                              post_fix="_cluster_std")
        x_val.drop("time_id", axis=1, inplace=True)
        x_val[normalized_columns] = scaler.transform(x_val[normalized_columns])

        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]

        # Train
        clf = TabNetRegressor(**params)
        clf.fit(
            x_train.values, y_train,
            eval_set=[(x_val.values, y_val)],
            max_epochs=200,
            patience=50,
            batch_size=1024 * 20,
            virtual_batch_size=128 * 20,
            num_workers=0,
            drop_last=False,
            eval_metric=[RMSPE],
            loss_fn=RMSPELoss
        )

        # save model
        saved_filepath = clf.save_model(os.path.join(CONFIG["ckpt_path"], "tabnet_fold{}".format(fold + 1)))

        # save statistics
        explain_matrix, masks = clf.explain(x_val.values)
        explain_matrices.append(explain_matrix)
        masks_.append(masks[0])
        masks_.append(masks[1])

        feature_importances["importance_fold{}".format(fold + 1)] = clf.feature_importances_
        stats["fold{}_train_rmspe".format(fold + 1)] = clf.history["loss"]
        stats["fold{}_val_rmspe".format(fold + 1)] = clf.history["val_0_rmspe"]

        # save oof and test predictions
        oof_predictions[val_ind] = clf.predict(x_val.values).flatten()
        x_test_ = x_test.copy()
        x_test_[normalized_columns] = scaler.transform(x_test_[normalized_columns])
        test_predictions += clf.predict(x_test_.values).flatten() / CONFIG["n_splits"]

    rmspe_score = rmspe(y, oof_predictions)
    print("Our out of folds RMSPE is {}".format(rmspe_score))

    # Return test predictions
    return test_predictions, stats, feature_importances, explain_matrices, masks_


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

    # abs log columns
    abs_log_columns = [column for column in train.columns if
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
    train[abs_log_columns] = (train[abs_log_columns].apply(np.abs)).apply(np.log1p)
    test[abs_log_columns] = (test[abs_log_columns].apply(np.abs)).apply(np.log1p)

    # fill inf with nan
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)

    train = train.fillna(train.mean())
    test = test.fillna(train.mean())

    test_predictions, stats, feature_importances, explain_matrices, masks_ = train_and_evaluate(train, test)


if __name__ == "__main__":
    main()
