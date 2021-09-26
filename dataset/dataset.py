import os
import gc
from tqdm import tqdm
from joblib import dump, load, Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset

from dataset.generate_feature import generate_interval_feature, agg_stat_features_by_market, \
    agg_stat_features_by_clusters, calc_wap, order_flow_imbalance, depth_imbalance, height_imbalance


class QuantDataset(Dataset):
    def __init__(self, config, df, sample_indices, order_books, trade_books, mode="train", fold=0, parallel=False):

        self.config = config
        self.parallel = parallel

        # load data df
        self.mode = mode
        if mode == "train" or mode == "val":
            other_df = pd.read_csv(self.config.test_data)
            full_df = pd.concat([df, other_df], axis=0)

            del other_df
            gc.collect()

        else:
            other_df = pd.read_csv(self.config.train_data)
            full_df = pd.concat([df, other_df], axis=0)

            del other_df
            gc.collect()

        self.sample_indices = sample_indices
        self.valid_time_ids = list(df["time_id"].iloc[self.sample_indices].unique())
        self.all_time_ids = list(df["time_id"].unique())

        self.seq_len = self.config.hist_window

        # store stock_id and time_id
        self.stock_ids = df.iloc[sample_indices]["stock_id"].values
        self.time_ids = df.iloc[sample_indices]["time_id"].values

        # cont_x init
        self.order_books = order_books
        self.trade_books = trade_books
        self.cont_cols = self.config.cont_cols

        # cont_x minmax normalization
        if mode == "train":

            minmax_stat_path = "../ckpts/{}_minmax_stat_fold_{}.pkl".format(self.config.model_type, fold + 1)

            # store or load minmax stat
            if os.path.exists(minmax_stat_path):
                self.minmax_stat = load(minmax_stat_path)
            else:

                print("generating minmax statistics")
                self.minmax_stat = {}

                order_min = None
                order_max = None
                trade_min = None
                trade_max = None

                for idx in range(len(sample_indices)):
                    stock_id = self.stock_ids[idx]
                    time_id = self.time_ids[idx]

                    try:
                        order_x = self.order_books[stock_id][time_id]
                        order_x[order_x == -np.inf] = np.nan
                        order_x[order_x == np.inf] = np.nan

                        if order_min is None:
                            order_min = np.nanmin(order_x, axis=0)
                        else:
                            order_min = np.fmin(order_min, np.nanmin(order_x, axis=0))

                        if order_max is None:
                            order_max = np.nanmax(order_x, axis=0)
                        else:
                            order_max = np.fmax(order_max, np.nanmax(order_x, axis=0))
                    except Exception as e:
                        pass

                    try:
                        trade_x = self.trade_books[stock_id][time_id]
                        trade_x[trade_x == -np.inf] = np.nan
                        trade_x[trade_x == np.inf] = np.nan

                        if trade_min is None:
                            trade_min = np.nanmin(trade_x, axis=0)
                        else:
                            trade_min = np.fmin(trade_min, np.nanmin(trade_x, axis=0))

                        if trade_max is None:
                            trade_max = np.nanmax(trade_x, axis=0)
                        else:
                            trade_max = np.fmax(trade_max, np.nanmax(trade_x, axis=0))
                    except Exception as e:
                        pass

                self.minmax_stat["order_min"] = order_min
                self.minmax_stat["order_max"] = order_max
                self.minmax_stat["trade_min"] = trade_min
                self.minmax_stat["trade_max"] = trade_max

                dump(self.minmax_stat, minmax_stat_path)

            # minmax normalization
            print("normalizing by minmax statistics")
            for idx in range(len(sample_indices)):
                stock_id = self.stock_ids[idx]
                time_id = self.time_ids[idx]

                try:
                    self.order_books[stock_id][time_id] = \
                        np.clip((self.order_books[stock_id][time_id] - self.minmax_stat["order_min"]) / \
                                (self.minmax_stat["order_max"] - self.minmax_stat["order_min"]) * 2 - 1, -1, 1)

                except Exception as e:
                    pass

                try:
                    self.trade_books[stock_id][time_id] = \
                        np.clip((self.trade_books[stock_id][time_id] - self.minmax_stat["trade_min"]) / \
                                (self.minmax_stat["trade_max"] - self.minmax_stat["trade_min"]) * 2 - 1, -1, 1)

                except Exception as e:
                    pass

        else:
            minmax_stat_path = "../ckpts/{}_minmax_stat_fold_{}.pkl".format(self.config.model_type, fold + 1)
            self.minmax_stat = load(minmax_stat_path)

            # minmax normalization
            print("normalizing by minmax statistics")
            for idx in range(len(sample_indices)):
                stock_id = self.stock_ids[idx]
                time_id = self.time_ids[idx]

                try:
                    self.order_books[stock_id][time_id] = \
                        np.clip((self.order_books[stock_id][time_id] - self.minmax_stat["order_min"]) / \
                                (self.minmax_stat["order_max"] - self.minmax_stat["order_min"]) * 2 - 1, -1, 1)

                except Exception as e:
                    pass

                try:
                    self.trade_books[stock_id][time_id] = \
                        np.clip((self.trade_books[stock_id][time_id] - self.minmax_stat["trade_min"]) / \
                                (self.minmax_stat["trade_max"] - self.minmax_stat["trade_min"]) * 2 - 1, -1, 1)

                except Exception as e:
                    pass

        # cate_x init
        self.cate_cols = self.config.cate_cols
        # label encoder transformation
        full_df[self.cate_cols] = full_df[self.cate_cols].apply(LabelEncoder().fit_transform)
        df[self.cate_cols] = full_df.iloc[:df.shape[0]][self.cate_cols].astype(np.int32)

        del full_df
        gc.collect()
        self.cate_x = df.iloc[sample_indices][self.cate_cols].values

        # target
        if mode == "train" or mode == "val":
            self.target = df.iloc[sample_indices][self.config.target_cols].values
        elif mode == "test":
            self.target = None
        else:
            raise NotImplementedError

        # generate agg features then normalization
        df = df.iloc[sample_indices]

        df = agg_stat_features_by_market(df)
        df = agg_stat_features_by_clusters(df, n_clusters=self.config.n_clusters, function=np.nanmean,
                                           post_fix="_cluster_mean")
        df = agg_stat_features_by_clusters(df, n_clusters=self.config.n_clusters, function=np.nanmax,
                                           post_fix="_cluster_max")
        df = agg_stat_features_by_clusters(df, n_clusters=self.config.n_clusters, function=np.nanmin,
                                           post_fix="_cluster_min")
        df = agg_stat_features_by_clusters(df, n_clusters=self.config.n_clusters, function=np.nanstd,
                                           post_fix="_cluster_std")

        except_columns = ["stock_id", "time_id", "target", "row_id"]
        self.extra_cols = [column for column in df.columns if column not in except_columns]
        df = df.replace([np.inf, -np.inf], np.nan)

        if self.mode == "train":
            scaler = StandardScaler()
            scaler = scaler.fit(df[self.extra_cols])

            dump(scaler, "../ckpts/{}_std_scaler_fold_{}.bin".format(self.config.model_type, fold + 1), compress=True)

            df[self.extra_cols] = df[self.extra_cols].fillna(df[self.extra_cols].mean())
            df[self.extra_cols] = scaler.transform(df[self.extra_cols])
        else:
            scaler = load("../ckpts/{}_std_scaler_fold_{}.bin".format(self.config.model_type, fold + 1))

            fill_value = scaler.mean_
            fill_value_dict = {self.extra_cols[idx]: fill_value[idx] for idx in range(len(self.extra_cols))}

            df[self.extra_cols] = df[self.extra_cols].fillna(fill_value_dict)
            df[self.extra_cols] = scaler.transform(df[self.extra_cols])

        # fill inf and na
        self.extra_x = df[self.extra_cols].values

        del df
        gc.collect()

    def __getitem__(self, idx):

        # continuous features
        order_x = self.order_books[self.stock_ids[idx]][self.time_ids[idx]]
        order_x = torch.nan_to_num(torch.from_numpy(order_x), nan=0.0, posinf=1, neginf=-1)

        try:
            trade_x = self.trade_books[self.stock_ids[idx]][self.time_ids[idx]]
            trade_x = torch.nan_to_num(torch.from_numpy(trade_x), nan=0.0, posinf=1, neginf=-1)

        except Exception as e:
            trade_x = torch.zeros((self.seq_len, len(self.config.trade_features)))

        cont_x = torch.cat([order_x, trade_x], dim=1)

        # category feature
        cate_x = torch.repeat_interleave(torch.from_numpy(self.cate_x[idx].astype(np.int32)).unsqueeze(0),
                                         self.seq_len, dim=0)

        # mask
        mask = torch.from_numpy(np.ones(self.seq_len))
        # should we use relative target gain???
        if self.mode == "test":
            target = torch.ones(len(self.config.target_cols))
        else:
            target = torch.from_numpy(self.target[idx] * self.config.target_scale)

        # extra features for ffnn
        extra_x = torch.from_numpy(self.extra_x[idx])

        return cate_x, cont_x, extra_x, mask, target

    def __len__(self):
        return len(self.sample_indices)


def load_data(config, mode="train", parallel=False):
    if mode == "train" or mode == "val":
        order_path = config.train_order
        trade_path = config.train_trade

    elif mode == "test":
        order_path = config.test_order
        trade_path = config.test_trade

    else:
        raise NotImplementedError

    # load order book
    order_books_path = os.path.join(config.data_dir, "order_book_{}.pkl".format(mode))

    if os.path.exists(order_books_path):
        order_books = load(order_books_path)

    else:

        order_books = dict()
        for path in tqdm(order_path):

            stock_id = int(path.replace("\\", "/").split("=")[1].split("/")[0])
            order_df = pd.read_parquet(path)

            # fe
            order_df["wap1"] = calc_wap(order_df, pos=1)
            order_df["wap2"] = calc_wap(order_df, pos=2)

            order_df["bid_ask_spread1"] = order_df["ask_price1"] / order_df["bid_price1"] - 1
            order_df["bid_ask_spread2"] = order_df["ask_price2"] / order_df["bid_price2"] - 1

            order_df["order_flow_imbalance1"] = order_flow_imbalance(order_df, 1)
            order_df["order_flow_imbalance2"] = order_flow_imbalance(order_df, 2)

            order_df["depth_imbalance1"] = depth_imbalance(order_df, pos=1)
            order_df["depth_imbalance2"] = depth_imbalance(order_df, pos=2)

            order_df["height_imbalance1"] = height_imbalance(order_df, pos=1)
            order_df["height_imbalance2"] = height_imbalance(order_df, pos=2)

            order_df = order_df.drop(["bid_price1", "ask_price1", "bid_price2", "ask_price2", "bid_size1", "ask_size1",
                                      "bid_size2", "ask_size2"], axis=1)

            # float 64 to float 32
            float_cols = order_df.select_dtypes(include=[np.float64]).columns
            order_df[float_cols] = order_df[float_cols].astype(np.float32)

            # int 64 to int 32
            int_cols = order_df.select_dtypes(include=[np.int64]).columns
            order_df[int_cols] = order_df[int_cols].astype(np.int32)

            if parallel:

                def fill_order(time_id):

                    df_ = order_df[order_df["time_id"] == time_id].reset_index(drop=True)
                    filled_df_ = pd.DataFrame({"seconds_in_bucket": range(config.hist_window)})
                    filled_df_ = pd.merge(filled_df_, df_, on=["seconds_in_bucket"], how="left")
                    filled_df_ = filled_df_.fillna(method="ffill")

                    # change prices to return
                    filled_df_["wap1"] = filled_df_["wap1"].apply(np.log).diff(periods=1).fillna(0)
                    filled_df_["wap2"] = filled_df_["wap2"].apply(np.log).diff(periods=1).fillna(0)

                    filled_df_ = filled_df_[config.order_features].values

                    return time_id, filled_df_

                book_by_time_list = Parallel(n_jobs=-1, verbose=1)(delayed(fill_order)(time_id)
                                                                   for time_id in list(order_df["time_id"].unique()))
                book_by_time = {element[0]: element[1] for element in enumerate(book_by_time_list)}

                order_books[stock_id] = book_by_time

            else:

                book_by_time = dict()
                for time_id in order_df.time_id.unique():
                    df = order_df[order_df["time_id"] == time_id].reset_index(drop=True)
                    filled_df = pd.DataFrame({"seconds_in_bucket": range(config.hist_window)})
                    filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
                    filled_df = filled_df.fillna(method="ffill")

                    # change prices to return
                    filled_df["wap1"] = filled_df["wap1"].apply(np.log).diff(periods=1).fillna(0)
                    filled_df["wap2"] = filled_df["wap2"].apply(np.log).diff(periods=1).fillna(0)

                    filled_df = filled_df[config.order_features].values

                    book_by_time[time_id] = filled_df

                order_books[stock_id] = book_by_time

        with open(order_books_path, "wb") as f:
            dump(order_books, f)

    # load trade book
    trade_books_path = os.path.join(config.data_dir, "trade_book_{}.pkl".format(mode))

    if os.path.exists(trade_books_path):
        trade_books = load(trade_books_path)

    else:
        trade_books = dict()
        for path in tqdm(trade_path):

            stock_id = int(path.replace("\\", "/").split("=")[1].split("/")[0])
            trade_df = pd.read_parquet(path)

            # fe
            trade_df["volumes"] = trade_df["price"] * trade_df["size"]
            trade_df = trade_df.drop(["size"], axis=1)

            # float 64 to float 32
            float_cols = trade_df.select_dtypes(include=[np.float64]).columns
            trade_df[float_cols] = trade_df[float_cols].astype(np.float32)

            # int 64 to int 32
            int_cols = trade_df.select_dtypes(include=[np.int64]).columns
            trade_df[int_cols] = trade_df[int_cols].astype(np.int32)

            if parallel:

                def fill_trade(time_id):

                    df_ = trade_df[trade_df["time_id"] == time_id].reset_index(drop=True)
                    filled_df_ = pd.DataFrame({"seconds_in_bucket": range(config.hist_window)})
                    filled_df_ = pd.merge(filled_df_, df_, on=["seconds_in_bucket"], how="left")
                    filled_df_ = filled_df_.fillna(0)

                    # change prices to return
                    filled_df_["price"] = filled_df_["price"].apply(np.log).diff(periods=1).fillna(0)

                    filled_df_ = filled_df_[config.trade_features].values

                    return time_id, filled_df_

                trade_by_time_list = Parallel(n_jobs=-1, verbose=1)(delayed(fill_trade)(time_id)
                                                                    for time_id in list(trade_df["time_id"].unique()))
                trade_by_time = {element[0]: element[1] for element in enumerate(trade_by_time_list)}
                trade_books[stock_id] = trade_by_time

            else:

                trade_by_time = dict()
                for time_id in trade_df.time_id.unique():
                    df = trade_df[trade_df["time_id"] == time_id].reset_index(drop=True)
                    filled_df = pd.DataFrame({"seconds_in_bucket": range(config.hist_window)})
                    filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
                    filled_df = filled_df.fillna(0)

                    # change prices to return
                    filled_df["price"] = filled_df["price"].apply(np.log).diff(periods=1).fillna(0)

                    filled_df = filled_df[config.trade_features].values

                    trade_by_time[time_id] = filled_df

                trade_books[stock_id] = trade_by_time

        with open(trade_books_path, "wb") as f:
            dump(trade_books, f)

    return order_books, trade_books


def get_train_val_loader(config):
    if config.split == "GroupKFold":
        kfold = GroupKFold(n_splits=config.n_splits)
    else:
        raise NotImplementedError

    # load sequence data
    order_books, trade_books = load_data(config, mode="train", parallel=False)

    # load interval data
    interval_feature_path = os.path.join(config.data_dir, "train_interval_features.pkl")
    if os.path.exists(interval_feature_path):
        df = pd.read_pickle(interval_feature_path)
    else:
        df = pd.read_csv(config.train_data)
        df["row_id"] = df["stock_id"].astype(str) + "-" + df["time_id"].astype(str)

        df = generate_interval_feature(df)
        df.to_pickle(interval_feature_path)

    # shuffle by random seed
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    # kfold
    train_loader, val_loader = None, None
    for fold, (train_index, val_index) in enumerate(kfold.split(df, groups=df["time_id"])):
        if fold != config.fold:
            continue
        else:
            train_dataset = QuantDataset(config, df.copy(), train_index, order_books, trade_books, mode="train",
                                         fold=fold)
            val_dataset = QuantDataset(config, df.copy(), val_index, order_books, trade_books, mode="val")
            train_loader = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      shuffle=True,
                                      drop_last=True)
            val_loader = DataLoader(val_dataset,
                                    batch_size=config.val_batch_size,
                                    num_workers=config.num_workers,
                                    shuffle=False,
                                    drop_last=False)
            break

    return train_loader, val_loader


def get_test_loader(config):
    # load sequence data
    order_books, trade_books = load_data(config, mode="test", parallel=False)

    # load interval data
    interval_feature_path = os.path.join(config.data_dir, "test_interval_features.pkl")
    if os.path.exists(interval_feature_path):
        df = pd.read_pickle(interval_feature_path)
    else:
        df = pd.read_csv(config.test_data)
        df["row_id"] = df["stock_id"].astype(str) + "-" + df["time_id"].astype(str)

        df = generate_interval_feature(df)
        df.to_pickle(interval_feature_path)

    # infer dataset
    test_index = range(df.shape[0])
    test_dataset = QuantDataset(config, df, test_index, order_books, trade_books, mode="test")

    test_loader = DataLoader(test_dataset,
                             batch_size=config.val_batch_size,
                             num_workers=config.num_workers,
                             shuffle=False,
                             drop_last=False
                             )

    return test_loader
