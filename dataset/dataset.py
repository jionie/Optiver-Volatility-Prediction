import os
import gc
from tqdm import tqdm
from joblib import dump, load, Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset


class QuantDataset(Dataset):
    def __init__(self, config, df, sample_indices, mode="train", parallel=False):

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

        self.means_order = np.array([0.9997, 1.0003, 769.9902, 766.7346, 0.9995, 1.0005, 959.3417, 928.2203])
        self.stds_order = np.array([3.6881e-03, 3.6871e-03, 5.3541e+03, 4.9549e+03, 3.7009e-03, 3.6991e-03, 6.6838e+03,
                                    5.7353e+03])

        self.means_trade = np.array([1.0, 100, 3.0])
        self.stds_trade = np.array([0.004, 153, 3.5])

        # store stock_id and time_id
        self.stock_ids = df["stock_id"].values
        self.time_ids = df["time_id"].values

        # cont_x init
        self.order_books = None
        self.trade_books = None
        self.cont_cols = self.config.cont_cols
        self.load_data()

        # cate_x init
        self.cate_cols = self.config.cate_cols
        # label encoder transformation
        full_df[self.cate_cols] = full_df[self.cate_cols].apply(LabelEncoder().fit_transform)
        df[self.cate_cols] = full_df.iloc[:df.shape[0]][self.cate_cols].astype(np.int32)

        del full_df
        gc.collect()
        self.cate_x = df[self.cate_cols].values

        # target
        if mode == "train" or mode == "val":
            self.target = df[self.config.target_cols].values
        elif mode == "test":
            self.target = None
        else:
            raise NotImplementedError

    def load_data(self):

        if self.mode == "train" or self.mode == "val":
            order_path = self.config.train_order
            trade_path = self.config.train_trade

        elif self.mode == "test":
            order_path = self.config.test_order
            trade_path = self.config.test_trade

        else:
            raise NotImplementedError

        # load order book
        order_books_path = os.path.join(self.config.data_dir, "order_book_seed_{}_fold_{}_{}.pkl"
                                        .format(self.config.seed, self.config.fold, self.mode))

        if os.path.exists(order_books_path):
            self.order_books = load(order_books_path)

        else:

            self.order_books = dict()
            for path in tqdm(order_path):

                stock_id = int(path.replace("\\", "/").split("=")[1].split("/")[0])
                book_df = pd.read_parquet(path)
                book_df = book_df.loc[book_df["time_id"].isin(self.valid_time_ids)]

                # float 64 to float 32
                float_cols = book_df.select_dtypes(include=[np.float64]).columns
                book_df[float_cols] = book_df[float_cols].astype(np.float32)

                # int 64 to int 32
                int_cols = book_df.select_dtypes(include=[np.int64]).columns
                book_df[int_cols] = book_df[int_cols].astype(np.int32)

                if self.parallel:

                    def fill_order(time_id):

                        df_ = book_df[book_df["time_id"] == time_id].reset_index(drop=True)
                        filled_df_ = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
                        filled_df_ = pd.merge(filled_df_, df_, on=["seconds_in_bucket"], how="left")
                        filled_df_ = filled_df_.fillna(method="ffill")
                        filled_df_ = filled_df_[self.config.order_features].values

                        if self.config.normalize == "standard":
                            filled_df_ = (filled_df_ - self.means_order) / self.stds_order
                        elif self.config.normalize == "log1p":
                            filled_df_ = np.log1p(filled_df_)
                        else:
                            raise NotImplementedError

                        return time_id, filled_df_

                    book_by_time_list = Parallel(n_jobs=-1, verbose=1)(delayed(fill_order)(time_id)
                                                                       for time_id in self.valid_time_ids)
                    book_by_time = {element[0]: element[1] for element in enumerate(book_by_time_list)}

                    self.order_books[stock_id] = book_by_time

                else:

                    book_by_time = dict()
                    for time_id in book_df.time_id.unique():

                        df = book_df[book_df["time_id"] == time_id].reset_index(drop=True)
                        filled_df = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
                        filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
                        filled_df = filled_df.fillna(method="ffill")
                        filled_df = filled_df[self.config.order_features].values

                        if self.config.normalize == "standard":
                            filled_df = (filled_df - self.means_order) / self.stds_order
                        elif self.config.normalize == "log1p":
                            filled_df = np.log1p(filled_df)
                        else:
                            raise NotImplementedError

                        book_by_time[time_id] = filled_df

                    self.order_books[stock_id] = book_by_time

            with open(order_books_path, "wb") as f:
                dump(self.order_books, f)

        # load trade book
        trade_books_path = os.path.join(self.config.data_dir, "trade_book_seed_{}_fold_{}_{}.pkl"
                                        .format(self.config.seed, self.config.fold, self.mode))

        if os.path.exists(trade_books_path):
            self.trade_books = load(trade_books_path)

        else:
            self.trade_books = dict()
            for path in tqdm(trade_path):

                stock_id = int(path.replace("\\", "/").split("=")[1].split("/")[0])
                trade_df = pd.read_parquet(path)
                trade_df = trade_df.loc[trade_df["time_id"].isin(self.valid_time_ids)]

                # float 64 to float 32
                float_cols = trade_df.select_dtypes(include=[np.float64]).columns
                trade_df[float_cols] = trade_df[float_cols].astype(np.float32)

                # int 64 to int 32
                int_cols = trade_df.select_dtypes(include=[np.int64]).columns
                trade_df[int_cols] = trade_df[int_cols].astype(np.int32)

                if self.parallel:

                    def fill_trade(time_id):

                        df_ = trade_df[trade_df["time_id"] == time_id].reset_index(drop=True)
                        filled_df_ = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
                        filled_df_ = pd.merge(filled_df_, df_, on=["seconds_in_bucket"], how="left")
                        filled_df_ = filled_df_.fillna(0)
                        filled_df_ = filled_df_[self.config.trade_features].values

                        if self.config.normalize == "standard":
                            filled_df_ = (filled_df_ - self.means_trade) / self.stds_trade
                        elif self.config.normalize == "log1p":
                            filled_df_ = np.log1p(filled_df_)
                        else:
                            raise NotImplementedError

                        return time_id, filled_df_

                    trade_by_time_list = Parallel(n_jobs=-1, verbose=1)(delayed(fill_trade)(time_id)
                                                                        for time_id in self.valid_time_ids)
                    trade_by_time = {element[0]: element[1] for element in enumerate(trade_by_time_list)}
                    self.trade_books[stock_id] = trade_by_time

                else:

                    trade_by_time = dict()
                    for time_id in trade_df.time_id.unique():

                        df = trade_df[trade_df["time_id"] == time_id].reset_index(drop=True)
                        filled_df = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
                        filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
                        filled_df = filled_df.fillna(0)
                        filled_df = filled_df[self.config.trade_features].values

                        if self.config.normalize == "standard":
                            filled_df = (filled_df - self.means_trade) / self.stds_trade
                        elif self.config.normalize == "log1p":
                            filled_df = np.log1p(filled_df)
                        else:
                            raise NotImplementedError

                        trade_by_time[time_id] = filled_df

                    self.trade_books[stock_id] = trade_by_time

            with open(trade_books_path, "wb") as f:
                dump(self.trade_books, f)

    def __getitem__(self, idx):

        indices = self.sample_indices[idx]

        # continuous features
        book_x = torch.from_numpy(self.order_books[self.stock_ids[indices]][self.time_ids[indices]])

        try:
            trade_x = torch.from_numpy(self.trade_books[self.stock_ids[indices]][self.time_ids[indices]])

        except Exception as e:
            trade_x = (torch.zeros((self.seq_len, len(self.config.trade_features)))
                       - torch.from_numpy(self.means_trade).unsqueeze(0)) / \
                      torch.from_numpy(self.stds_trade).unsqueeze(0)

        cont_x = torch.cat([book_x, trade_x], dim=1)

        # category feature
        cate_x = torch.repeat_interleave(torch.from_numpy(self.cate_x[indices].astype(np.int32)).unsqueeze(0),
                                         self.seq_len, dim=0)

        # mask
        mask = torch.from_numpy(np.ones(self.seq_len))
        # should we use relative target gain???
        if self.mode == "test":
            target = torch.ones(len(self.config.target_cols))
        else:
            target = torch.from_numpy(self.target[indices] * self.config.target_scale)

        return cate_x, cont_x, mask, target

    def __len__(self):
        return len(self.sample_indices)


def get_train_val_loader(config):
    if config.split == "GroupKFold":
        kfold = GroupKFold(n_splits=config.n_splits)
    else:
        raise NotImplementedError

    # load data
    df = pd.read_csv(config.train_data)

    # shuffle by random seed
    df = df.sample(frac=1, random_state=config.seed).reset_index(drop=True)

    # kfold
    train_loader, val_loader = None, None
    for fold, (train_index, val_index) in enumerate(kfold.split(df, groups=df["time_id"])):
        if fold != config.fold:
            continue
        else:
            train_dataset = QuantDataset(config, df.copy(), train_index, mode="train")
            val_dataset = QuantDataset(config, df.copy(), val_index, mode="val")
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
    # load data
    df = pd.read_csv(config.test_data)

    # infer dataset
    test_index = range(df.shape[0])
    test_dataset = QuantDataset(config, df, test_index, mode="test")

    test_loader = DataLoader(test_dataset,
                             batch_size=config.val_batch_size,
                             num_workers=config.num_workers,
                             shuffle=False,
                             drop_last=False
                             )

    return test_loader
