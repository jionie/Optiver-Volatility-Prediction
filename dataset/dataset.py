from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
# from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset


class QuantDataset(Dataset):
    def __init__(self, config, sample_indices, mode="train"):

        self.config = config

        # load data df
        self.mode = mode
        if mode == "train":
            self.df = pd.read_csv(self.config.train_data)
        else:
            self.df = pd.read_csv(self.config.test_data)

        self.order_books = None
        self.trade_books = None
        self.load_data()

        self.sample_indices = sample_indices
        self.seq_len = self.config.hist_window

        self.cate_cols = self.config.cate_cols
        self.cont_cols = self.config.cont_cols
        self.target_cols = self.config.target_cols

        self.means_order = torch.from_numpy(np.array([0.9997, 1.0003, 769.9902, 766.7346, 0.9995, 1.0005, 959.3417,
                                                      928.2203]))
        self.stds_order = torch.from_numpy(
            np.array([3.6881e-03, 3.6871e-03, 5.3541e+03, 4.9549e+03, 3.7009e-03, 3.6991e-03,
                      6.6838e+03, 5.7353e+03]))

        self.means_trade = torch.from_numpy(np.array([1.0, 100, 3.0]))
        self.stds_trade = torch.from_numpy(np.array([0.004, 153, 3.5]))

    def load_data(self):

        if self.mode == "train":
            order_path = self.config.train_order
            trade_path = self.config.train_trade

        elif self.mode == "test":
            order_path = self.config.test_order
            trade_path = self.config.test_trade

        else:
            raise NotImplementedError

        def read_parquet_data(path):
            stock_id = int(path.replace("\\", "/").split("=")[1].split("/")[0])

            df = pd.read_parquet(path)

            # float 64 to float 32
            float_cols = df.select_dtypes(include=[np.float64]).columns
            df[float_cols] = df[float_cols].astype(np.float32)

            # int 64 to int 32
            int_cols = df.select_dtypes(include=[np.int64]).columns
            df[int_cols] = df[int_cols].astype(np.int32)

            by_time = dict()

            for time_id in df.time_id.unique():
                by_time[time_id] = df[df["time_id"] == time_id].reset_index(drop=True)

            return {"stock_id": stock_id, "by_time": by_time}

        # load order book
        self.order_books = Parallel(n_jobs=-1, verbose=1)(delayed(read_parquet_data)(path) for path in order_path)
        self.order_books = {item["stock_id"]: item["by_time"] for item in self.order_books}

        # load trade book
        self.trade_books = Parallel(n_jobs=-1, verbose=1)(delayed(read_parquet_data)(path) for path in trade_path)
        self.trade_books = {item["stock_id"]: item["by_time"] for item in self.trade_books}

    def extract_book_features(self, data_dict, stock_id, time_id, features, normalize="standard"):

        df = data_dict[stock_id][time_id]

        filled_df = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
        filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
        filled_df = filled_df.fillna(method="ffill")

        if normalize == "standard":
            book_x = (torch.from_numpy(filled_df[features].values) - self.means_order) / self.stds_order
        elif normalize == "log1p":
            book_x = torch.from_numpy(np.log1p(filled_df[features].values))
        else:
            raise NotImplementedError

        book_mask = torch.from_numpy(np.ones(self.seq_len))

        return book_x, book_mask

    def extract_trade_features(self, data_dict, stock_id, time_id, features, normalize="standard"):

        df = data_dict[stock_id][time_id]

        filled_df = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
        filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
        trade_mask = torch.from_numpy(1 - (filled_df.isnull().astype(np.int32).sum(axis=1) >= 1)
                                      .astype(np.int32).values)
        filled_df = filled_df.fillna(-1)

        if normalize == "standard":
            trade_x = (torch.from_numpy(filled_df[features].values) - self.means_order) / self.stds_order
        elif normalize == "log1p":
            trade_x = torch.from_numpy(np.log1p(filled_df[features].values))
        else:
            raise NotImplementedError

        return trade_x, trade_mask

    def __getitem__(self, idx):

        indices = self.sample_indices[idx]
        row = self.df.iloc[indices]

        book_x, book_mask = self.extract_book_features(self.order_books, row.stock_id, row.time_id,
                                                       self.config.order_features)
        try:
            trade_x, trade_mask = self.extract_trade_features(self.trade_books, row.stock_id, row.time_id,
                                                              self.config.trade_features)
        except Exception as e:
            trade_x = -torch.ones((self.seq_len, len(self.config.trade_features)))
            trade_mask = torch.zeros(self.seq_len)

        cont_x = torch.cat([book_x, trade_x], dim=1)
        cont_mask = torch.cat([book_mask, trade_mask], dim=1)

        # should we use label encoder here???
        cate_x = torch.from_numpy(row[self.cate_cols].values)
        cate_mask = torch.ones_like(cate_x)

        # should we use relative target gain???
        if self.mode == "test":
            target = torch.ones(len(self.target_cols))
        else:
            target = torch.from_numpy(row[self.target_cols].values * self.config.target_scale)

        return cate_x, cate_mask, cont_x, cont_mask, target

    def __len__(self):
        return len(self.sample_indices)


def get_train_val_loader(config):
    if config.split == "GroupKFold":
        kfold = GroupKFold(n_splits=config.n_splits)
    else:
        raise NotImplementedError

    train_loader, val_loader = None, None
    df = pd.read_csv(config.train_data)
    for fold, (train_index, val_index) in enumerate(kfold.split(df, groups=df["time_id"])):
        if fold != config.fold:
            continue
        else:
            train_dataset = QuantDataset(config, train_index, mode="train")
            val_dataset = QuantDataset(config, val_index, mode="train")
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


def get_test_loader(config, online=False):
    df = pd.read_csv(config.test_data)

    test_index = range(df.shape[0])
    if online:
        test_dataset = QuantDataset(config, test_index, mode="test")
    else:
        test_dataset = QuantDataset(config, test_index, mode="train")

    test_loader = DataLoader(test_dataset,
                             batch_size=config.val_batch_size,
                             num_workers=config.num_workers,
                             shuffle=False,
                             drop_last=False
                             )

    return test_loader
