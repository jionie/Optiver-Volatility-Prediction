import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset


class QuantDataset(Dataset):
    def __init__(self, config, sample_indices, mode="train"):

        self.config = config

        # load data df
        self.mode = mode
        if mode == "train":
            self.df = pd.read_csv(self.config.train_data)
            other_df = pd.read_csv(self.config.test_data)
            full_df = pd.concat([self.df, other_df], axis=0)

            del other_df
            gc.collect()

        else:
            self.df = pd.read_csv(self.config.test_data)
            other_df = pd.read_csv(self.config.train_data)
            full_df = pd.concat([self.df, other_df], axis=0)

            del other_df
            gc.collect()

        self.sample_indices = sample_indices
        self.seq_len = self.config.hist_window

        self.cate_cols = self.config.cate_cols
        # label encoder transformation
        self.df["stock_id_orig"] = self.df["stock_id"].copy()
        full_df[self.cate_cols] = full_df[self.cate_cols].apply(LabelEncoder().fit_transform)
        self.df[self.cate_cols] = full_df.iloc[:self.df.shape[0]][self.cate_cols].astype(np.int32)

        del full_df
        gc.collect()

        self.cont_cols = self.config.cont_cols
        self.target_cols = self.config.target_cols

        self.means_order = torch.from_numpy(np.array([0.9997, 1.0003, 769.9902, 766.7346, 0.9995, 1.0005, 959.3417,
                                                      928.2203]))
        self.stds_order = torch.from_numpy(
            np.array([3.6881e-03, 3.6871e-03, 5.3541e+03, 4.9549e+03, 3.7009e-03, 3.6991e-03,
                      6.6838e+03, 5.7353e+03]))

        self.means_trade = torch.from_numpy(np.array([1.0, 100, 3.0]))
        self.stds_trade = torch.from_numpy(np.array([0.004, 153, 3.5]))

    def load_data(self, stock_id):

        if self.mode == "train":
            order_path = self.config.train_order
            trade_path = self.config.train_trade

        elif self.mode == "test":
            order_path = self.config.test_order
            trade_path = self.config.test_trade

        else:
            raise NotImplementedError

        def read_parquet_data(path):

            df = pd.read_parquet(os.path.join(path, "stock_id={}".format(int(stock_id))))

            # float 64 to float 32
            float_cols = df.select_dtypes(include=[np.float64]).columns
            df[float_cols] = df[float_cols].astype(np.float32)

            # int 64 to int 32
            int_cols = df.select_dtypes(include=[np.int64]).columns
            df[int_cols] = df[int_cols].astype(np.int32)

            return df

        order_df = read_parquet_data(order_path)
        trade_df = read_parquet_data(trade_path)

        return order_df, trade_df

    def extract_book_features(self, df, time_id, features, normalize="standard"):

        df = df.loc[df["time_id"] == time_id]

        filled_df = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
        filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
        filled_df = filled_df.fillna(method="ffill")

        if normalize == "standard":
            book_x = (torch.from_numpy(filled_df[features].values) - self.means_order) / self.stds_order
        elif normalize == "log1p":
            book_x = torch.from_numpy(np.log1p(filled_df[features].values))
        else:
            raise NotImplementedError

        return book_x

    def extract_trade_features(self, df, time_id, features, normalize="standard"):

        df = df.loc[df["time_id"] == time_id]

        filled_df = pd.DataFrame({"seconds_in_bucket": range(self.seq_len)})
        filled_df = pd.merge(filled_df, df, on=["seconds_in_bucket"], how="left")
        filled_df = filled_df.fillna(-1)

        if normalize == "standard":
            trade_x = (torch.from_numpy(filled_df[features].values) - self.means_order) / self.stds_order
        elif normalize == "log1p":
            trade_x = torch.from_numpy(np.log1p(filled_df[features].values))
        else:
            raise NotImplementedError

        return trade_x

    def __getitem__(self, idx):

        indices = self.sample_indices[idx]
        row = self.df.iloc[indices]

        # load parquet
        order_df, trade_df = self.load_data(row.stock_id_orig)

        # continuous features
        book_x = self.extract_book_features(order_df, row.time_id, self.config.order_features)
        try:
            trade_x = self.extract_trade_features(trade_df, row.time_id, self.config.trade_features)
        except Exception as e:
            trade_x = -torch.ones((self.seq_len, len(self.config.trade_features)))

        cont_x = torch.cat([book_x, trade_x], dim=1)

        # category feature
        cate_x = torch.from_numpy(row[self.cate_cols].values.astype(np.int32))

        # mask
        mask = torch.from_numpy(np.ones(self.seq_len))
        # should we use relative target gain???
        if self.mode == "test":
            target = torch.ones(len(self.target_cols))
        else:
            target = torch.from_numpy(row[self.target_cols].values * self.config.target_scale)

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

    # kfold
    train_loader, val_loader = None, None
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


def get_test_loader(config):

    # load data
    df = pd.read_csv(config.test_data)

    # infer dataset
    test_index = range(df.shape[0])
    test_dataset = QuantDataset(config, test_index, mode="test")

    test_loader = DataLoader(test_dataset,
                             batch_size=config.val_batch_size,
                             num_workers=config.num_workers,
                             shuffle=False,
                             drop_last=False
                             )

    return test_loader
