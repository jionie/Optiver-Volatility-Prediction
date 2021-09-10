import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
from torch.utils.data import DataLoader


class QuantDataset:
    def __init__(self, config, sample_indices, mode="train"):

        self.config = config

        # load data df
        if mode == "train":
            self.df = pd.read_csv(self.config.train_data)
        else:
            self.df = pd.read_csv(self.config.test_data)

        self.sample_indices = sample_indices
        self.seq_len = self.config.hist_window

        self.cate_cols = self.config.cate_cols
        self.cont_cols = self.config.cont_cols
        self.target_cols = self.config.target_cols

        if len(self.cate_cols) > 0:
            self.cate_df = self.df[self.cate_cols]
        else:
            self.cate_df = None

        self.cont_df = np.log1p(self.df[self.cont_cols])

        # load mask df if exists
        mask_path = os.path.join(self.config.data_dir,
                                 "mask_sz_index_daily_{}_hist_{}_target_{}.csv".format(
                                     mode,
                                     self.config.hist_window,
                                     self.config.target_window
                                 )
                                 )

        if os.path.exists(mask_path):
            self.mask_df = pd.read_csv(mask_path)
        else:
            self.mask_df = None

        self.target_df = self.df[self.target_cols]
        # if mode != "test":
        #     self.target_df = self.df[self.target_cols]
        # else:
        #     self.target_df = None

    def __getitem__(self, idx):

        indices = self.sample_indices[idx]

        # get category features if exists
        if self.cate_df is not None:
            cate_x = torch.from_numpy(self.cate_df.iloc[indices].values)
        else:
            cate_x = None

        # get continuous features
        cont_x = torch.from_numpy(self.cont_df.iloc[indices].values)

        # get mask if exists
        if self.mask_df is not None:
            mask = torch.from_numpy(self.mask_df.iloc[indices].values)
        else:
            mask = torch.from_numpy(np.ones(self.seq_len))

        if self.target_df is not None:
            target = torch.from_numpy(self.target_df.iloc[indices].values)
        else:
            target = None

        return cate_x, cont_x, mask, target

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
