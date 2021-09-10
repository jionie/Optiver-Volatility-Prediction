import os


class Config:
    # config settings
    def __init__(self,
                 fold,
                 model_type="bert",
                 seed=2020,
                 batch_size=16,
                 accumulation_steps=1
                 ):

        # data configs
        self.data_dir = "../input/optiver-realized-volatility-prediction/"
        self.hist_window = 30
        self.target_window = 7
        self.train_data = os.path.join(self.data_dir,
                                       "sz_index_daily_train_hist_{}_target_{}.csv".format(
                                           self.hist_window,
                                           self.target_window
                                       ))
        self.test_data = os.path.join(self.data_dir,
                                      "sz_index_daily_test_hist_{}_target_{}.csv".format(
                                          self.hist_window,
                                          self.target_window
                                      ))
        self.cate_cols = []
        self.cont_cols = []
        for window in range(1, self.hist_window + 1):
            self.cont_cols += [
                "day_{}_before_close".format(window),
                "day_{}_before_open".format(window),
                "day_{}_before_high".format(window),
                "day_{}_before_low".format(window),
                "day_{}_before_pre_close".format(window),
                "day_{}_before_change".format(window),
                "day_{}_before_pct_chg".format(window),
                "day_{}_before_vol".format(window),
                "day_{}_before_amount".format(window),
                "day_{}_before_total_mv".format(window),
                "day_{}_before_float_mv".format(window),
                "day_{}_before_total_share".format(window),
                "day_{}_before_float_share".format(window),
                "day_{}_before_free_share".format(window),
                "day_{}_before_turnover_rate".format(window),
                "day_{}_before_turnover_rate_f".format(window),
                "day_{}_before_pe".format(window),
                "day_{}_before_pe_ttm".format(window),
                "day_{}_before_pb".format(window)
            ]

        self.target_cols = []
        for window in range(0, self.target_window):
            self.target_cols += [
                "day_{}_after_close".format(window),
            ]

        # cross validation configs
        self.split = "TimeSeriesSplit"
        self.seed = seed
        self.n_splits = 5
        self.fold = fold
        self.num_workers = 4

        self.batch_size = batch_size
        self.val_batch_size = 32

        # setting
        self.reuse_model = True
        self.load_from_load_from_data_parallel = False
        self.load_pretrain = False
        self.data_parallel = False  # enable data parallel training
        self.apex = True  # enable mix precision training
        self.load_optimizer = False
        self.skip_layers = []

        # model
        self.model_type = model_type
        self.model_name = "QuantModel"

        # path, specify the path for saving model
        self.checkpoint_pretrain = os.path.join("/media/jionie/my_disk/Kaggle/Tweet/pretrain",
                                                self.model_name + "/" + self.model_type + "-" + str(self.seed)
                                                + "/fold_0/pytorch_model.bin")
        self.model_folder = os.path.join("/media/jionie/my_disk/Kaggle/Tweet/model", self.model_name)
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)
        self.checkpoint_folder_all_fold = os.path.join(self.model_folder, self.model_type + "-" + str(self.seed))
        if not os.path.exists(self.checkpoint_folder_all_fold):
            os.mkdir(self.checkpoint_folder_all_fold)
        self.checkpoint_folder = os.path.join(self.checkpoint_folder_all_fold, "fold_" + str(self.fold) + "/")
        if not os.path.exists(self.checkpoint_folder):
            os.mkdir(self.checkpoint_folder)
        self.save_point = os.path.join(self.checkpoint_folder, "{}_step_{}_epoch.pth")
        self.load_points = [p for p in os.listdir(self.checkpoint_folder) if p.endswith(".pth")]
        if len(self.load_points) != 0:
            self.load_point = sorted(self.load_points, key=lambda x: int(x.split("_")[0]))[-1]
            self.load_point = os.path.join(self.checkpoint_folder, self.load_point)
        else:
            self.reuse_model = False

        # optimizer
        self.optimizer_name = "AdamW"
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 2

        # lr scheduler, can choose to use proportion or steps
        self.lr_scheduler_name = "WarmupLinear"
        self.warmup_proportion = 0
        self.warmup_steps = 200

        # lr
        self.max_lr = 1e-5
        self.min_lr = 1e-5
        self.lr = 2e-4
        self.weight_decay = 0.001

        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        # epochs
        self.num_epoch = 8
        # saving rate
        self.saving_rate = 1 / 3
        # early stopping
        self.early_stopping = 6
        # progress rate
        self.progress_rate = 1 / 3
