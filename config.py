import os
import glob


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
        self.hist_window = 600
        self.train_data = os.path.join(self.data_dir, "train.csv")
        self.train_order = glob.glob(os.path.join(self.data_dir, "book_train.parquet/*/*"))
        self.train_trade = glob.glob(os.path.join(self.data_dir, "trade_train.parquet/*/*"))
        self.test_data = os.path.join(self.data_dir, "test.csv")
        self.test_order = glob.glob(os.path.join(self.data_dir, "book_test.parquet/*/*"))
        self.test_trade = glob.glob(os.path.join(self.data_dir, "trade_test.parquet/*/*"))
        
        self.cate_cols = [
            "stock_id"
        ]
        self.total_cate_size = 112  # 112 stocks in total

        self.order_features = [
            "bid_price1",
            "ask_price1",
            "bid_size1",
            "ask_size1",
            "bid_price2",
            "ask_price2",
            "bid_size2",
            "ask_size2"
        ]
        self.trade_features = [
            "price",
            "size",
            "order_count"
        ]
        self.cont_cols = self.order_features + self.trade_features

        self.target_cols = [
            "target"
        ]
        self.target_scale = 100
        self.normalize = "standard"

        self.total_extra_size = 511

        # cross validation configs
        self.n_clusters = 7
        self.split = "GroupKFold"
        self.seed = seed
        self.n_splits = 5
        self.fold = fold
        self.num_workers = 0

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
        self.emb_size = 64
        self.hidden_size = 64
        self.target_size = len(self.target_cols)
        self.dropout = 0.1
        self.num_hidden_layers = 1
        self.num_attention_heads = 4

        # path, specify the path for saving model
        self.checkpoint_pretrain = os.path.join("../ckpts/pretrain", self.model_name + "/" + self.model_type + "-" + str(self.seed) + "/fold_0/pytorch_model.bin")
        self.model_folder = os.path.join("../ckpts/", self.model_name)
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
        self.milestones = [5, 10]
        self.warmup_proportion = 0
        self.warmup_steps = 200
        self.warmup_epoches = 1

        # lr
        self.warmup_lr = 5e-5
        self.min_lr = 1e-5
        self.lr = 1e-3
        self.weight_decay = 0.001

        # gradient accumulation
        self.accumulation_steps = accumulation_steps
        # epochs
        self.num_epoch = 30
        # saving rate
        self.saving_rate = 1
        # early stopping
        self.early_stopping = 5  # 5 epoch
        # progress rate
        self.progress_rate = 1
