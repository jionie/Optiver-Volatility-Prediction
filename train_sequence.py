# import os and define graphic card
import os

os.environ["OMP_NUM_THREADS"] = "1"

# import common libraries
import random
import argparse
import numpy as np

# import pytorch related libraries
import torch
from tensorboardX import SummaryWriter
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup, AdamW

# import dataset class
from dataset.dataset import get_train_val_loader, get_test_loader

# import utils
from utils.ranger import Ranger
from utils.lrs_scheduler import WarmRestart
from utils.metric import rmspe
from utils.file import Logger
from utils.loss_function import RMSPELoss

# import model
from models.transformer import TransfomerModel, LSTMATTNModel

# import config
from config import Config

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--fold", type=int, default=0, required=False, help="specify the fold for training")
parser.add_argument("--model_type", type=str, default="bert", required=False, help="specify the model type")
parser.add_argument("--seed", type=int, default=2021, required=False, help="specify the seed")
parser.add_argument("--batch_size", type=int, default=512, required=False, help="specify the batch size")
parser.add_argument("--accumulation_steps", type=int, default=1, required=False, help="specify the accumulation_steps")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
    os.environ["PYHTONHASHseed"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


class Quant:
    def __init__(self, config):
        super(Quant).__init__()
        self.config = config
        self.setup_logger()
        self.setup_gpu()
        self.load_data()
        self.prepare_train()
        self.setup_model()

    def setup_logger(self):
        self.log = Logger()
        self.log.open((os.path.join(self.config.checkpoint_folder, "train_log.txt")), mode="a+")

    def setup_gpu(self):
        # confirm the device which can be either cpu or gpu
        self.config.use_gpu = torch.cuda.is_available()
        self.num_device = torch.cuda.device_count()
        if self.config.use_gpu:
            self.config.device = "cuda"
            if self.num_device <= 1:
                self.config.data_parallel = False
            elif self.config.data_parallel:
                torch.multiprocessing.set_start_method("spawn", force=True)
        else:
            self.config.device = "cpu"
            self.config.data_parallel = False

    def load_data(self):
        self.log.write("\nLoading data...")

        self.train_data_loader, self.val_data_loader = get_train_val_loader(
            self.config
        )

        self.test_data_loader = get_test_loader(
            self.config
        )

    def prepare_train(self):
        # preparation for training
        self.step = 0
        self.epoch = 0
        self.finished = False
        self.valid_epoch = 0
        self.train_loss, self.valid_loss, self.valid_metric_optimal = float("inf"), float("inf"), float("inf")
        self.writer = SummaryWriter()

        # eval setting
        self.eval_step = int(len(self.train_data_loader) * self.config.saving_rate)
        self.log_step = int(len(self.train_data_loader) * self.config.progress_rate)
        self.eval_count = 0
        self.count = 0

    def pick_model(self):
        # for switching model
        if self.config.model_type == "bert":
            self.model = TransfomerModel(self.config).to(self.config.device)

        elif self.config.model_type == "lstm":
            self.model = LSTMATTNModel(self.config).to(self.config.device)

        else:
            raise NotImplementedError

        if self.config.load_pretrain:
            checkpoint_to_load = torch.load(self.config.checkpoint_pretrain, map_location=self.config.device)
            model_state_dict = checkpoint_to_load

            if self.config.data_parallel:
                state_dict = self.model.model.state_dict()
            else:
                state_dict = self.model.state_dict()

            keys = list(state_dict.keys())

            for key in keys:
                if any(s in key for s in self.config.skip_layers):
                    continue
                try:
                    state_dict[key] = model_state_dict[key]
                except:
                    print("Missing key:", key)

            if self.config.data_parallel:
                self.model.model.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)

    def differential_lr(self, warmup=True):

        param_optimizer = list(self.model.named_parameters())

        if warmup:
            self.optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer],
                 "lr": self.config.warmup_lr}
            ]

        else:
            self.optimizer_grouped_parameters = [
                {"params": [p for n, p in param_optimizer],
                 "lr": self.config.lr}
            ]

    def prepare_optimizer(self):

        # optimizer
        if self.config.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.optimizer_grouped_parameters,
                                              eps=self.config.adam_epsilon,
                                              weight_decay=self.config.weight_decay,
                                              )
        elif self.config.optimizer_name == "Ranger":
            self.optimizer = Ranger(self.optimizer_grouped_parameters)
        elif self.config.optimizer_name == "AdamW":
            self.optimizer = AdamW(self.optimizer_grouped_parameters,
                                   eps=self.config.adam_epsilon,
                                   weight_decay=self.config.weight_decay,
                                   betas=(0.9, 0.999))
        else:
            raise NotImplementedError

        # lr scheduler
        if self.config.lr_scheduler_name == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config.milestones,
                                                                  gamma=0.5)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupCosineAnealing":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "WarmRestart":
            self.scheduler = WarmRestart(self.optimizer, T_max=5, T_mult=1, eta_min=1e-6)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupLinear":
            num_train_optimization_steps = self.config.num_epoch * len(self.train_data_loader) \
                                           // self.config.accumulation_steps
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                             num_warmup_steps=self.config.warmup_steps,
                                                             num_training_steps=num_train_optimization_steps)
            self.lr_scheduler_each_iter = True
        elif self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.8,
                                                                        patience=3, min_lr=1e-5)
            self.lr_scheduler_each_iter = False
        elif self.config.lr_scheduler_name == "WarmupConstant":
            self.scheduler = get_constant_schedule_with_warmup(self.optimizer,
                                                               num_warmup_steps=self.config.warmup_steps)
            self.lr_scheduler_each_iter = True
        else:
            raise NotImplementedError

        # lr scheduler step for checkpoints
        if self.lr_scheduler_each_iter:
            self.scheduler.step(self.step)
        else:
            self.scheduler.step(self.epoch)

    def prepare_apex(self):
        self.scaler = torch.cuda.amp.GradScaler()

    def load_check_point(self):
        self.log.write("Model loaded as {}.".format(self.config.load_point))
        checkpoint_to_load = torch.load(self.config.load_point, map_location=self.config.device)
        self.step = checkpoint_to_load["step"]
        self.epoch = checkpoint_to_load["epoch"]

        model_state_dict = checkpoint_to_load["model"]
        if self.config.load_from_load_from_data_parallel:
            model_state_dict = {k[13:]: v for k, v in model_state_dict.items()}

        if self.config.data_parallel:
            state_dict = self.model.model.state_dict()
        else:
            state_dict = self.model.state_dict()

        keys = list(state_dict.keys())

        for key in keys:
            if any(s in key for s in self.config.skip_layers):
                continue
            try:
                state_dict[key] = model_state_dict[key]
            except Exception as e:
                print("Missing key:", key)

        if self.config.data_parallel:
            self.model.model.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)

        if self.config.load_optimizer:
            self.optimizer.load_state_dict(checkpoint_to_load["optimizer"])

    def save_check_point(self):
        # save model, optimizer, and everything required to keep
        checkpoint_to_save = {
            "step": self.step,
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            # "optimizer": self.optimizer.state_dict()
        }

        save_path = self.config.save_point.format(self.step, self.epoch)
        torch.save(checkpoint_to_save, save_path)
        self.log.write("Model saved as {}.".format(save_path))

    def setup_model(self):
        self.pick_model()

        if self.config.data_parallel:

            self.differential_lr(warmup=True)
            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

            if self.config.reuse_model:
                self.load_check_point()

            self.model = torch.nn.DataParallel(self.model)

        else:
            if self.config.reuse_model:
                self.load_check_point()

            self.differential_lr(warmup=True)
            self.prepare_optimizer()

            if self.config.apex:
                self.prepare_apex()

    def count_parameters(self):
        # get total size of trainable parameters
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def show_info(self):
        # show general information before training
        self.log.write("\n*General Setting*")
        self.log.write("\nseed: {}".format(self.config.seed))
        self.log.write("\nmodel: {}".format(self.config.model_name))
        self.log.write("\ntrainable parameters:{:,.0f}".format(self.count_parameters()))
        self.log.write("\nmodel's state_dict:")
        self.log.write("\ndevice: {}".format(self.config.device))
        self.log.write("\nuse gpu: {}".format(self.config.use_gpu))
        self.log.write("\ndevice num: {}".format(self.num_device))
        self.log.write("\noptimizer: {}".format(self.optimizer))
        self.log.write("\nreuse model: {}".format(self.config.reuse_model))
        if self.config.reuse_model:
            self.log.write("\nModel restored from {}.".format(self.config.load_point))
        self.log.write("\n")

    def train_op(self):
        self.show_info()
        self.log.write("** start training here! **\n")
        self.log.write("   batch_size=%d,  accumulation_steps=%d\n" % (self.config.batch_size,
                                                                       self.config.accumulation_steps))
        self.log.write("   experiment  = %s\n" % str(__file__.split("/")[-2:]))

        criterion = RMSPELoss()

        while self.epoch <= self.config.num_epoch:

            predictions = []
            labels = []

            # warmup lr for parameter warmup_epoch
            if self.epoch >= self.config.warmup_epoches:
                self.differential_lr(warmup=False)
                self.prepare_optimizer()

            self.log.write("Epoch%s\n" % self.epoch)
            self.log.write("\n")

            sum_train_loss = np.zeros_like(self.train_loss)
            sum_train = np.zeros_like(self.train_loss)

            # init optimizer
            torch.cuda.empty_cache()
            self.model.zero_grad()

            for tr_batch_i, (cate_x, cont_x, extra_x, mask, target) in enumerate(self.train_data_loader):

                rate = 0
                for param_group in self.optimizer.param_groups:
                    rate += param_group["lr"] / len(self.optimizer.param_groups)

                # set model training mode
                self.model.train()

                # set input to cuda mode
                if cate_x is not None:
                    cate_x = cate_x.to(self.config.device).long()
                cont_x = cont_x.to(self.config.device).float()
                extra_x = extra_x.to(self.config.device).float()
                mask = mask.to(self.config.device).float()
                target = target.to(self.config.device).float()

                outputs = self.model(cate_x, cont_x, extra_x, mask)

                if self.config.apex:
                    with torch.cuda.amp.autocast():
                        loss = criterion(outputs, target)
                    self.scaler.scale(loss).backward()
                else:
                    loss = criterion(outputs, target)
                    loss.backward()

                if (tr_batch_i + 1) % self.config.accumulation_steps == 0:
                    # use apex
                    if self.config.apex:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm,
                                                       norm_type=2)
                        self.optimizer.step()

                    self.model.zero_grad()

                    # adjust lr
                    if self.lr_scheduler_each_iter:
                        self.scheduler.step()

                    self.writer.add_scalar("train_loss_" + str(self.config.fold), loss.item(),
                                           (self.epoch - 1) * len(
                                               self.train_data_loader) * self.config.batch_size + tr_batch_i *
                                           self.config.batch_size)
                    self.step += 1

                # translate to predictions
                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                outputs = to_numpy(outputs)
                target = to_numpy(target)

                predictions.append(outputs.flatten())
                labels.append(target.flatten())

                sum_train_loss = sum_train_loss + np.array([loss.item() * self.config.batch_size])
                sum_train = sum_train + np.array([self.config.batch_size])

                # log for training
                if (tr_batch_i + 1) % self.log_step == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train[...] = 0
                    mean_train_rmspe = rmspe(np.concatenate(labels), np.concatenate(predictions))

                    self.log.write(
                        "lr: {} train loss: {} train_rmspe: {} \n"
                            .format(rate, train_loss[0], mean_train_rmspe)
                    )

                if (tr_batch_i + 1) % self.eval_step == 0:
                    self.evaluate_op()

            # update lr and start from start_epoch
            if (not self.lr_scheduler_each_iter) and (self.config.lr_scheduler_name != "ReduceLROnPlateau"):
                self.scheduler.step()

            if self.count >= self.config.early_stopping:
                break

            self.epoch += 1

    def evaluate_op(self):

        self.eval_count += 1
        valid_loss = np.zeros(1, np.float32)
        valid_num = np.zeros_like(valid_loss)

        criterion = RMSPELoss()

        predictions = []
        labels = []

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()

            for val_batch_i, (cate_x, cont_x, extra_x, mask, target) in enumerate(self.val_data_loader):

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                if cate_x is not None:
                    cate_x = cate_x.to(self.config.device).long()
                cont_x = cont_x.to(self.config.device).float()
                extra_x = extra_x.to(self.config.device).float()
                mask = mask.to(self.config.device).float()
                target = target.to(self.config.device).float()

                outputs = self.model(cate_x, cont_x, extra_x, mask)
                loss = criterion(outputs, target)

                self.writer.add_scalar("val_loss_" + str(self.config.fold), loss.item(), (self.eval_count - 1) * len(
                    self.val_data_loader) * self.config.val_batch_size + val_batch_i * self.config.val_batch_size)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                outputs = to_numpy(outputs)
                target = to_numpy(target)

                predictions.append(outputs.flatten())
                labels.append(target.flatten())

                valid_loss = valid_loss + np.array([loss.item() * self.config.val_batch_size])
                valid_num = valid_num + np.array([self.config.val_batch_size])

            valid_loss = valid_loss / valid_num
            mean_eval_rmspe = rmspe(np.concatenate(labels), np.concatenate(predictions))

            self.log.write(
                "mean validation loss: {} eval_rmspe: {} \n"
                    .format(valid_loss[0], mean_eval_rmspe)
            )

        if self.config.lr_scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(mean_eval_rmspe)

        if mean_eval_rmspe <= self.valid_metric_optimal:

            self.log.write("Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...".format(
                self.valid_metric_optimal, mean_eval_rmspe))

            self.valid_metric_optimal = mean_eval_rmspe
            self.save_check_point()

            self.count = 0

        else:
            self.count += 1

    def test_op(self, online=False):

        self.eval_count += 1
        test_loss = np.zeros(1, np.float32)
        test_num = np.zeros_like(test_loss)

        criterion = RMSPELoss()

        self.test_pearson = []
        self.test_spearman = []
        predictions = []

        with torch.no_grad():

            # init cache
            torch.cuda.empty_cache()

            for test_batch_i, (cate_x, cont_x, extra_x, mask, target) in enumerate(self.test_data_loader):

                # set model to eval mode
                self.model.eval()

                # set input to cuda mode
                if cate_x is not None:
                    cate_x = cate_x.to(self.config.device).long()
                cont_x = cont_x.to(self.config.device).float()
                extra_x = extra_x.to(self.config.device).float()
                mask = mask.to(self.config.device).float()

                outputs = self.model(cate_x, cont_x, extra_x, mask)

                def to_numpy(tensor):
                    return tensor.detach().cpu().numpy()

                predictions.append(to_numpy(outputs))

            return np.concatenate(predictions, axis=0)


if __name__ == "__main__":
    args = parser.parse_args()

    # config = Config(
    #     args.fold,
    #     model_type=args.model_type,
    #     seed=args.seed,
    #     batch_size=args.batch_size,
    #     accumulation_steps=args.accumulation_steps
    # )

    # update fold
    for fold in range(5):
        config = Config(
            fold,
            model_type=args.model_type,
            seed=args.seed,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps
        )

        # seed
        seed_everything(config.seed)

        # init class instance
        qa = Quant(config)

        # trainig
        qa.train_op()

        # back testing
        # qa.test_op(online=False)
