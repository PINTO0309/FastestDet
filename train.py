import os
import math
import torch
import argparse
import warnings
import random
import numpy as np
from tqdm import tqdm
from torch import optim
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.resize import resize_output_channels
from utils.evaluation import CocoDetectionEvaluator

from module.loss import DetectorLoss
from module.detector import Detector

# Suppress noisy future warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# Select backend device: CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastestDet:
    def __init__(self):
        seed_everything(SEED)
        # Training config
        parser = argparse.ArgumentParser()
        parser.add_argument('--yaml', type=str, default="", help='.yaml config')
        parser.add_argument('--weight', type=str, default=None, help='.weight config')
        parser.add_argument('--classes', type=str, default=None, help='comma-separated class ids')
        parser.add_argument('--exp-name', type=str, default="exp", help='experiment name (runs/<exp-name>)')
        parser.add_argument('--aug-yaml', type=str, default="utils/aug_headpose.yaml", help='augmentation yaml')
        resize_group = parser.add_mutually_exclusive_group()
        resize_group.add_argument(
            "--resize-mode",
            choices=[
                "torch_bilinear",
                "torch_nearest",
                "opencv_inter_linear",
                "opencv_inter_nearest",
                "opencv_inter_nearest_y_bin",
                "opencv_inter_nearest_y",
                "opencv_inter_nearest_y_tri",
                "opencv_inter_nearest_yuv422",
            ],
            dest="resize_mode",
            help="Resize mode used during training preprocessing.",
        )
        resize_group.add_argument(
            "--torch_bilinear",
            dest="resize_mode",
            action="store_const",
            const="torch_bilinear",
            help="Shortcut for --resize-mode torch_bilinear.",
        )
        resize_group.add_argument(
            "--torch_nearest",
            dest="resize_mode",
            action="store_const",
            const="torch_nearest",
            help="Shortcut for --resize-mode torch_nearest.",
        )
        resize_group.add_argument(
            "--opencv_inter_linear",
            dest="resize_mode",
            action="store_const",
            const="opencv_inter_linear",
            help="Shortcut for --resize-mode opencv_inter_linear.",
        )
        resize_group.add_argument(
            "--opencv_inter_nearest",
            dest="resize_mode",
            action="store_const",
            const="opencv_inter_nearest",
            help="Shortcut for --resize-mode opencv_inter_nearest.",
        )
        resize_group.add_argument(
            "--opencv_inter_nearest_y",
            dest="resize_mode",
            action="store_const",
            const="opencv_inter_nearest_y",
            help="Shortcut for --resize-mode opencv_inter_nearest_y.",
        )
        resize_group.add_argument(
            "--opencv_inter_nearest_y_bin",
            dest="resize_mode",
            action="store_const",
            const="opencv_inter_nearest_y_bin",
            help="Shortcut for --resize-mode opencv_inter_nearest_y_bin.",
        )
        resize_group.add_argument(
            "--opencv_inter_nearest_y_tri",
            dest="resize_mode",
            action="store_const",
            const="opencv_inter_nearest_y_tri",
            help="Shortcut for --resize-mode opencv_inter_nearest_y_tri.",
        )
        resize_group.add_argument(
            "--opencv_inter_nearest_yuv422",
            dest="resize_mode",
            action="store_const",
            const="opencv_inter_nearest_yuv422",
            help="Shortcut for --resize-mode opencv_inter_nearest_yuv422.",
        )

        opt = parser.parse_args()
        assert os.path.exists(opt.yaml), "Please provide a valid config file path."

        # Parse yaml config
        self.cfg = LoadYaml(opt.yaml)
        cli_classes = parse_classes(opt.classes)
        if cli_classes is not None:
            self.cfg.classes = cli_classes
        self.resize_mode = opt.resize_mode
        self.aug_yaml = opt.aug_yaml if opt.aug_yaml else None
        self.input_channels = resize_output_channels(self.resize_mode)
        self.exp_dir = os.path.join("runs", opt.exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.best_map05 = float("-inf")
        print(self.cfg)

        # Initialize model
        if opt.weight is not None:
            print("load weight from:%s"%opt.weight)
            self.model = Detector(self.cfg.category_num, True, self.input_channels).to(device)
            self.model.load_state_dict(torch.load(opt.weight))
        else:
            self.model = Detector(self.cfg.category_num, False, self.input_channels).to(device)

        # # Print tensor shapes of network layers
        summary(self.model, input_size=(self.input_channels, self.cfg.input_height, self.cfg.input_width))

        # Build optimizer
        print("use SGD optimizer")
        self.optimizer = optim.SGD(params=self.model.parameters(),
                                   lr=self.cfg.learn_rate,
                                   momentum=0.949,
                                   weight_decay=0.0005,
                                   )
        # Learning rate decay schedule
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                        milestones=self.cfg.milestones,
                                                        gamma=0.1)

        # Define loss
        self.loss_function = DetectorLoss(device)

        # Define evaluation
        self.evaluation = CocoDetectionEvaluator(self.cfg.names, device)

        # Dataset loading
        val_dataset = TensorDataset(
            self.cfg.val_txt,
            self.cfg.input_width,
            self.cfg.input_height,
            False,
            self.cfg.classes,
            self.resize_mode,
        )
        train_dataset = TensorDataset(
            self.cfg.train_txt,
            self.cfg.input_width,
            self.cfg.input_height,
            True,
            self.cfg.classes,
            self.resize_mode,
            self.aug_yaml,
        )

        # Validation set
        generator = torch.Generator()
        generator.manual_seed(SEED)
        self.val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                          batch_size=self.cfg.batch_size,
                                                          shuffle=False,
                                                          collate_fn=collate_fn,
                                                          num_workers=12,
                                                          drop_last=False,
                                                          persistent_workers=True,
                                                          worker_init_fn=seed_worker,
                                                          generator=generator,
                                                          pin_memory=True,
                                                          )
        # Training set
        self.train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                            batch_size=self.cfg.batch_size,
                                                            shuffle=True,
                                                            collate_fn=collate_fn,
                                                            num_workers=12,
                                                            persistent_workers=True,
                                                            worker_init_fn=seed_worker,
                                                            generator=generator,
                                                            pin_memory=True,
                                                            )

    def _prune_checkpoints(self, max_keep=10):
        checkpoints = []
        for name in os.listdir(self.exp_dir):
            if not name.startswith("weight_AP05-") or not name.endswith(".pth"):
                continue
            path = os.path.join(self.exp_dir, name)
            epoch_num = None
            try:
                epoch_part = name.split("_")[-1]
                epoch_num = int(epoch_part.split("-")[0])
            except (ValueError, IndexError):
                epoch_num = None
            mtime = os.path.getmtime(path)
            checkpoints.append((epoch_num, mtime, path))
        if len(checkpoints) <= max_keep:
            return
        def sort_key(item):
            epoch_num, mtime, _ = item
            return (epoch_num if epoch_num is not None else -1, mtime)
        checkpoints.sort(key=sort_key, reverse=True)
        for _, _, path in checkpoints[max_keep:]:
            try:
                os.remove(path)
            except OSError:
                pass

    def _prune_best_checkpoints(self, max_keep=10):
        checkpoints = []
        for name in os.listdir(self.exp_dir):
            if not name.startswith("best_AP05-") or not name.endswith(".pth"):
                continue
            path = os.path.join(self.exp_dir, name)
            epoch_num = None
            try:
                epoch_part = name.split("_")[-1]
                epoch_num = int(epoch_part.split("-")[0])
            except (ValueError, IndexError):
                epoch_num = None
            mtime = os.path.getmtime(path)
            checkpoints.append((epoch_num, mtime, path))
        if len(checkpoints) <= max_keep:
            return
        def sort_key(item):
            epoch_num, mtime, _ = item
            return (epoch_num if epoch_num is not None else -1, mtime)
        checkpoints.sort(key=sort_key, reverse=True)
        for _, _, path in checkpoints[max_keep:]:
            try:
                os.remove(path)
            except OSError:
                pass

    def train(self):
        # Training loop
        batch_num = 0
        input_is_normalized = getattr(self.train_dataloader.dataset, "input_is_normalized", False)
        print('Starting training for %g epochs...' % self.cfg.end_epoch)
        for epoch in range(self.cfg.end_epoch + 1):
            self.model.train()
            pbar = tqdm(self.train_dataloader)
            for imgs, targets in pbar:
                # Data preprocessing
                imgs = imgs.to(device).float()
                if not input_is_normalized:
                    imgs = imgs / 255.0
                targets = targets.to(device)
                # Model forward
                preds = self.model(imgs)

                # Loss calculation
                iou, obj, cls, total = self.loss_function(preds, targets)
                # Backpropagation
                total.backward()
                # Update model parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Learning rate warmup
                for g in self.optimizer.param_groups:
                    warmup_num =  5 * len(self.train_dataloader)
                    if batch_num <= warmup_num:
                        scale = math.pow(batch_num/warmup_num, 4)
                        g['lr'] = self.cfg.learn_rate * scale
                    lr = g["lr"]

                # Log training info
                info = "Epoch:%d LR:%f IOU:%f Obj:%f Cls:%f Total:%f" % (
                        epoch, lr, iou, obj, cls, total)
                pbar.set_description(info)
                batch_num += 1

            # Validate and save model
            if epoch % 10 == 0 and epoch > 0:
                # Model evaluation
                self.model.eval()
                print("compute mAP...")
                mAP05 = self.evaluation.compute_map(self.val_dataloader, self.model)
                save_name = "weight_AP05-{:.6f}_{}-epoch.pth".format(mAP05, epoch)
                save_path = os.path.join(self.exp_dir, save_name)
                torch.save(self.model.state_dict(), save_path)
                self._prune_checkpoints()
                if mAP05 > self.best_map05:
                    self.best_map05 = mAP05
                    best_name = "best_AP05-{:.6f}_{}-epoch.pth".format(mAP05, epoch)
                    best_path = os.path.join(self.exp_dir, best_name)
                    torch.save(self.model.state_dict(), best_path)
                    self._prune_best_checkpoints()

            # Adjust learning rate
            self.scheduler.step()

if __name__ == "__main__":
    model = FastestDet()
    model.train()
