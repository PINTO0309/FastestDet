import os
import math
import shlex
import torch
import argparse
import warnings
import random
import numpy as np
import torch.nn.functional as F
import cv2
from datetime import datetime
import sys
import yaml
from tqdm import tqdm
from torch import optim
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from utils.tool import *
from utils.datasets import *
from utils.resize import (
    resize_output_channels,
    resize_image_numpy,
    is_yuv422_mode,
    is_y_only_mode,
    is_y_bin_mode,
    is_y_tri_mode,
)
from utils.evaluation import CocoDetectionEvaluator

from module.loss import DetectorLoss
from module.detector import Detector, normalize_pyramid_levels

# Suppress noisy future warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
BASE_STAGE_REPEATS = [4, 8, 4]
BASE_STAGE_OUT_CHANNELS = [-1, 24, 48, 96, 192]

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

class _TeeStream:
    def __init__(self, *streams):
        self.streams = streams
        self.primary = streams[0] if streams else None

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return getattr(self.primary, "isatty", lambda: False)()

def _validate_eighth_step(value, name):
    if value is None:
        return
    if abs(value * 8 - round(value * 8)) > 1e-6:
        raise ValueError(f"{name} must be in 0.125 increments.")

def _scale_stage_list(values, mult, keep_first=False):
    if mult is None:
        return list(values)
    scaled = []
    for i, v in enumerate(values):
        if keep_first and i == 0 and v < 0:
            scaled.append(v)
            continue
        scaled.append(max(1, int(round(v * mult))))
    return scaled

def _parse_img_size(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("img-size must be a string in HxW format.")
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise ValueError("img-size must be in HxW format.")
    try:
        height = int(parts[0])
        width = int(parts[1])
    except ValueError as exc:
        raise ValueError("img-size must be in HxW format.") from exc
    if height <= 0 or width <= 0:
        raise ValueError("img-size values must be positive.")
    return height, width

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
        parser.add_argument('--aug-yaml', type=str, default="utils/aug.yaml", help='augmentation yaml')
        parser.add_argument('--lr', type=float, default=None, help='override learning rate from yaml')
        parser.add_argument('--epoch', type=int, default=None, help='override end epoch from yaml')
        parser.add_argument('--img-size', type=str, default=None, help='override input size as HxW (height x width)')
        parser.add_argument('--val-interval', type=int, default=1, help='validation interval in epochs')
        parser.add_argument('--stage-out-channels', type=float, default=1.0, help='stage_out_channels multiplier (0.125 step)')
        parser.add_argument('--stage-repeats', type=float, default=1.0, help='stage_repeats multiplier (0.125 step)')
        parser.add_argument('--pyramid-levels', type=str, default="P1,P2,P3", help='comma-separated pyramid levels to fuse (P1,P2,P3)')
        parser.add_argument('--teacher-weight', type=str, default=None, help='teacher weight for distillation')
        parser.add_argument('--distill-weight-max', type=float, default=1.0, help='max distillation weight (cosine ramp from 0 to max)')
        parser.add_argument('--distill-temperature', type=float, default=1.5, help='distillation temperature for KL')
        parser.add_argument('--resume', type=str, default=None, help='resume checkpoint path')
        parser.add_argument('--use-ema', action='store_true', default=False, help='enable EMA for model weights')
        parser.add_argument('--ema-decay', type=float, default=0.9998, help='EMA decay rate')
        parser.add_argument('--use-amp', action='store_true', default=False, help='enable mixed precision training')
        parser.add_argument('--multi-label-robust-mode', action='store_true', default=False, help='enable multi-label robust training')
        parser.add_argument('--use-skip-residual', action='store_true', default=False, help='enable skip residual in backbone')
        se_group = parser.add_mutually_exclusive_group()
        se_group.add_argument('--use-se', action='store_true', default=False, help='enable SE on shared features')
        se_group.add_argument('--use-ese', action='store_true', default=False, help='enable eSE on shared features')
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
        if opt.teacher_weight is not None:
            assert os.path.exists(opt.teacher_weight), "Please provide a valid teacher weight path."
        if opt.resume is not None:
            assert os.path.exists(opt.resume), "Please provide a valid resume checkpoint path."
        self.cli_params = vars(opt)
        with open(opt.yaml, 'r', encoding='utf-8') as f:
            self.yaml_params = yaml.safe_load(f) or {}

        if opt.resume is not None:
            self.exp_dir = os.path.dirname(os.path.abspath(opt.resume))
        else:
            self.exp_dir = os.path.join("runs", opt.exp_name)
        self.is_resume = opt.resume is not None
        os.makedirs(self.exp_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.exp_dir)
        self.log_path = os.path.join(self.exp_dir, "train.log")
        self.log_file = open(self.log_path, "a", encoding="utf-8")
        self._enable_console_log()

        # Parse yaml config
        self.cfg = LoadYaml(opt.yaml)
        img_size = _parse_img_size(opt.img_size)
        if img_size is not None:
            self.cfg.input_height, self.cfg.input_width = img_size
        cli_classes = parse_classes(opt.classes)
        if cli_classes is not None:
            self.cfg.classes = cli_classes
            self.cfg.refresh_category_num()
        if opt.lr is not None:
            self.cfg.learn_rate = float(opt.lr)
        if opt.epoch is not None:
            self.cfg.end_epoch = int(opt.epoch)
        self.val_interval = max(1, int(opt.val_interval))
        self.render_priority_rules = self._normalize_render_priority_rules(self.cfg.render_priority_rules)
        self.render_label_ids = self._normalize_render_label_ids(self.cfg.render_label_ids)
        score_ids_defined = self._render_key_defined("SCORE_IDS")
        if score_ids_defined:
            self.render_score_ids = self._normalize_render_score_ids(self.cfg.render_score_ids)
        else:
            self.render_score_ids = None
        self.resize_mode = opt.resize_mode
        self.aug_yaml = opt.aug_yaml if opt.aug_yaml else None
        self.input_channels = resize_output_channels(self.resize_mode)
        _validate_eighth_step(opt.stage_out_channels, "stage_out_channels")
        _validate_eighth_step(opt.stage_repeats, "stage_repeats")
        self.stage_out_channels = _scale_stage_list(BASE_STAGE_OUT_CHANNELS, opt.stage_out_channels, keep_first=True)
        self.stage_repeats = _scale_stage_list(BASE_STAGE_REPEATS, opt.stage_repeats)
        self.pyramid_levels = normalize_pyramid_levels(opt.pyramid_levels)
        self.best_map05 = float("-inf")
        self.latest_map05 = None
        self.best_epochs = []
        self.start_epoch = 0
        self.batch_num = 0
        self.checkpoint_meta = None
        self.label_names = self._load_label_names()
        self._log_start()
        print(self.cfg)

        self.use_ema = opt.use_ema
        self.ema_decay = float(opt.ema_decay)
        self.ema = None
        self.use_amp = opt.use_amp and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.multi_label_robust_mode = opt.multi_label_robust_mode
        self.use_skip_residual = opt.use_skip_residual
        self.use_se = opt.use_se
        self.use_ese = opt.use_ese
        self.distill_weight_max = float(opt.distill_weight_max)
        if self.distill_weight_max < 0:
            raise ValueError("distill_weight_max must be >= 0.")
        self.distill_temp = float(opt.distill_temperature)
        if self.distill_temp <= 0:
            raise ValueError("distill_temp must be > 0.")
        self.onnx_exported = False

        # Initialize model
        if opt.weight is not None:
            print("load weight from:%s"%opt.weight)
            self.model = Detector(
                self.cfg.category_num,
                True,
                self.input_channels,
                stage_repeats=self.stage_repeats,
                stage_out_channels=self.stage_out_channels,
                use_skip_residual=self.use_skip_residual,
                use_ese=self.use_ese,
                use_se=self.use_se,
                multi_label=self.multi_label_robust_mode,
                pyramid_levels=self.pyramid_levels,
            ).to(device)
            weight_data = torch.load(opt.weight, map_location=device)
            weight_state = weight_data
            if isinstance(weight_data, dict):
                if "model" in weight_data:
                    weight_state = weight_data["model"]
                elif "state_dict" in weight_data:
                    weight_state = weight_data["state_dict"]
            self.model.load_state_dict(weight_state)
        else:
            self.model = Detector(
                self.cfg.category_num,
                False,
                self.input_channels,
                stage_repeats=self.stage_repeats,
                stage_out_channels=self.stage_out_channels,
                use_skip_residual=self.use_skip_residual,
                use_ese=self.use_ese,
                use_se=self.use_se,
                multi_label=self.multi_label_robust_mode,
                pyramid_levels=self.pyramid_levels,
            ).to(device)

        if self.use_ema:
            self.ema = EMA(self.model, decay=self.ema_decay)
            self.ema.register()

        self.teacher_model = None
        self.teacher_stage_out_channels = None
        self.teacher_stage_repeats = None
        if opt.teacher_weight:
            teacher_ckpt = torch.load(opt.teacher_weight, map_location=device)
            ckpt_stage_out = None
            ckpt_stage_repeats = None
            ckpt_state = None
            ckpt_is_teacher = False
            if isinstance(teacher_ckpt, dict):
                if "teacher_stage_out_channels" in teacher_ckpt and "teacher_stage_repeats" in teacher_ckpt:
                    ckpt_stage_out = teacher_ckpt["teacher_stage_out_channels"]
                    ckpt_stage_repeats = teacher_ckpt["teacher_stage_repeats"]
                    ckpt_state = teacher_ckpt.get("teacher_model") or teacher_ckpt.get("model")
                    ckpt_is_teacher = True
                elif "stage_out_channels" in teacher_ckpt and "stage_repeats" in teacher_ckpt and "model" in teacher_ckpt:
                    ckpt_stage_out = teacher_ckpt["stage_out_channels"]
                    ckpt_stage_repeats = teacher_ckpt["stage_repeats"]
                    ckpt_state = teacher_ckpt.get("model")
            use_ckpt_backbone = (
                ckpt_stage_out is not None
                and ckpt_stage_repeats is not None
                and ckpt_state is not None
            )
            teacher_use_skip = self.use_skip_residual
            teacher_use_se = self.use_se
            teacher_use_ese = self.use_ese
            if isinstance(teacher_ckpt, dict):
                if ckpt_is_teacher:
                    if "teacher_use_skip_residual" in teacher_ckpt:
                        teacher_use_skip = bool(teacher_ckpt["teacher_use_skip_residual"])
                    if "teacher_use_se" in teacher_ckpt:
                        teacher_use_se = bool(teacher_ckpt["teacher_use_se"])
                    if "teacher_use_ese" in teacher_ckpt:
                        teacher_use_ese = bool(teacher_ckpt["teacher_use_ese"])
                else:
                    if "use_skip_residual" in teacher_ckpt:
                        teacher_use_skip = bool(teacher_ckpt["use_skip_residual"])
                    if "use_se" in teacher_ckpt:
                        teacher_use_se = bool(teacher_ckpt["use_se"])
                    if "use_ese" in teacher_ckpt:
                        teacher_use_ese = bool(teacher_ckpt["use_ese"])
            if teacher_use_se and teacher_use_ese:
                raise ValueError("Teacher model cannot enable both SE and eSE.")
            if not use_ckpt_backbone:
                raise ValueError("Teacher checkpoint must include backbone metadata (stage_out_channels/stage_repeats).")
            teacher_out_channels = ckpt_stage_out
            teacher_repeats = ckpt_stage_repeats
            teacher_state = ckpt_state

            self.teacher_stage_out_channels = teacher_out_channels
            self.teacher_stage_repeats = teacher_repeats
            self.teacher_use_skip_residual = teacher_use_skip
            self.teacher_use_se = teacher_use_se
            self.teacher_use_ese = teacher_use_ese
            self.teacher_model = Detector(
                self.cfg.category_num,
                True,
                self.input_channels,
                stage_repeats=teacher_repeats,
                stage_out_channels=teacher_out_channels,
                use_skip_residual=teacher_use_skip,
                use_ese=teacher_use_ese,
                use_se=teacher_use_se,
                multi_label=self.multi_label_robust_mode,
                pyramid_levels=self.pyramid_levels,
            ).to(device)
            self.teacher_model.load_state_dict(teacher_state)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        self.is_distilling = self.teacher_model is not None

        # # Print tensor shapes of network layers
        summary(self.model, input_size=(self.input_channels, self.cfg.input_height, self.cfg.input_width))

        # Build optimizer
        opt_name = getattr(self.cfg, "optimizer", "sgd")
        opt_name = str(opt_name).lower()
        weight_decay = float(getattr(self.cfg, "weight_decay", 0.0005))
        if opt_name == "adamw":
            print("use AdamW optimizer")
            self.optimizer = optim.AdamW(
                params=self.model.parameters(),
                lr=self.cfg.learn_rate,
                weight_decay=weight_decay,
            )
        else:
            print("use SGD optimizer")
            self.optimizer = optim.SGD(
                params=self.model.parameters(),
                lr=self.cfg.learn_rate,
                momentum=0.949,
                weight_decay=weight_decay,
            )
        # Learning rate decay schedule
        sched_name = str(getattr(self.cfg, "scheduler", "multistep")).lower()
        gamma = float(getattr(self.cfg, "scheduler_gamma", 0.1))
        if sched_name == "step":
            step_size = int(getattr(self.cfg, "scheduler_step_size", 10))
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif sched_name == "cosine":
            t_max = int(getattr(self.cfg, "scheduler_t_max", self.cfg.end_epoch))
            eta_min = float(getattr(self.cfg, "scheduler_min_lr", 0.0))
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=eta_min)
        elif sched_name == "none":
            self.scheduler = None
        else:
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=self.cfg.milestones,
                                                            gamma=gamma)

        # Define loss
        self.loss_function = DetectorLoss(device, multi_label=self.multi_label_robust_mode)

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
        self.val_preview_paths = list(val_dataset.data_list[:10])
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
        self.data_gen = torch.Generator()
        self.data_gen.manual_seed(SEED)
        self.val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=12,
            drop_last=False,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=self.data_gen,
            pin_memory=True,
        )
        # Training set
        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=12,
            persistent_workers=True,
            worker_init_fn=seed_worker,
            generator=self.data_gen,
            pin_memory=True,
        )
        if opt.resume is not None:
            self._load_checkpoint(opt.resume)

        self._disable_console_log()

    def _prune_checkpoints(self, max_keep=10):
        checkpoints = []
        for name in os.listdir(self.exp_dir):
            if not name.startswith("last_") or not name.endswith(".pth"):
                continue
            path = os.path.join(self.exp_dir, name)
            epoch_num = None
            try:
                parts = name.split("_")
                if len(parts) >= 3:
                    epoch_num = int(parts[1])
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
            if not name.startswith("best_") or not name.endswith(".pth"):
                continue
            path = os.path.join(self.exp_dir, name)
            epoch_num = None
            try:
                parts = name.split("_")
                if len(parts) >= 3:
                    epoch_num = int(parts[1])
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

    def _prune_render_dirs(self, max_keep_latest=10, max_keep_best=10):
        epoch_dirs = []
        for name in os.listdir(self.exp_dir):
            if len(name) != 4 or not name.isdigit():
                continue
            epoch_dirs.append(int(name))
        if not epoch_dirs:
            return
        epoch_dirs.sort()
        if self.is_distilling:
            keep = set(epoch_dirs[-max_keep_latest:]) if max_keep_latest > 0 else set()
            if self.best_epochs:
                keep.add(self.best_epochs[-1])
        else:
            keep = set(self.best_epochs[-max_keep_best:]) if max_keep_best > 0 else set()
        for epoch_num in epoch_dirs:
            if epoch_num in keep:
                continue
            dir_path = os.path.join(self.exp_dir, f"{epoch_num:04d}")
            if not os.path.isdir(dir_path):
                continue
            try:
                for root, _, files in os.walk(dir_path, topdown=False):
                    for filename in files:
                        try:
                            os.remove(os.path.join(root, filename))
                        except OSError:
                            pass
                os.rmdir(dir_path)
            except OSError:
                pass

    def _enable_console_log(self):
        if getattr(self, "_console_tee_enabled", False):
            return
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = _TeeStream(self._stdout, self.log_file)
        sys.stderr = _TeeStream(self._stderr, self.log_file)
        self._console_tee_enabled = True

    def _disable_console_log(self):
        if not getattr(self, "_console_tee_enabled", False):
            return
        sys.stdout = self._stdout
        sys.stderr = self._stderr
        self._console_tee_enabled = False

    def _log_line(self, line):
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def _log_start(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._log_line(f"=== run start {ts} ===")
        cli_command = " ".join(shlex.quote(arg) for arg in sys.argv)
        if cli_command:
            self._log_line(f"cli.command={cli_command}")
        for key in sorted(self.cli_params.keys()):
            if key == "yaml":
                self._log_line(f"cli.{key}={self.cli_params[key]!r}")
                if self.yaml_params:
                    yaml_text = yaml.safe_dump(self.yaml_params, sort_keys=False)
                    for line in yaml_text.rstrip().splitlines():
                        self._log_line(f"yaml.{line}")
                continue
            self._log_line(f"cli.{key}={self.cli_params[key]!r}")
        self._log_line("")

    def _log_epoch(self, epoch, lr, train_metrics, distill_metrics, val_map05, last_name, best_name):
        parts = [
            f"epoch={epoch:04d}",
            f"lr={lr:.6f}",
            f"train_total={train_metrics['total']:.6f}",
            f"train_iou={train_metrics['iou']:.6f}",
            f"train_obj={train_metrics['obj']:.6f}",
            f"train_cls={train_metrics['cls']:.6f}",
        ]
        if distill_metrics is not None:
            if "weight" in distill_metrics:
                parts.append(f"distill_weight={distill_metrics['weight']:.4f}")
            parts.extend([
                f"distill_total={distill_metrics['total']:.6f}",
                f"distill_obj={distill_metrics['obj']:.6f}",
                f"distill_box={distill_metrics['box']:.6f}",
                f"distill_cls={distill_metrics['cls']:.6f}",
            ])
        if val_map05 is None:
            parts.append("val_mAP50=na")
        else:
            parts.append(f"val_mAP50={val_map05:.6f}")
        parts.append(f"best_mAP50={self.best_map05:.6f}")
        if last_name:
            parts.append(f"last_ckpt={last_name}")
        if best_name:
            parts.append(f"best_ckpt={best_name}")
        self._log_line(" ".join(parts))

    def _export_onnx(self, path):
        self.model.eval()
        dummy = torch.zeros(
            1,
            self.input_channels,
            self.cfg.input_height,
            self.cfg.input_width,
            device=device,
        )
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow()
        torch.onnx.export(
            self.model,
            dummy,
            path,
            export_params=True,
            opset_version=17,
            input_names=["input_rgb"],
            output_names=["output"],
        )
        import onnx
        from onnxsim import simplify
        onnx_model = onnx.load(path)
        model_simp, check = simplify(onnx_model)
        if not check:
            raise RuntimeError("onnxsim simplification check failed.")
        onnx.save(model_simp, path)
        if self.use_ema and self.ema is not None:
            self.ema.restore()
        print(f"export onnx: {path}")
        print("onnx sim success...")
        self._log_line(f"export_onnx={path}")

    def _save_checkpoint(self, epoch, path):
        state = {
            "epoch": epoch,
            "batch_num": self.batch_num,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "category_num": self.cfg.category_num,
            "best_map05": self.best_map05,
            "best_epochs": self.best_epochs,
            "latest_map05": self.latest_map05,
            "val_interval": self.val_interval,
            "resize_mode": self.resize_mode,
            "aug_yaml": self.aug_yaml,
            "classes": self.cfg.classes,
            "stage_out_channels": self.stage_out_channels,
            "stage_repeats": self.stage_repeats,
            "pyramid_levels": self.pyramid_levels,
            "input_channels": self.input_channels,
            "use_skip_residual": self.use_skip_residual,
            "use_se": self.use_se,
            "use_ese": self.use_ese,
            "use_ema": self.use_ema,
            "use_amp": self.use_amp,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "data_gen_state": self.data_gen.get_state() if hasattr(self, "data_gen") else None,
            "meta": {
                "yaml_path": self.cli_params.get("yaml"),
                "yaml": self.yaml_params,
                "cli": self.cli_params,
                "derived": {
                    "resize_mode": self.resize_mode,
                    "stage_out_channels": self.stage_out_channels,
                    "stage_repeats": self.stage_repeats,
                    "input_channels": self.input_channels,
                    "category_num": self.cfg.category_num,
                    "pyramid_levels": self.pyramid_levels,
                    "use_ema": self.use_ema,
                    "use_amp": self.use_amp,
                },
            },
        }
        if self.teacher_model is not None:
            state["teacher_model"] = self.teacher_model.state_dict()
            state["teacher_stage_out_channels"] = self.teacher_stage_out_channels
            state["teacher_stage_repeats"] = self.teacher_stage_repeats
            state["teacher_use_skip_residual"] = getattr(self, "teacher_use_skip_residual", None)
            state["teacher_use_se"] = getattr(self, "teacher_use_se", None)
            state["teacher_use_ese"] = getattr(self, "teacher_use_ese", None)
            state["teacher_pyramid_levels"] = self.pyramid_levels
        if self.use_ema and self.ema is not None:
            state["ema_shadow"] = self.ema.shadow
            state["ema_decay"] = self.ema.decay
        if self.use_amp and self.scaler is not None:
            state["amp_scaler"] = self.scaler.state_dict()
        torch.save(state, path)

    def _coerce_rng_state(self, state, name):
        if state is None:
            return None
        if isinstance(state, torch.Tensor):
            return state.to(dtype=torch.uint8, device="cpu")
        if isinstance(state, np.ndarray):
            return torch.from_numpy(state).to(dtype=torch.uint8, device="cpu")
        if isinstance(state, (bytes, bytearray)):
            return torch.ByteTensor(list(state))
        if isinstance(state, (list, tuple)):
            return torch.tensor(state, dtype=torch.uint8)
        raise TypeError(f"{name} must be a torch.ByteTensor or convertible type, got {type(state)}")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        required_keys = {
            "model",
            "optimizer",
            "scheduler",
            "epoch",
            "batch_num",
            "rng_state",
            "numpy_rng_state",
            "python_rng_state",
            "data_gen_state",
        }
        if not isinstance(checkpoint, dict) or not required_keys.issubset(checkpoint.keys()):
            raise ValueError("Resume requires a full checkpoint saved in last_*.pth.")
        self.checkpoint_meta = checkpoint.get("meta")
        ckpt_stage_out = checkpoint.get("stage_out_channels")
        ckpt_stage_repeats = checkpoint.get("stage_repeats")
        ckpt_input_channels = checkpoint.get("input_channels")
        ckpt_use_skip = checkpoint.get("use_skip_residual")
        ckpt_use_se = checkpoint.get("use_se")
        ckpt_use_ese = checkpoint.get("use_ese")
        ckpt_pyramid_levels = checkpoint.get("pyramid_levels")
        ckpt_category_num = checkpoint.get("category_num")
        if ckpt_stage_out is not None and ckpt_stage_out != self.stage_out_channels:
            raise ValueError("stage_out_channels mismatch with checkpoint.")
        if ckpt_stage_repeats is not None and ckpt_stage_repeats != self.stage_repeats:
            raise ValueError("stage_repeats mismatch with checkpoint.")
        if ckpt_input_channels is not None and ckpt_input_channels != self.input_channels:
            raise ValueError("input_channels mismatch with checkpoint.")
        if ckpt_use_skip is not None and ckpt_use_skip != self.use_skip_residual:
            raise ValueError("use_skip_residual mismatch with checkpoint.")
        if ckpt_use_se is not None and ckpt_use_se != self.use_se:
            raise ValueError("use_se mismatch with checkpoint.")
        if ckpt_use_ese is not None and ckpt_use_ese != self.use_ese:
            raise ValueError("use_ese mismatch with checkpoint.")
        if ckpt_pyramid_levels is not None and tuple(ckpt_pyramid_levels) != self.pyramid_levels:
            raise ValueError("pyramid_levels mismatch with checkpoint.")
        if ckpt_category_num is not None and ckpt_category_num != self.cfg.category_num:
            raise ValueError("category_num mismatch with checkpoint.")
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        ckpt_use_ema = checkpoint.get("use_ema", False)
        ckpt_use_amp = checkpoint.get("use_amp", False)
        if ckpt_use_ema and not self.use_ema:
            self.use_ema = True
            self.ema = EMA(self.model, decay=checkpoint.get("ema_decay", 0.9998))
            self.ema.register()
        if self.use_ema and self.ema is not None and "ema_shadow" in checkpoint:
            self.ema.shadow = checkpoint["ema_shadow"]
        if self.use_ema and self.ema is not None and "ema_decay" in checkpoint:
            self.ema.decay = checkpoint["ema_decay"]
        if ckpt_use_amp and not self.use_amp and torch.cuda.is_available():
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        if self.use_amp and self.scaler is not None and "amp_scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["amp_scaler"])
        if "teacher_model" in checkpoint and self.teacher_model is None:
            teacher_out_channels = checkpoint.get("teacher_stage_out_channels")
            teacher_repeats = checkpoint.get("teacher_stage_repeats")
            teacher_use_skip = checkpoint.get("teacher_use_skip_residual")
            teacher_use_se = checkpoint.get("teacher_use_se")
            teacher_use_ese = checkpoint.get("teacher_use_ese")
            teacher_pyramid_levels = checkpoint.get("teacher_pyramid_levels", self.pyramid_levels)
            if teacher_out_channels is None or teacher_repeats is None:
                raise ValueError("Checkpoint is missing teacher backbone settings.")
            self.teacher_stage_out_channels = teacher_out_channels
            self.teacher_stage_repeats = teacher_repeats
            self.teacher_use_skip_residual = bool(teacher_use_skip) if teacher_use_skip is not None else False
            self.teacher_use_se = bool(teacher_use_se) if teacher_use_se is not None else False
            self.teacher_use_ese = bool(teacher_use_ese) if teacher_use_ese is not None else False
            if self.teacher_use_se and self.teacher_use_ese:
                raise ValueError("Checkpoint enables both SE and eSE for teacher.")
            self.teacher_model = Detector(
                self.cfg.category_num,
                True,
                self.input_channels,
                stage_repeats=teacher_repeats,
                stage_out_channels=teacher_out_channels,
                use_skip_residual=self.teacher_use_skip_residual,
                use_se=self.teacher_use_se,
                use_ese=self.teacher_use_ese,
                multi_label=self.multi_label_robust_mode,
                pyramid_levels=teacher_pyramid_levels,
            ).to(device)
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad = False
        if self.teacher_model is not None and "teacher_model" in checkpoint:
            self.teacher_model.load_state_dict(checkpoint["teacher_model"])
        self.best_map05 = checkpoint.get("best_map05", self.best_map05)
        self.best_epochs = checkpoint.get("best_epochs", self.best_epochs)
        self.latest_map05 = checkpoint.get("latest_map05", self.latest_map05)
        self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
        self.batch_num = int(checkpoint.get("batch_num", 0))
        if "rng_state" in checkpoint:
            rng_state = self._coerce_rng_state(checkpoint["rng_state"], "rng_state")
            if rng_state is not None:
                torch.set_rng_state(rng_state)
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            cuda_states = checkpoint["cuda_rng_state"]
            if isinstance(cuda_states, (list, tuple)):
                cuda_states = [self._coerce_rng_state(state, "cuda_rng_state") for state in cuda_states]
            else:
                cuda_states = [self._coerce_rng_state(cuda_states, "cuda_rng_state")]
            torch.cuda.set_rng_state_all(cuda_states)
        if "numpy_rng_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_rng_state"])
        if "python_rng_state" in checkpoint:
            random.setstate(checkpoint["python_rng_state"])
        if hasattr(self, "data_gen") and checkpoint.get("data_gen_state") is not None:
            data_gen_state = self._coerce_rng_state(checkpoint["data_gen_state"], "data_gen_state")
            if data_gen_state is not None:
                self.data_gen.set_state(data_gen_state)

    def _load_label_names(self):
        names = []
        if hasattr(self, "cfg") and self.cfg.names and os.path.exists(self.cfg.names):
            with open(self.cfg.names, 'r') as f:
                for line in f.readlines():
                    name = line.strip()
                    if name:
                        names.append(name)
        return names

    def _color_for_class(self, class_id):
        # Deterministic vivid colors (BGR) for readability.
        palette = [
            (255, 99, 71),   # tomato
            (0, 255, 255),   # yellow
            (50, 205, 50),   # lime green
            (255, 215, 0),   # gold
            (30, 144, 255),  # dodger blue
            (255, 105, 180), # hot pink
            (0, 191, 255),   # deep sky blue
            (154, 205, 50),  # yellow green
            (255, 165, 0),   # orange
            (138, 43, 226),  # blue violet
            (64, 224, 208),  # turquoise
            (220, 20, 60),   # crimson
        ]
        if class_id < 0:
            class_id = 0
        return palette[class_id % len(palette)]

    def _normalize_render_priority_rules(self, rules):
        if not rules:
            return []
        normalized = []
        class_map = None
        if self.cfg.classes is not None:
            class_map = {int(cid): idx for idx, cid in enumerate(self.cfg.classes)}
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            parents = parse_classes(rule.get("parent", rule.get("parents")))
            children = parse_classes(rule.get("children", rule.get("child")))
            if not parents or not children:
                continue
            mapped_parents = []
            for pid in parents:
                try:
                    pid = int(pid)
                except (TypeError, ValueError):
                    continue
                if class_map is not None:
                    if pid not in class_map:
                        continue
                    pid = class_map[pid]
                mapped_parents.append(pid)
            mapped_children = []
            for cid in children:
                try:
                    cid = int(cid)
                except (TypeError, ValueError):
                    continue
                if class_map is not None:
                    if cid not in class_map:
                        continue
                    cid = class_map[cid]
                mapped_children.append(cid)
            if not mapped_parents or not mapped_children:
                continue
            try:
                iou = float(rule.get("iou", 0.9))
            except (TypeError, ValueError):
                iou = 0.9
            iou = min(max(iou, 0.0), 1.0)
            require_parent_match = rule.get("require_parent_match", False)
            if isinstance(require_parent_match, str):
                require_parent_match = require_parent_match.strip().lower() in ("1", "true", "yes", "y", "on")
            require_parent_match = bool(require_parent_match)
            use_parent_box = rule.get("use_parent_box", False)
            if isinstance(use_parent_box, str):
                use_parent_box = use_parent_box.strip().lower() in ("1", "true", "yes", "y", "on")
            use_parent_box = bool(use_parent_box)
            normalized.append({
                "parents": mapped_parents,
                "children": mapped_children,
                "iou": iou,
                "require_parent_match": require_parent_match,
                "use_parent_box": use_parent_box,
            })
        return normalized

    def _normalize_render_label_ids(self, label_ids):
        if label_ids is None:
            return None
        if isinstance(label_ids, str) and not label_ids.strip():
            return set()
        ids = parse_classes(label_ids)
        if not ids:
            return set()
        class_map = None
        if self.cfg.classes is not None:
            class_map = {int(cid): idx for idx, cid in enumerate(self.cfg.classes)}
        normalized = set()
        for cid in ids:
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            if class_map is not None:
                if cid not in class_map:
                    continue
                cid = class_map[cid]
            normalized.add(cid)
        return normalized

    def _render_key_defined(self, key):
        if not isinstance(self.yaml_params, dict):
            return False
        render_cfg = self.yaml_params.get("RENDER")
        return isinstance(render_cfg, dict) and key in render_cfg

    def _normalize_render_score_ids(self, score_ids):
        if score_ids is None:
            return set()
        if isinstance(score_ids, str) and not score_ids.strip():
            return set()
        ids = parse_classes(score_ids)
        if not ids:
            return set()
        class_map = None
        if self.cfg.classes is not None:
            class_map = {int(cid): idx for idx, cid in enumerate(self.cfg.classes)}
        normalized = set()
        for cid in ids:
            try:
                cid = int(cid)
            except (TypeError, ValueError):
                continue
            if class_map is not None:
                if cid not in class_map:
                    continue
                cid = class_map[cid]
            normalized.add(cid)
        return normalized

    def _bbox_iou_one_to_many(self, box, boxes):
        px1, py1, px2, py2 = box
        cx1, cy1, cx2, cy2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        inter_w = np.maximum(0.0, np.minimum(px2, cx2) - np.maximum(px1, cx1))
        inter_h = np.maximum(0.0, np.minimum(py2, cy2) - np.maximum(py1, cy1))
        inter = inter_w * inter_h
        p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        c_area = np.maximum(0.0, cx2 - cx1) * np.maximum(0.0, cy2 - cy1)
        union = p_area + c_area - inter + 1e-9
        return inter / union

    def _apply_render_priority_rules(self, boxes):
        if not self.render_priority_rules:
            return boxes
        keep = np.ones(len(boxes), dtype=bool)
        for rule in self.render_priority_rules:
            parents = rule["parents"]
            children = rule["children"]
            thresh = rule["iou"]
            require_parent_match = rule.get("require_parent_match", False)
            use_parent_box = rule.get("use_parent_box", False)
            if not parents or not children:
                continue
            parent_idx = [i for i in range(len(boxes)) if keep[i] and int(boxes[i, 5]) in parents]
            child_idx = [i for i in range(len(boxes)) if keep[i] and int(boxes[i, 5]) in children]
            if not parent_idx or not child_idx:
                if require_parent_match and child_idx:
                    for i in child_idx:
                        keep[i] = False
                continue
            child_boxes = boxes[child_idx, :4]
            parent_boxes = boxes[parent_idx, :4]
            iou_matrix = np.stack(
                [self._bbox_iou_one_to_many(child_box, parent_boxes) for child_box in child_boxes],
                axis=0,
            )
            best_parent = iou_matrix.argmax(axis=1)
            best_iou = iou_matrix[np.arange(iou_matrix.shape[0]), best_parent]
            if require_parent_match:
                for row, idx in enumerate(child_idx):
                    if best_iou[row] < thresh:
                        keep[idx] = False
            child_keep_mask = np.array([keep[i] for i in child_idx], dtype=bool)
            matched_mask = child_keep_mask & (best_iou >= thresh)
            if matched_mask.any():
                child_scores = boxes[child_idx, 4]
                for col, p_idx in enumerate(parent_idx):
                    rows = np.where(matched_mask & (best_parent == col))[0]
                    if rows.size == 0:
                        continue
                    if rows.size > 1:
                        best_row = rows[np.argmax(child_scores[rows])]
                        drop_rows = rows[rows != best_row]
                        for row in drop_rows:
                            keep[child_idx[row]] = False
                    else:
                        best_row = rows[0]
                    if use_parent_box:
                        boxes[child_idx[best_row], :4] = boxes[p_idx, :4]
                    keep[p_idx] = False
        return boxes[keep]

    def _prepare_infer_input(self, img_bgr):
        if self.resize_mode is None:
            resized = cv2.resize(
                img_bgr,
                (self.cfg.input_width, self.cfg.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            img = resized.astype(np.float32) / 255.0
        else:
            img_float = img_bgr.astype(np.float32) / 255.0
            if (
                is_yuv422_mode(self.resize_mode)
                or is_y_only_mode(self.resize_mode)
                or is_y_bin_mode(self.resize_mode)
                or is_y_tri_mode(self.resize_mode)
            ):
                img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
            img = resize_image_numpy(
                img_float,
                (self.cfg.input_height, self.cfg.input_width),
                self.resize_mode,
            )
        img = np.ascontiguousarray(img)
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
        return tensor

    def _render_val_predictions(self, epoch, max_images=10):
        out_dir = os.path.join(self.exp_dir, f"{epoch:04d}")
        os.makedirs(out_dir, exist_ok=True)
        image_paths = self.val_preview_paths[:max_images] if self.val_preview_paths else []
        self.model.eval()
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            input_tensor = self._prepare_infer_input(img)
            with torch.no_grad():
                preds = self.model(input_tensor)
                output = handle_preds(preds, device, multi_label=self.multi_label_robust_mode)
            if not output:
                continue
            boxes = output[0].cpu().numpy() if output[0].numel() else []
            if len(boxes):
                boxes = self._apply_render_priority_rules(boxes)
            h, w = img.shape[:2]
            for box in boxes:
                x1, y1, x2, y2, score, cls_id = box.tolist()
                x1 = max(0, min(w - 1, int(x1 * w)))
                y1 = max(0, min(h - 1, int(y1 * h)))
                x2 = max(0, min(w - 1, int(x2 * w)))
                y2 = max(0, min(h - 1, int(y2 * h)))
                cls_id = int(cls_id)
                label_allowed = True
                if self.render_label_ids is not None:
                    label_allowed = cls_id in self.render_label_ids
                label = None
                if label_allowed:
                    label = self.label_names[cls_id] if cls_id < len(self.label_names) else str(cls_id)
                color = self._color_for_class(cls_id)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,255), 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                if label:
                    show_score = True
                    if self.render_score_ids is not None:
                        show_score = cls_id in self.render_score_ids
                    text = f"{label}:{score:.2f}" if show_score else label
                    cv2.putText(img, text, (x1, max(0, y1 - 5)), 0, 0.6, (255,255,255), 2)
                    cv2.putText(img, text, (x1, max(0, y1 - 5)), 0, 0.6, color, 1)
            save_path = os.path.join(out_dir, os.path.basename(path))
            cv2.imwrite(save_path, img)

    def _distill_loss(self, preds, teacher_preds, student_logits=None, teacher_logits=None):
        pred = preds.permute(0, 2, 3, 1)
        teacher = teacher_preds.permute(0, 2, 3, 1)
        pobj = pred[..., 0]
        tobj = teacher[..., 0]
        preg = pred[..., 1:5]
        treg = teacher[..., 1:5]
        pcls = pred[..., 5:]
        tcls = teacher[..., 5:]
        obj_loss = F.mse_loss(pobj, tobj)
        box_loss = F.smooth_l1_loss(preg, treg)
        if self.multi_label_robust_mode:
            if student_logits is not None and teacher_logits is not None:
                slogits = student_logits.permute(0, 2, 3, 1)
                tlogits = teacher_logits.permute(0, 2, 3, 1)
                cls_loss = F.binary_cross_entropy_with_logits(slogits, torch.sigmoid(tlogits))
            else:
                cls_loss = F.binary_cross_entropy(pcls, tcls)
        else:
            if student_logits is not None and teacher_logits is not None:
                slogits = student_logits.permute(0, 2, 3, 1)
                tlogits = teacher_logits.permute(0, 2, 3, 1)
                temp = self.distill_temp
                s_log_prob = F.log_softmax(slogits / temp, dim=-1)
                t_prob = F.softmax(tlogits / temp, dim=-1)
                cls_loss = F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (temp * temp)
            else:
                cls_loss = F.kl_div(torch.log(pcls + 1e-9), tcls, reduction="batchmean")
        total = obj_loss + box_loss + cls_loss
        return obj_loss, box_loss, cls_loss, total

    def _distill_weight(self, epoch):
        if self.distill_weight_max <= 0:
            return 0.0
        total_epochs = max(1, int(self.cfg.end_epoch))
        if total_epochs == 1:
            return self.distill_weight_max
        progress = (epoch - 1) / float(total_epochs - 1)
        progress = min(max(progress, 0.0), 1.0)
        return self.distill_weight_max * 0.5 * (1.0 - math.cos(math.pi * progress))

    def train(self):
        # Training loop
        batch_num = self.batch_num
        input_is_normalized = getattr(self.train_dataloader.dataset, "input_is_normalized", False)
        start_line = "Starting training for %g epochs..." % self.cfg.end_epoch
        print(start_line)
        self._log_line(start_line)
        if not self.onnx_exported and not self.is_resume:
            export_path = os.path.join(self.exp_dir, "model.onnx")
            self._export_onnx(export_path)
            self.onnx_exported = True
        for epoch in range(self.start_epoch, self.cfg.end_epoch + 1):
            self.model.train()
            epoch_iou = 0.0
            epoch_obj = 0.0
            epoch_cls = 0.0
            epoch_total = 0.0
            epoch_distill_obj = 0.0
            epoch_distill_box = 0.0
            epoch_distill_cls = 0.0
            epoch_distill_total = 0.0
            distill_weight = 0.0
            batch_count = 0
            val_map05 = None
            last_name = None
            best_name = None
            if self.teacher_model is not None:
                distill_weight = self._distill_weight(epoch)
            pbar = tqdm(self.train_dataloader, dynamic_ncols=True)
            for imgs, targets in pbar:
                # Data preprocessing
                imgs = imgs.to(device).float()
                if not input_is_normalized:
                    imgs = imgs / 255.0
                targets = targets.to(device)
                # Model forward
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    student_logits = None
                    use_logits = self.teacher_model is not None or self.multi_label_robust_mode
                    if use_logits:
                        preds, student_logits = self.model(imgs, return_logits=True)
                    else:
                        preds = self.model(imgs)

                    # Loss calculation
                    iou, obj, cls, total = self.loss_function(preds, targets, cls_logits=student_logits)
                    distill_obj = distill_box = distill_cls = distill_total = None
                    if self.teacher_model is not None:
                        with torch.no_grad():
                            teacher_preds, teacher_logits = self.teacher_model(imgs, return_logits=True)
                        distill_obj, distill_box, distill_cls, distill_total = self._distill_loss(
                            preds,
                            teacher_preds,
                            student_logits,
                            teacher_logits,
                        )
                        total = total + distill_total * distill_weight
                # Backpropagation
                self.scaler.scale(total).backward()
                # Update model parameters
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.use_ema and self.ema is not None:
                    self.ema.update()
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
                self.batch_num = batch_num
                batch_count += 1
                epoch_iou += float(iou)
                epoch_obj += float(obj)
                epoch_cls += float(cls)
                epoch_total += float(total)
                if distill_total is not None:
                    epoch_distill_obj += float(distill_obj)
                    epoch_distill_box += float(distill_box)
                    epoch_distill_cls += float(distill_cls)
                    epoch_distill_total += float(distill_total)

            # Validate and save model
            if epoch % self.val_interval == 0 and epoch > 0:
                # Model evaluation
                self.model.eval()
                print("compute mAP...")
                if self.use_ema and self.ema is not None:
                    self.ema.apply_shadow()
                mAP05 = self.evaluation.compute_map(
                    self.val_dataloader,
                    self.model,
                    multi_label=self.multi_label_robust_mode,
                )
                self.latest_map05 = mAP05
                val_map05 = mAP05
                self.writer.add_scalar("val/010_mAP50", mAP05, epoch)
                if self.evaluation.last_per_class_ap:
                    for idx, (name, ap) in enumerate(self.evaluation.last_per_class_ap):
                        if np.isnan(ap):
                            continue
                        self.writer.add_scalar(f"val/001_AP50_{idx:03d}_{name}", ap, epoch)
                if mAP05 > self.best_map05:
                    self.best_map05 = mAP05
                    self.best_epochs.append(epoch)
                    best_name = "best_{:04d}_{:.6f}.pth".format(epoch, mAP05)
                    best_path = os.path.join(self.exp_dir, best_name)
                    self._save_checkpoint(epoch, best_path)
                    self._prune_best_checkpoints()
                self._render_val_predictions(epoch)
                if self.use_ema and self.ema is not None:
                    self.ema.restore()
                self._prune_render_dirs()

            last_map05 = self.latest_map05 if self.latest_map05 is not None else 0.0
            save_name = "last_{:04d}_{:.6f}.pth".format(epoch, last_map05)
            save_path = os.path.join(self.exp_dir, save_name)
            self._save_checkpoint(epoch, save_path)
            last_name = save_name
            self._prune_checkpoints()

            train_metrics = None
            distill_metrics = None
            if batch_count > 0:
                inv = 1.0 / batch_count
                train_metrics = {
                    "total": epoch_total * inv,
                    "iou": epoch_iou * inv,
                    "obj": epoch_obj * inv,
                    "cls": epoch_cls * inv,
                }
                self.writer.add_scalar("train/100_loss_total", train_metrics["total"], epoch)
                self.writer.add_scalar("train/101_loss_iou", train_metrics["iou"], epoch)
                self.writer.add_scalar("train/102_loss_obj", train_metrics["obj"], epoch)
                self.writer.add_scalar("train/103_loss_cls", train_metrics["cls"], epoch)
                if self.teacher_model is not None:
                    distill_metrics = {
                        "total": epoch_distill_total * inv,
                        "obj": epoch_distill_obj * inv,
                        "box": epoch_distill_box * inv,
                        "cls": epoch_distill_cls * inv,
                        "weight": distill_weight,
                    }
                    self.writer.add_scalar("train/109_distill_weight", distill_metrics["weight"], epoch)
                    self.writer.add_scalar("train/110_loss_distill_total", distill_metrics["total"], epoch)
                    self.writer.add_scalar("train/111_loss_distill_obj", distill_metrics["obj"], epoch)
                    self.writer.add_scalar("train/112_loss_distill_box", distill_metrics["box"], epoch)
                    self.writer.add_scalar("train/113_loss_distill_cls", distill_metrics["cls"], epoch)
                self.writer.add_scalar("train/120_lr", lr, epoch)

            # Adjust learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            if train_metrics is not None:
                self._log_epoch(epoch, lr, train_metrics, distill_metrics, val_map05, last_name, best_name)
        self.writer.close()
        self.log_file.close()

if __name__ == "__main__":
    model = FastestDet()
    model.train()
