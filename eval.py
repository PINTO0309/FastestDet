import os
import torch
import argparse
import warnings
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.detector import Detector, normalize_pyramid_levels

# Default backbone configuration for scaling.
BASE_STAGE_REPEATS = [4, 8, 4]
BASE_STAGE_OUT_CHANNELS = [-1, 24, 48, 96, 192]

# Suppress noisy future warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)

# Select backend device: CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

if __name__ == '__main__':
    # Training config
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None, help='.weight config')
    parser.add_argument('--stage-out-channels', type=float, default=1.0, help='stage_out_channels multiplier (0.125 step)')
    parser.add_argument('--stage-repeats', type=float, default=1.0, help='stage_repeats multiplier (0.125 step)')
    parser.add_argument('--pyramid-levels', type=str, default="P1,P2,P3", help='comma-separated pyramid levels to fuse (P1,P2,P3)')
    parser.add_argument('--multi-label-robust-mode', action='store_true', default=False, help='enable multi-label robust inference')
    parser.add_argument('--use-skip-residual', action='store_true', default=False, help='enable skip residual in backbone')
    se_group = parser.add_mutually_exclusive_group()
    se_group.add_argument('--use-se', action='store_true', default=False, help='enable SE on shared features')
    se_group.add_argument('--use-ese', action='store_true', default=False, help='enable eSE on shared features')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "Please provide a valid config file path."
    assert os.path.exists(opt.weight), "Please provide a valid weight file path."

    # Parse yaml config
    cfg = LoadYaml(opt.yaml)
    print(cfg)

    # Load model weights
    print("load weight from:%s"%opt.weight)
    _validate_eighth_step(opt.stage_out_channels, "stage_out_channels")
    _validate_eighth_step(opt.stage_repeats, "stage_repeats")
    stage_out_channels = _scale_stage_list(BASE_STAGE_OUT_CHANNELS, opt.stage_out_channels, keep_first=True)
    stage_repeats = _scale_stage_list(BASE_STAGE_REPEATS, opt.stage_repeats)
    pyramid_levels = normalize_pyramid_levels(opt.pyramid_levels)
    model = Detector(
        cfg.category_num,
        True,
        stage_repeats=stage_repeats,
        stage_out_channels=stage_out_channels,
        use_skip_residual=opt.use_skip_residual,
        use_ese=opt.use_ese,
        use_se=opt.use_se,
        multi_label=opt.multi_label_robust_mode,
        pyramid_levels=pyramid_levels,
    ).to(device)
    model.load_state_dict(torch.load(opt.weight))
    model.eval()

    # # Print tensor shapes of network layers
    summary(model, input_size=(3, cfg.input_height, cfg.input_width))

    # Define evaluation
    evaluation = CocoDetectionEvaluator(cfg.names, device)

    # Dataset loading
    val_dataset = TensorDataset(cfg.val_txt, cfg.input_width, cfg.input_height, False, cfg.classes)

    # Validation set
    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.batch_size,
                                                 shuffle=False,
                                                 collate_fn=collate_fn,
                                                 num_workers=4,
                                                 drop_last=False,
                                                 persistent_workers=True
                                                 )

    # Model evaluation
    print("compute mAP...")
    evaluation.compute_map(val_dataloader, model, multi_label=opt.multi_label_robust_mode)
