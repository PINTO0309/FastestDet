import os
import torch
import argparse
import warnings
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.resize import resize_output_channels
from utils.evaluation import CocoDetectionEvaluator

from module.detector import Detector, normalize_pyramid_levels

# Suppress noisy future warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)

# Select backend device: CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _parse_img_size(value, *, strict=False):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            height = int(value[0])
            width = int(value[1])
        except (TypeError, ValueError):
            if strict:
                raise ValueError("img-size must be in HxW format.")
            return None
        if height <= 0 or width <= 0:
            if strict:
                raise ValueError("img-size values must be positive.")
            return None
        return height, width
    if not isinstance(value, str):
        if strict:
            raise ValueError("img-size must be in HxW format.")
        return None
    parts = value.lower().split("x")
    if len(parts) != 2:
        if strict:
            raise ValueError("img-size must be in HxW format.")
        return None
    try:
        height = int(parts[0])
        width = int(parts[1])
    except ValueError:
        if strict:
            raise ValueError("img-size must be in HxW format.")
        return None
    if height <= 0 or width <= 0:
        if strict:
            raise ValueError("img-size values must be positive.")
        return None
    return height, width


def _load_weight_state(weight_path):
    try:
        weight_data = torch.load(weight_path, map_location=device, weights_only=False)
    except TypeError:
        weight_data = torch.load(weight_path, map_location=device)
    weight_state = weight_data
    if isinstance(weight_data, dict):
        if "model" in weight_data:
            weight_state = weight_data["model"]
        elif "state_dict" in weight_data:
            weight_state = weight_data["state_dict"]
    return weight_state, weight_data


def _get_meta_yaml(meta):
    meta_yaml = meta.get("yaml")
    if isinstance(meta_yaml, dict):
        return meta_yaml
    return None


def _get_yaml_input_size(meta_yaml):
    if not meta_yaml:
        return None
    model_cfg = meta_yaml.get("MODEL")
    if not isinstance(model_cfg, dict):
        return None
    height = model_cfg.get("INPUT_HEIGHT")
    width = model_cfg.get("INPUT_WIDTH")
    if height is None or width is None:
        return None
    return int(height), int(width)


def _infer_eval_settings(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a full training checkpoint saved by train.py.")
    meta = checkpoint.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    derived = meta.get("derived") or {}
    if not isinstance(derived, dict):
        derived = {}
    cli = meta.get("cli") or {}
    if not isinstance(cli, dict):
        cli = {}

    stage_out_channels = _first_not_none(
        checkpoint.get("stage_out_channels"),
        derived.get("stage_out_channels"),
    )
    stage_repeats = _first_not_none(
        checkpoint.get("stage_repeats"),
        derived.get("stage_repeats"),
    )
    pyramid_levels = _first_not_none(
        checkpoint.get("pyramid_levels"),
        derived.get("pyramid_levels"),
    )
    use_skip_residual = _first_not_none(
        checkpoint.get("use_skip_residual"),
        cli.get("use_skip_residual"),
    )
    use_se = _first_not_none(
        checkpoint.get("use_se"),
        cli.get("use_se"),
    )
    use_ese = _first_not_none(
        checkpoint.get("use_ese"),
        cli.get("use_ese"),
    )
    spp_separate_1x1 = _first_not_none(
        checkpoint.get("spp_separate_1x1"),
        derived.get("spp_separate_1x1"),
        cli.get("spp_separate_1x1"),
    )
    p1_stride = _first_not_none(
        checkpoint.get("p1_stride"),
        derived.get("p1_stride"),
    )
    resize_mode = _first_not_none(
        checkpoint.get("resize_mode"),
        derived.get("resize_mode"),
        cli.get("resize_mode"),
    )
    input_channels = _first_not_none(
        checkpoint.get("input_channels"),
        derived.get("input_channels"),
    )
    multi_label = cli.get("multi_label_robust_mode")

    if input_channels is None and resize_mode is not None:
        input_channels = resize_output_channels(resize_mode)
    if p1_stride is None:
        p1_stride = 8

    missing = []
    if stage_out_channels is None:
        missing.append("stage_out_channels")
    if stage_repeats is None:
        missing.append("stage_repeats")
    if pyramid_levels is None:
        missing.append("pyramid_levels")
    if use_skip_residual is None:
        missing.append("use_skip_residual")
    if use_se is None:
        missing.append("use_se")
    if use_ese is None:
        missing.append("use_ese")
    if input_channels is None:
        missing.append("input_channels/resize_mode")
    if multi_label is None:
        missing.append("meta.cli.multi_label_robust_mode")
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Checkpoint is missing metadata for: "
            f"{missing_text}. Please provide a full checkpoint from train.py."
        )

    if use_se and use_ese:
        raise ValueError("Checkpoint enables both SE and eSE.")
    if spp_separate_1x1 is None:
        spp_separate_1x1 = False

    return {
        "stage_out_channels": stage_out_channels,
        "stage_repeats": stage_repeats,
        "pyramid_levels": pyramid_levels,
        "use_skip_residual": bool(use_skip_residual),
        "use_se": bool(use_se),
        "use_ese": bool(use_ese),
        "spp_separate_1x1": bool(spp_separate_1x1),
        "multi_label": bool(multi_label),
        "resize_mode": resize_mode,
        "input_channels": int(input_channels),
        "p1_stride": int(p1_stride),
    }


def _infer_img_size(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    meta = checkpoint.get("meta") or {}
    if not isinstance(meta, dict):
        return None
    cli = meta.get("cli") or {}
    if isinstance(cli, dict):
        img_size = _parse_img_size(cli.get("img_size"))
        if img_size is not None:
            return img_size
    meta_yaml = _get_meta_yaml(meta)
    return _get_yaml_input_size(meta_yaml)

if __name__ == '__main__':
    # Training config
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None, help='.weight config')
    parser.add_argument('--img-size', type=str, default=None, help='override input size as HxW (height x width)')
    parser.add_argument('--batch-size', type=int, default=None, help='override batch size from yaml')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "Please provide a valid config file path."
    assert os.path.exists(opt.weight), "Please provide a valid weight file path."

    # Parse yaml config
    cfg = LoadYaml(opt.yaml)

    # Load model weights
    print("load weight from:%s"%opt.weight)
    weight_state, weight_data = _load_weight_state(opt.weight)
    cli_img_size = _parse_img_size(opt.img_size, strict=True) if opt.img_size is not None else None
    img_size = cli_img_size or _infer_img_size(weight_data)
    if img_size is not None:
        if (cfg.input_height, cfg.input_width) != img_size:
            source = "--img-size" if cli_img_size is not None else "checkpoint"
            print(f"override input size from {source}: {img_size[0]}x{img_size[1]}")
        cfg.input_height, cfg.input_width = img_size
    print(cfg)

    inferred = _infer_eval_settings(weight_data)
    stage_out_channels = inferred["stage_out_channels"]
    stage_repeats = inferred["stage_repeats"]
    pyramid_levels = normalize_pyramid_levels(inferred["pyramid_levels"])
    multi_label_robust_mode = inferred["multi_label"]
    resize_mode = inferred["resize_mode"]
    input_channels = inferred["input_channels"]
    p1_stride = inferred["p1_stride"]
    model = Detector(
        cfg.category_num,
        True,
        input_channels=input_channels,
        p1_stride=p1_stride,
        stage_repeats=stage_repeats,
        stage_out_channels=stage_out_channels,
        use_skip_residual=inferred["use_skip_residual"],
        use_ese=inferred["use_ese"],
        use_se=inferred["use_se"],
        multi_label=multi_label_robust_mode,
        pyramid_levels=pyramid_levels,
        spp_separate_1x1=inferred["spp_separate_1x1"],
    ).to(device)
    model.load_state_dict(weight_state)
    model.eval()

    # # Print tensor shapes of network layers
    summary(model, input_size=(input_channels, cfg.input_height, cfg.input_width))

    # Define evaluation
    evaluation = CocoDetectionEvaluator(cfg.names, device)

    # Dataset loading
    val_dataset = TensorDataset(
        cfg.val_txt,
        cfg.input_width,
        cfg.input_height,
        False,
        cfg.classes,
        resize_mode=resize_mode,
    )

    # Validation set
    batch_size = cfg.batch_size if opt.batch_size is None else int(opt.batch_size)
    if batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=False,
        persistent_workers=True,
    )

    # Model evaluation
    print("compute mAP...")
    evaluation.compute_map(val_dataloader, model, multi_label=multi_label_robust_mode)
