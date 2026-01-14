import argparse
import re
import tempfile

import numpy as np
import onnx
import torch
from torch.utils.data import DataLoader
from esp_ppq.api import espdl_quantize_onnx
from esp_ppq.api.setting import QuantizationSettingFactory
from esp_ppq.core import TargetPlatform

from utils.datasets import TensorDataset

DEFAULT_ONNX_MODEL_PATH = "fastestdetnext_x3_00_x1_00_96x96_opencv_inter_nearest_cls09.onnx"
DEFAULT_ESPDL_MODEL_PATH = "fastestdetnext_x3_00_x1_00_96x96_opencv_inter_nearest_cls09.espdl"
DEFAULT_TARGET = "esp32s3"
DEFAULT_NUM_OF_BITS = 8
DEFAULT_DEVICE = "cpu"

DEVICE = DEFAULT_DEVICE
INPUT_IS_NORMALIZED = False


def sanitize_onnx_model(onnx_model_path, batch_size=None, expand_group_conv=False):
    model = onnx.load(onnx_model_path)
    init_by_name = {init.name: init for init in model.graph.initializer}
    group_conv_updates = []
    if expand_group_conv:
        for node in model.graph.node:
            if node.op_type != "Conv":
                continue
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            groups = int(attrs.get("group", 1) or 1)
            if groups <= 1:
                continue
            if len(node.input) < 2:
                continue
            weight_name = node.input[1]
            weight_init = init_by_name.get(weight_name)
            if weight_init is None:
                continue
            weight = onnx.numpy_helper.to_array(weight_init)
            if weight.ndim != 4:
                continue
            in_per_group = weight.shape[1]
            in_channels = in_per_group * groups
            out_channels = weight.shape[0]
            if out_channels % groups != 0:
                continue
            out_per_group = out_channels // groups
            new_weight = np.zeros((out_channels, in_channels, weight.shape[2], weight.shape[3]), dtype=weight.dtype)
            for group_idx in range(groups):
                out_start = group_idx * out_per_group
                in_start = group_idx * in_per_group
                new_weight[
                    out_start : out_start + out_per_group,
                    in_start : in_start + in_per_group,
                    :,
                    :,
                ] = weight[out_start : out_start + out_per_group, :, :, :]
            weight_init.CopyFrom(onnx.numpy_helper.from_array(new_weight, weight_name))
            kept_attrs = [attr for attr in node.attribute if attr.name != "group"]
            del node.attribute[:]
            node.attribute.extend(kept_attrs)
            node.attribute.extend([onnx.helper.make_attribute("group", 1)])
            group_conv_updates.append(node.name or weight_name)
    resize_updates = []
    if batch_size is not None:
        for node in model.graph.node:
            if node.op_type != "Resize":
                continue
            if len(node.input) < 4:
                continue
            size_name = node.input[3]
            init = init_by_name.get(size_name)
            if init is None:
                continue
            arr = onnx.numpy_helper.to_array(init)
            if arr.ndim != 1 or arr.size != 4:
                continue
            old_n = int(arr[0])
            if old_n > 0 and old_n != batch_size:
                arr = arr.copy()
                arr[0] = batch_size
                init.CopyFrom(onnx.numpy_helper.from_array(arr, init.name))
                resize_updates.append(f"{size_name}: {old_n} -> {batch_size}")
    node_inputs = {name for node in model.graph.node for name in node.input}
    node_outputs = {name for node in model.graph.node for name in node.output}
    init_names = {init.name for init in model.graph.initializer}

    kept_outputs = [out for out in model.graph.output if out.name in node_outputs]
    kept_inputs = [
        inp for inp in model.graph.input if inp.name in node_inputs or inp.name in init_names
    ]

    removed_outputs = [out.name for out in model.graph.output if out.name not in node_outputs]
    removed_inputs = [
        inp.name
        for inp in model.graph.input
        if inp.name not in node_inputs and inp.name not in init_names
    ]

    if not removed_outputs and not removed_inputs and not resize_updates and not group_conv_updates:
        return onnx_model_path

    if removed_outputs:
        del model.graph.output[:]
        model.graph.output.extend(kept_outputs)
    if removed_inputs:
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        onnx.save(model, tmp.name)
        sanitized_path = tmp.name

    if group_conv_updates:
        print(f"Sanitized ONNX: expanded group convs: {len(group_conv_updates)}")
    if resize_updates:
        print("Sanitized ONNX: updated Resize sizes:", ", ".join(resize_updates))
    if removed_outputs:
        print(f"Sanitized ONNX: removed unlinked outputs: {removed_outputs}")
    if removed_inputs:
        print(f"Sanitized ONNX: removed unlinked inputs: {removed_inputs}")
    print(f"Sanitized ONNX saved to: {sanitized_path}")
    return sanitized_path


def get_int16_platform(target):
    target = target.lower()
    if target == "esp32s3":
        return TargetPlatform.ESPDL_S3_INT16
    if target == "esp32p4":
        return TargetPlatform.ESPDL_INT16
    if target == "c":
        return TargetPlatform.ESPDL_C_INT16
    return TargetPlatform.ESPDL_INT16


def build_dispatching_override(onnx_model_path, patterns, target):
    if not patterns:
        return None
    model = onnx.load(onnx_model_path)
    platform = get_int16_platform(target)
    override = {}
    matched = set()
    for node in model.graph.node:
        if not node.name:
            continue
        for pattern in patterns:
            if re.search(pattern, node.name):
                matched.add(node.name)
                break
    for name in matched:
        override[name] = platform
    if override:
        print(f"Forcing int16 for {len(override)} ops via pattern match.")
    else:
        print("No ops matched int16 patterns.")
    return override or None


def parse_class_ids(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        parts = value
    else:
        parts = str(value).split(",")
    return [int(part) for part in parts if str(part).strip()]


def _get_dim_value(dim):
    value = getattr(dim, "dim_value", None)
    if value is None or value <= 0:
        return None
    return int(value)


def get_onnx_input_size(onnx_model_path):
    try:
        model = onnx.load(onnx_model_path)
    except Exception:
        return None
    init_names = {init.name for init in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name in init_names:
            continue
        shape = inp.type.tensor_type.shape
        dims = [_get_dim_value(dim) for dim in shape.dim]
        if len(dims) < 4:
            continue
        dim1, dim2, dim3 = dims[1], dims[2], dims[3]
        if dim1 in (1, 3) and dim3 not in (1, 3):
            h, w = dim2, dim3
        elif dim3 in (1, 3) and dim1 not in (1, 3):
            h, w = dim1, dim2
        else:
            h, w = dim2, dim3
        if h and w:
            return w, h
    return None


def get_onnx_metadata_value(onnx_model_path, key):
    try:
        model = onnx.load(onnx_model_path)
    except Exception:
        return None
    for prop in model.metadata_props:
        if prop.key == key:
            value = str(prop.value).strip()
            return value or None
    return None


def collate_fn(batch):
    if not batch:
        return torch.empty(0, device=DEVICE)
    if isinstance(batch, (tuple, list)) and len(batch) == 2 and isinstance(batch[0], torch.Tensor):
        imgs = batch[0]
    elif isinstance(batch[0], (tuple, list)):
        images = [item[0] for item in batch]
        if isinstance(images[0], torch.Tensor):
            imgs = torch.stack(images, dim=0)
        else:
            imgs = torch.stack([torch.as_tensor(img) for img in images], dim=0)
    else:
        images = list(batch)
        if isinstance(images[0], torch.Tensor):
            imgs = torch.stack(images, dim=0)
        else:
            imgs = torch.stack([torch.as_tensor(img) for img in images], dim=0)
    imgs = imgs.to(DEVICE).float()
    if not INPUT_IS_NORMALIZED:
        imgs = imgs / 255.0
    return imgs


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Quantize an ONNX model for ESP-DL.")
    parser.add_argument(
        "--list-path",
        required=True,
        help="Path to a text file listing images to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Calibration batch size.",
    )
    parser.add_argument(
        "--calib-steps",
        type=int,
        default=32,
        help="Number of calibration steps.",
    )
    parser.add_argument(
        "--calib-algorithm",
        default="kl",
        help="Calibration algorithm (e.g., kl, minmax, mse, percentile).",
    )
    parser.add_argument(
        "--int16-op-pattern",
        action="append",
        default=[],
        help="Regex pattern to force matched ops to int16 (repeatable).",
    )
    parser.add_argument(
        "--onnx-model",
        default=DEFAULT_ONNX_MODEL_PATH,
        help="Path to the input ONNX model.",
    )
    parser.add_argument(
        "--espdl-model",
        default=DEFAULT_ESPDL_MODEL_PATH,
        help="Path to the output .espdl file.",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=["c", "esp32s3", "esp32p4"],
        help="Quantize target type.",
    )
    parser.add_argument(
        "--num-of-bits",
        type=int,
        default=DEFAULT_NUM_OF_BITS,
        help="Quantization bits.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=["cpu", "cuda"],
        help="Device for calibration.",
    )
    parser.add_argument(
        "--expand-group-conv",
        action="store_true",
        help="Expand group conv (groups > 1) into group=1.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    global DEVICE, INPUT_IS_NORMALIZED
    DEVICE = args.device

    list_path = args.list_path
    onnx_size = get_onnx_input_size(args.onnx_model)
    if onnx_size is None:
        raise ValueError("Input size missing in ONNX model; provide fixed input dims.")
    else:
        img_w, img_h = onnx_size
    onnx_model_path = args.onnx_model
    espdl_model_path = args.espdl_model
    target = args.target
    num_of_bits = args.num_of_bits
    onnx_model_path = sanitize_onnx_model(
        onnx_model_path,
        batch_size=args.batch_size,
        expand_group_conv=args.expand_group_conv,
    )
    metadata_classes = get_onnx_metadata_value(onnx_model_path, "classes")
    class_ids = parse_class_ids(metadata_classes)
    resize_mode = get_onnx_metadata_value(onnx_model_path, "resize-mode")
    dataset = TensorDataset(
        list_path,
        img_w,
        img_h,
        aug=False,
        class_ids=class_ids,
        resize_mode=resize_mode,
    )
    INPUT_IS_NORMALIZED = getattr(dataset, "input_is_normalized", False)
    # The dataloader shuffle setting must be set to False.
    # Because the dataset is traversed multiple times when calculating the quantization error,
    # if shuffle is set to True, an incorrect quantization error will be obtained.
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    sample_image, _ = dataset[0]
    input_shape = [1, *sample_image.shape]
    calib_steps = min(args.calib_steps, len(dataloader))

    setting = QuantizationSettingFactory.espdl_setting()
    if args.calib_algorithm:
        setting.quantize_activation_setting.calib_algorithm = args.calib_algorithm
    # Mixed precision quantization
    # https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_mobilenetv2.html#mixed-precision-quantization
    dispatching_override = build_dispatching_override(
        onnx_model_path,
        args.int16_op_pattern,
        target,
    )
    # Layerwise equalization quantization
    # https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_deploy_mobilenetv2.html#layerwise-equalization-quantization
    setting.equalization = True
    setting.equalization_setting.iterations = 4
    setting.equalization_setting.value_threshold = .4
    setting.equalization_setting.opt_level = 2
    setting.equalization_setting.interested_layers = None

    quant_ppq_graph = espdl_quantize_onnx(
        onnx_import_file=onnx_model_path,
        espdl_export_file=espdl_model_path,
        calib_dataloader=dataloader,
        calib_steps=calib_steps,  # Number of calibration steps
        input_shape=input_shape,  # Input shape, batch number 1
        inputs=None,
        target=target,  # Quantify target types
        num_of_bits=num_of_bits,  # Quantization bits
        setting=setting,
        collate_fn=collate_fn,
        dispatching_override=dispatching_override,
        device=DEVICE,
        error_report=True,
        skip_export=False,
        export_test_values=True,
        verbose=1,  # Output detailed log information
    )


if __name__ == "__main__":
    main()
