import argparse
import os
import re

import torch
from utils.resize import resize_output_channels, input_name_for_resize_mode
from module.detector import Detector, normalize_pyramid_levels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _first_not_none(*values):
    for value in values:
        if value is not None:
            return value
    return None


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


def _get_yaml_category_num(meta_yaml):
    if not meta_yaml:
        return None
    model_cfg = meta_yaml.get("MODEL")
    if isinstance(model_cfg, dict) and model_cfg.get("NC") is not None:
        return int(model_cfg.get("NC"))
    train_cfg = meta_yaml.get("TRAIN")
    if isinstance(train_cfg, dict) and train_cfg.get("CLASSES") is not None:
        classes = train_cfg.get("CLASSES")
        if isinstance(classes, str):
            items = [v.strip() for v in classes.split(",") if v.strip()]
            return len(items)
        if isinstance(classes, (list, tuple)):
            return len(classes)
    dataset_cfg = meta_yaml.get("DATASET")
    if isinstance(dataset_cfg, dict):
        names_path = dataset_cfg.get("NAMES")
        if isinstance(names_path, str) and os.path.exists(names_path):
            with open(names_path, encoding="utf8") as f:
                names = [line.strip() for line in f if line.strip()]
            if names:
                return len(names)
    return None


def _infer_model_settings(checkpoint):
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
    meta_yaml = _get_meta_yaml(meta)
    yaml_input_size = _get_yaml_input_size(meta_yaml)
    yaml_category_num = _get_yaml_category_num(meta_yaml)

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
    category_num = _first_not_none(
        checkpoint.get("category_num"),
        derived.get("category_num"),
        yaml_category_num,
    )
    yaml_input_size = _first_not_none(yaml_input_size, None)

    if input_channels is None and resize_mode is not None:
        input_channels = resize_output_channels(resize_mode)

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
    if category_num is None:
        missing.append("category_num")
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            "Checkpoint is missing metadata for: "
            f"{missing_text}. Please provide a full checkpoint from train.py."
        )

    if use_se and use_ese:
        raise ValueError("Checkpoint enables both SE and eSE.")

    return {
        "stage_out_channels": stage_out_channels,
        "stage_repeats": stage_repeats,
        "pyramid_levels": pyramid_levels,
        "use_skip_residual": bool(use_skip_residual),
        "use_se": bool(use_se),
        "use_ese": bool(use_ese),
        "multi_label": bool(multi_label),
        "resize_mode": resize_mode,
        "input_channels": int(input_channels),
        "cli_img_size": cli.get("img_size"),
        "yaml_input_size": yaml_input_size,
        "category_num": int(category_num),
    }


def _rename_dynamic_batch_dims(model, replacement="N"):
    import onnx

    pattern = re.compile(r"^unk__\d+$")

    def rename_value_info(value_info):
        if not value_info.type.HasField("tensor_type"):
            return 0
        shape = value_info.type.tensor_type.shape
        renamed = 0
        for dim in shape.dim:
            if dim.dim_param and pattern.match(dim.dim_param):
                dim.dim_param = replacement
                renamed += 1
        return renamed

    def iter_graphs(graph):
        yield graph
        for node in graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    yield from iter_graphs(attr.g)
                elif attr.type == onnx.AttributeProto.GRAPHS:
                    for g in attr.graphs:
                        yield from iter_graphs(g)

    renamed = 0
    for graph in iter_graphs(model.graph):
        for value_info in graph.input:
            renamed += rename_value_info(value_info)
        for value_info in graph.output:
            renamed += rename_value_info(value_info)
        for value_info in graph.value_info:
            renamed += rename_value_info(value_info)
    return renamed


def _set_onnx_metadata(onnx_path, key, value):
    import onnx

    model = onnx.load(onnx_path)
    for prop in model.metadata_props:
        if prop.key == key:
            prop.value = str(value)
            onnx.save(model, onnx_path)
            return
    entry = model.metadata_props.add()
    entry.key = key
    entry.value = str(value)
    onnx.save(model, onnx_path)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Export FastestDet ONNX from a checkpoint.")
    parser.add_argument("--weight", type=str, required=True, help="Checkpoint path (.pth).")
    parser.add_argument("--onnx-out", type=str, default=None, help="Output ONNX path (default: <weight_dir>/model.onnx).")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the exported model.")
    parser.add_argument("--img-size", type=str, default=None, help="Override input size as HxW (height x width).")
    parser.add_argument("--dynamic-batch", action="store_true", default=False, help="Export with dynamic N axis.")
    return parser


def main():
    args = build_arg_parser().parse_args()
    if not os.path.exists(args.weight):
        raise FileNotFoundError(f"Weight not found: {args.weight}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.opset <= 0:
        raise ValueError("--opset must be positive.")

    weight_state, checkpoint = _load_weight_state(args.weight)
    inferred = _infer_model_settings(checkpoint)

    if args.img_size:
        img_size = _parse_img_size(args.img_size)
    else:
        img_size = None
        if inferred["cli_img_size"]:
            img_size = _parse_img_size(inferred["cli_img_size"])
        if img_size is None and inferred["yaml_input_size"]:
            img_size = inferred["yaml_input_size"]
    if img_size is None:
        raise ValueError("Input size is missing. Use a checkpoint with meta.yaml or pass --img-size.")
    input_height, input_width = img_size

    stage_out_channels = inferred["stage_out_channels"]
    stage_repeats = inferred["stage_repeats"]
    pyramid_levels = normalize_pyramid_levels(inferred["pyramid_levels"])
    input_channels = inferred["input_channels"]

    model = Detector(
        inferred["category_num"],
        True,
        input_channels=input_channels,
        stage_repeats=stage_repeats,
        stage_out_channels=stage_out_channels,
        use_skip_residual=inferred["use_skip_residual"],
        use_ese=inferred["use_ese"],
        use_se=inferred["use_se"],
        multi_label=inferred["multi_label"],
        pyramid_levels=pyramid_levels,
    ).to(device)

    model.load_state_dict(weight_state)
    model.eval()

    onnx_out = args.onnx_out
    if onnx_out is None:
        onnx_out = os.path.join(os.path.dirname(os.path.abspath(args.weight)), "model.onnx")

    dummy = torch.zeros(
        args.batch_size,
        input_channels,
        input_height,
        input_width,
        device=device,
    )
    input_name = input_name_for_resize_mode(inferred.get("resize_mode"))
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            input_name: {0: "N"},
            "output": {0: "N"},
        }
    torch.onnx.export(
        model,
        dummy,
        onnx_out,
        export_params=True,
        opset_version=args.opset,
        input_names=[input_name],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    import onnx
    from onnxsim import simplify
    onnx_model = onnx.load(onnx_out)
    model_simp, check = simplify(onnx_model)
    if not check:
        raise RuntimeError("onnxsim simplification check failed.")
    if args.dynamic_batch:
        _rename_dynamic_batch_dims(model_simp, "N")
    onnx.save(model_simp, onnx_out)
    if inferred.get("resize_mode") is not None:
        _set_onnx_metadata(onnx_out, "resize-mode", inferred["resize_mode"])
    print(f"export onnx: {onnx_out}")
    print("onnx sim success...")


if __name__ == "__main__":
    main()
