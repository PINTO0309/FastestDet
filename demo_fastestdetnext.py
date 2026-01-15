import os
import argparse
import json
import threading
from pathlib import Path
import time
from typing import Tuple

def suppress_qt_warnings():
    patterns = (
        "QObject::moveToThread",
        "Cannot move to target thread",
        "Only C and default locale supported with the posix collation implementation",
        "Case insensitive sorting unsupported in the posix collation implementation",
        "Numeric mode unsupported in the posix collation implementation",
        "Ignoring punctuation unsupported in the posix collation implementation",
    )
    try:
        r_fd, w_fd = os.pipe()
    except OSError:
        return
    orig_fd = os.dup(2)
    os.dup2(w_fd, 2)
    os.close(w_fd)

    def _reader():
        with os.fdopen(r_fd, "r", encoding="utf8", errors="replace") as reader, \
            os.fdopen(orig_fd, "w", encoding="utf8", errors="replace", buffering=1) as writer:
            for line in reader:
                if not line.strip():
                    continue
                if any(pat in line for pat in patterns):
                    continue
                writer.write(line)

    threading.Thread(target=_reader, daemon=True).start()


suppress_qt_warnings()

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import yaml


def normalize_resize_mode(mode):
    if not mode:
        return None
    return str(mode).lower().replace("-", "_")


def get_resize_interpolation(mode):
    mode = normalize_resize_mode(mode)
    if mode in ("opencv_inter_nearest", "opencv_inter_nearest_y", "opencv_inter_nearest_y_bin",
                "opencv_inter_nearest_y_tri", "opencv_inter_nearest_yuv422", "torch_nearest"):
        return cv2.INTER_NEAREST
    if mode in ("opencv_inter_linear", "torch_bilinear"):
        return cv2.INTER_LINEAR
    if mode is None:
        return cv2.INTER_LINEAR
    print(f"[WARN] Unknown resize-mode '{mode}', fallback to cv2.INTER_LINEAR")
    return cv2.INTER_LINEAR


def is_y_only_mode(mode):
    return normalize_resize_mode(mode) == "opencv_inter_nearest_y"


def is_y_bin_mode(mode):
    return normalize_resize_mode(mode) == "opencv_inter_nearest_y_bin"


def is_y_tri_mode(mode):
    return normalize_resize_mode(mode) == "opencv_inter_nearest_y_tri"


def is_yuv422_mode(mode):
    return normalize_resize_mode(mode) == "opencv_inter_nearest_yuv422"


def rgb_to_y(img: np.ndarray):
    img_u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    yuv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV)
    return yuv[..., 0:1].astype(np.float32) / 255.0


def rgb_to_yuyv422(img: np.ndarray):
    h, w, _ = img.shape
    if w % 2 != 0:
        raise ValueError(f"YUYV422 requires even width; got {w}")
    img_u8 = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
    yuyv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2YUV_YUYV)
    return yuyv.astype(np.float32) / 255.0


def y_to_bin(img: np.ndarray, threshold: float=0.5):
    y = np.clip(img, 0.0, 1.0)
    return (y >= threshold).astype(np.float32)


def y_to_tri(img, t1=1.0 / 3.0, t2=2.0 / 3.0):
    y = np.clip(img, 0.0, 1.0)
    out = np.zeros_like(y, dtype=np.float32)
    out[(y >= t1) & (y < t2)] = 0.5
    out[y >= t2] = 1.0
    return out


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def parse_classes(value):
    if value is None:
        return None
    if isinstance(value, str):
        items = [v.strip() for v in value.split(",") if v.strip()]
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        items = [value]
    class_ids = []
    for item in items:
        if isinstance(item, str):
            if not item:
                continue
            class_ids.append(int(float(item)))
        else:
            class_ids.append(int(item))
    return class_ids if class_ids else None


def _parse_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return None


def load_label_names(path):
    if not path or not os.path.exists(path):
        return None
    names = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.append(name)
    return names if names else None


def load_metadata(onnx_path):
    model = onnx.load(onnx_path)
    metadata = {p.key: p.value for p in model.metadata_props}
    resize_mode = metadata.get("resize-mode")
    model_classes = parse_classes(metadata.get("classes"))
    multi_label = None
    for key in ("meta.cli.multi_label_robust_mode", "multi_label_robust_mode", "multi_label"):
        if key in metadata:
            multi_label = _parse_bool(metadata.get(key))
            if multi_label is not None:
                break

    class_names = None
    remapped = metadata.get("remapped-class-ids")
    if remapped:
        try:
            mapping = json.loads(remapped)
            items = sorted(mapping.items(), key=lambda kv: int(kv[0]))
            class_names = [v for _, v in items]
        except (ValueError, json.JSONDecodeError):
            print("[WARN] Failed to parse remapped-class-ids metadata")

    return metadata, resize_mode, class_names, model_classes, multi_label


def map_class_id(cid, class_map, fallback_max):
    try:
        cid_int = int(cid)
    except (TypeError, ValueError):
        return None
    if class_map is not None and cid_int in class_map:
        return class_map[cid_int]
    if fallback_max is not None and 0 <= cid_int < fallback_max:
        return cid_int
    if class_map is None:
        return cid_int
    return None


def normalize_render_priority_rules(rules, class_map, fallback_max):
    if not rules:
        return []
    normalized = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        parents = parse_classes(rule.get("parent", rule.get("parents")))
        children = parse_classes(rule.get("children", rule.get("child")))
        if not parents or not children:
            continue
        mapped_parents = []
        for pid in parents:
            mapped = map_class_id(pid, class_map, fallback_max)
            if mapped is None:
                continue
            mapped_parents.append(mapped)
        mapped_children = []
        for cid in children:
            mapped = map_class_id(cid, class_map, fallback_max)
            if mapped is None:
                continue
            mapped_children.append(mapped)
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
        drawing_mode = rule.get("drawing_mode", "box")
        if isinstance(drawing_mode, str):
            drawing_mode = drawing_mode.strip().lower()
        if drawing_mode not in ("box", "arrow"):
            drawing_mode = "box"
        normalized.append({
            "parents": mapped_parents,
            "children": mapped_children,
            "iou": iou,
            "require_parent_match": require_parent_match,
            "use_parent_box": use_parent_box,
            "drawing_mode": drawing_mode,
        })
    return normalized


def collect_render_drawing_modes(rules):
    modes = {}
    for rule in rules:
        mode = rule.get("drawing_mode", "box")
        if mode != "arrow":
            continue
        for cid in rule.get("children", []):
            modes[int(cid)] = mode
    return modes


def normalize_render_label_ids(label_ids, class_map, fallback_max):
    if label_ids is None:
        return None
    if isinstance(label_ids, str) and not label_ids.strip():
        return set()
    ids = parse_classes(label_ids)
    if not ids:
        return set()
    normalized = set()
    for cid in ids:
        mapped = map_class_id(cid, class_map, fallback_max)
        if mapped is None:
            continue
        normalized.add(mapped)
    return normalized


def normalize_render_score_ids(score_ids, class_map, fallback_max):
    if score_ids is None:
        return set()
    if isinstance(score_ids, str) and not score_ids.strip():
        return set()
    ids = parse_classes(score_ids)
    if not ids:
        return set()
    normalized = set()
    for cid in ids:
        mapped = map_class_id(cid, class_map, fallback_max)
        if mapped is None:
            continue
        normalized.add(mapped)
    return normalized


def load_yaml_render_config(yaml_path, model_class_ids=None):
    if not yaml_path:
        return None
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return None
    render_cfg = data.get("RENDER")
    if not isinstance(render_cfg, dict):
        return None
    train_cfg = data.get("TRAIN") or {}
    classes = parse_classes(train_cfg.get("CLASSES"))
    mapping_source = None
    if model_class_ids:
        class_map = {int(cid): idx for idx, cid in enumerate(model_class_ids)}
        fallback_max = len(model_class_ids)
        mapping_source = "onnx.classes"
    elif classes:
        class_map = {int(cid): idx for idx, cid in enumerate(classes)}
        fallback_max = len(classes)
        mapping_source = "yaml.TRAIN.CLASSES"
    else:
        class_map = None
        fallback_max = None
    rules = normalize_render_priority_rules(render_cfg.get("PRIORITY_RULES"), class_map, fallback_max)
    drawing_modes = collect_render_drawing_modes(rules)
    priority_children = set()
    priority_parents = set()
    for rule in rules:
        for cid in rule.get("parents", []):
            priority_parents.add(int(cid))
        for cid in rule.get("children", []):
            priority_children.add(int(cid))
    label_ids = normalize_render_label_ids(render_cfg.get("LABEL_IDS"), class_map, fallback_max)
    score_ids = None
    if "SCORE_IDS" in render_cfg:
        score_ids = normalize_render_score_ids(render_cfg.get("SCORE_IDS"), class_map, fallback_max)
    names_path = (data.get("DATASET") or {}).get("NAMES")
    label_names = load_label_names(names_path)
    return {
        "priority_rules": rules,
        "priority_children": priority_children,
        "priority_parents": priority_parents,
        "drawing_modes": drawing_modes,
        "label_ids": label_ids,
        "score_ids": score_ids,
        "label_names": label_names,
        "class_map": class_map,
        "mapping_source": mapping_source,
    }


def get_input_info(session, onnx_path):
    input_info = session.get_inputs()[0]
    shape = list(input_info.shape)
    if any(dim is None for dim in shape):
        model = onnx.load(onnx_path)
        graph_input = model.graph.input[0]
        shape = []
        for dim in graph_input.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else None)
    if len(shape) != 4 or shape[2] is None or shape[3] is None:
        raise ValueError("Unable to infer input shape from ONNX model")
    return input_info.name, shape


def prepare_input(img_bgr, input_size, input_channels, resize_mode):
    input_tensor: np.ndarray
    img_yuv: np.ndarray
    img_y: np.ndarray
    resized: np.ndarray

    input_w, input_h = input_size
    interp = get_resize_interpolation(resize_mode)
    resized = cv2.resize(img_bgr, (input_w, input_h), interpolation=interp)

    if input_channels == 3:
        img_float = resized.astype(np.float32) / 255.0
        input_tensor = img_float.transpose(2, 0, 1)
    else:
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if input_channels == 1:
            img_y = rgb_to_y(img_rgb)
            if is_y_bin_mode(resize_mode):
                img_y = y_to_bin(img_y)
            elif is_y_tri_mode(resize_mode):
                img_y = y_to_tri(img_y)
            input_tensor = img_y.transpose(2, 0, 1)
        elif input_channels == 2:
            img_yuv = rgb_to_yuyv422(img_rgb)
            input_tensor = img_yuv.transpose(2, 0, 1)
        else:
            raise ValueError(f"Unsupported input channels: {input_channels}")

    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    return input_tensor, resized


def decode_predictions(feature_map: np.ndarray, score_thresh: float, multi_label: bool = False):
    cls_scores: np.ndarray
    cls_idx: np.ndarray
    cls_max: np.ndarray

    if feature_map.ndim != 3:
        raise ValueError(f"Expected CHW feature map, got shape {feature_map.shape}")

    c, h, w = feature_map.shape
    feature = feature_map.transpose(1, 2, 0)

    obj = feature[..., 0]
    cls_scores = feature[..., 5:]
    if multi_label:
        scores = np.power(obj[..., None], 0.6) * np.power(cls_scores, 0.4)
        keep = scores > score_thresh
    else:
        cls_idx = np.argmax(cls_scores, axis=-1)
        cls_max = np.max(cls_scores, axis=-1)
        scores = np.power(obj, 0.6) * np.power(cls_max, 0.4)
        keep = scores > score_thresh
    if not np.any(keep):
        return np.zeros((0, 6), dtype=np.float32)

    x_offset = np.tanh(feature[..., 1])
    y_offset = np.tanh(feature[..., 2])
    box_w = sigmoid(feature[..., 3])
    box_h = sigmoid(feature[..., 4])

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    bcx = (x_offset + grid_x) / w
    bcy = (y_offset + grid_y) / h

    x1 = bcx - 0.5 * box_w
    y1 = bcy - 0.5 * box_h
    x2 = bcx + 0.5 * box_w
    y2 = bcy + 0.5 * box_h

    if multi_label:
        ys, xs, cs = np.where(keep)
        scores_sel = scores[ys, xs, cs]
        dets = np.stack(
            (
                x1[ys, xs],
                y1[ys, xs],
                x2[ys, xs],
                y2[ys, xs],
                scores_sel,
                cs.astype(np.float32),
            ),
            axis=-1,
        )
        return dets
    dets = np.stack((x1, y1, x2, y2, scores, cls_idx.astype(np.float32)), axis=-1)
    dets = dets[keep]
    return dets


def scale_detections(dets: np.ndarray, output_size: Tuple[int, int], clip=True):
    if dets.size == 0:
        return dets
    scaled = dets.copy()
    if clip:
        scaled[:, 0:4] = np.clip(scaled[:, 0:4], 0.0, 1.0)
    out_w, out_h = output_size
    scaled[:, 0] *= out_w
    scaled[:, 2] *= out_w
    scaled[:, 1] *= out_h
    scaled[:, 3] *= out_h
    return scaled


def nms_single_class(dets: np.ndarray, iou_thresh: float):
    x1: np.ndarray
    y1: np.ndarray
    x2: np.ndarray
    y2: np.ndarray
    scores: np.ndarray

    if dets.size == 0:
        return []
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


def nms_by_class(dets: np.ndarray, iou_thresh: float):
    if dets.size == 0:
        return dets
    output = []
    classes = np.unique(dets[:, 5]).astype(np.int64)
    for cls_id in classes:
        cls_mask = dets[:, 5] == cls_id
        dets_cls = dets[cls_mask]
        keep = nms_single_class(dets_cls, iou_thresh)
        if keep:
            output.append(dets_cls[keep])
    if not output:
        return np.zeros((0, 6), dtype=np.float32)
    return np.concatenate(output, axis=0)


def filter_detections_by_thresholds(dets, score_thresh, attr_score_thresh, attr_class_ids):
    if dets.size == 0:
        return dets
    scores = dets[:, 4]
    if not attr_class_ids:
        return dets[scores >= score_thresh]
    cls_ids = dets[:, 5].astype(np.int64)
    attr_mask = np.isin(cls_ids, list(attr_class_ids))
    keep = np.where(attr_mask, scores >= attr_score_thresh, scores >= score_thresh)
    return dets[keep]


def color_for_class(class_id):
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


def arrow_direction_for_label(label: str):
    if not label:
        return None
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "front": (0.0, 1.0),
        "right_front": (-1.0, 1.0),
        "right_side": (-1.0, 0.0),
        "right": (-1.0, 0.0),
        "right_back": (-1.0, -1.0),
        "back": (0.0, -1.0),
        "left_back": (1.0, -1.0),
        "left_side": (1.0, 0.0),
        "left": (1.0, 0.0),
        "left_front": (1.0, 1.0),
    }
    return mapping.get(key)


def draw_orientation_arrow(img: np.ndarray, center, length, direction, color=(0, 0, 255)):
    dx, dy = direction
    norm = np.hypot(dx, dy)
    if norm <= 0:
        return
    dx /= norm
    dy /= norm
    length = max(6, int(length))
    start = (int(round(center[0])), int(round(center[1])))
    end = (int(round(center[0] + dx * length)), int(round(center[1] + dy * length)))
    end = (
        max(0, min(img.shape[1] - 1, end[0])),
        max(0, min(img.shape[0] - 1, end[1])),
    )
    cv2.arrowedLine(img, start, end, color, 2, line_type=cv2.LINE_AA, tipLength=0.3)


def bbox_iou_one_to_many(box, boxes):
    px1, py1, px2, py2 = box
    cx1, cy1, cx2, cy2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    inter_w = np.maximum(0.0, np.minimum(px2, cx2) - np.maximum(px1, cx1))
    inter_h = np.maximum(0.0, np.minimum(py2, cy2) - np.maximum(py1, cy1))
    inter = inter_w * inter_h
    p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    c_area = np.maximum(0.0, cx2 - cx1) * np.maximum(0.0, cy2 - cy1)
    union = p_area + c_area - inter + 1e-9
    return inter / union


def apply_render_priority_rules(boxes, rules):
    if not rules or boxes.size == 0:
        return boxes
    keep = np.ones(len(boxes), dtype=bool)
    for rule in rules:
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
            [bbox_iou_one_to_many(child_box, parent_boxes) for child_box in child_boxes],
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


def draw_detections(img, dets, class_names):
    for det in dets:
        x1, y1, x2, y2, score, cls_id = det
        cls_id = int(cls_id)
        label = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1_i, y1_i), (x2_i, y2_i), (255, 255, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1_i, y1_i - 5), 0, 0.6, (0, 255, 0), 2)
        cv2.putText(img, label, (x1_i, y1_i - 22), 0, 0.6, (0, 255, 0), 2)


def draw_detections_with_render_config(img, dets, class_names, render_config):
    if dets.size == 0:
        return
    boxes = dets
    h, w = img.shape[:2]
    for det in boxes:
        x1, y1, x2, y2, score, cls_id = det.tolist()
        cls_id = int(cls_id)
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        label_allowed = True
        render_label_ids = render_config.get("label_ids")
        if render_label_ids is not None:
            label_allowed = cls_id in render_label_ids
        label_name = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        drawing_mode = render_config.get("drawing_modes", {}).get(cls_id, "box")
        color = color_for_class(cls_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        arrow_drawn = False
        if drawing_mode == "arrow" and label_allowed:
            direction = arrow_direction_for_label(label_name)
            if direction is not None:
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                length = 0.45 * max(1.0, min(x2 - x1, y2 - y1))
                draw_orientation_arrow(img, center, length, direction)
                arrow_drawn = True
        if not arrow_drawn:
            label = label_name if label_allowed else None
            if label:
                show_score = True
                render_score_ids = render_config.get("score_ids")
                if render_score_ids is not None:
                    show_score = cls_id in render_score_ids
                text = f"{label}:{score:.2f}" if show_score else label
                cv2.putText(img, text, (x1, max(0, y1 - 5)), 0, 0.6, (255, 255, 255), 2)
                cv2.putText(img, text, (x1, max(0, y1 - 5)), 0, 0.6, color, 1)


def draw_inference_time(img, elapsed_ms):
    text = f"{elapsed_ms:.2f} ms"
    cv2.putText(img, text, (10, 25), 0, 0.8, (0, 0, 255), 2)


def iter_images(path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Input path not found: {path}")
    files = [p for p in sorted(path.iterdir()) if p.suffix.lower() in exts]
    return files


def ensure_dir(path):
    if path is None:
        return None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_path(output_path: Path, input_path: Path, is_dir_input: bool):
    if output_path is None:
        return None
    output_path = Path(output_path)
    if is_dir_input:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / input_path.name
    if output_path.suffix:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / input_path.name


def parse_camera_source(value):
    if value is None:
        return None
    value = str(value)
    if value.isdigit():
        return int(value)
    return value


def create_session(onnx_path, provider):
    available = ort.get_available_providers()
    if provider == "cuda":
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("[WARN] CUDAExecutionProvider not available, fallback to CPU")
            providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, providers=providers)


def run_on_images(args, session, input_name, input_shape, resize_mode, class_names, render_config):
    paths = iter_images(args.input)
    is_dir_input: bool = Path(args.input).is_dir()
    attr_class_ids = render_config.get("priority_children") if render_config else set()
    base_thresh = args.score_threshold
    if attr_class_ids:
        base_thresh = min(args.score_threshold, args.attr_score_threshold)
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[WARN] Failed to read image: {path}")
            continue
        input_tensor, resized = prepare_input(
            img,
            (input_shape[3], input_shape[2]),
            input_shape[1],
            resize_mode,
        )
        start = time.perf_counter()
        output = session.run(None, {input_name: input_tensor})[0][0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        display_img = resized.copy() if args.actual_size else img.copy()
        output_size = (display_img.shape[1], display_img.shape[0])
        dets = decode_predictions(output, base_thresh, multi_label=args.multi_label_robust_mode)
        dets = filter_detections_by_thresholds(
            dets,
            args.score_threshold,
            args.attr_score_threshold,
            attr_class_ids,
        )
        dets = nms_by_class(dets, args.iou_threshold)
        if render_config:
            dets = apply_render_priority_rules(dets, render_config.get("priority_rules"))
        dets = scale_detections(dets, output_size, clip=True)
        if render_config:
            draw_detections_with_render_config(display_img, dets, class_names, render_config)
        else:
            draw_detections(display_img, dets, class_names)
        if not args.actual_size:
            draw_inference_time(display_img, elapsed_ms)

        out_path = resolve_output_path(args.output, path, is_dir_input)
        if out_path is not None:
            cv2.imwrite(str(out_path), display_img)

        if not args.no_display:
            cv2.imshow("FastestDetNext", display_img)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:
                break


def run_on_camera(args, session, input_name, input_shape, resize_mode, class_names, render_config):
    source = parse_camera_source(args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source: {args.camera}")

    output_dir = ensure_dir(args.output)
    frame_idx = 0
    attr_class_ids = render_config.get("priority_children") if render_config else set()
    base_thresh = args.score_threshold
    if attr_class_ids:
        base_thresh = min(args.score_threshold, args.attr_score_threshold)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        input_tensor, resized = prepare_input(
            frame,
            (input_shape[3], input_shape[2]),
            input_shape[1],
            resize_mode,
        )
        start = time.perf_counter()
        output = session.run(None, {input_name: input_tensor})[0][0]
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        display_img = resized.copy() if args.actual_size else frame.copy()
        output_size = (display_img.shape[1], display_img.shape[0])
        dets = decode_predictions(output, base_thresh, multi_label=args.multi_label_robust_mode)
        dets = filter_detections_by_thresholds(
            dets,
            args.score_threshold,
            args.attr_score_threshold,
            attr_class_ids,
        )
        dets = nms_by_class(dets, args.iou_threshold)
        if render_config:
            dets = apply_render_priority_rules(dets, render_config.get("priority_rules"))
        dets = scale_detections(dets, output_size, clip=True)
        if render_config:
            draw_detections_with_render_config(display_img, dets, class_names, render_config)
        else:
            draw_detections(display_img, dets, class_names)
        if not args.actual_size:
            draw_inference_time(display_img, elapsed_ms)

        if output_dir is not None:
            out_path = output_dir / f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), display_img)
        frame_idx += 1

        if not args.no_display:
            cv2.imshow("FastestDetNext", display_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="FastestDetNext ONNX runtime demo")
    parser.add_argument(
        "--onnx",
        default="fastestdetnext_x3_00_x1_00_96x96_opencv_inter_nearest_cls09.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument("--input", help="Path to image or directory")
    parser.add_argument("--camera", help="Camera index or device path")
    parser.add_argument("--output", help="Output file or directory")
    parser.add_argument("--actual-size", action="store_true", help="Render output at model input size")
    parser.add_argument("--provider", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--score-threshold", "--socre-threshold", dest="score_threshold", type=float, default=0.70)
    parser.add_argument("--attr-score-threshold", dest="attr_score_threshold", type=float, default=0.01)
    parser.add_argument("--iou-threshold", dest="iou_threshold", type=float, default=0.35)
    parser.add_argument("--yaml", help="Path to yaml config (for render rules)")
    parser.add_argument("--no-display", action="store_true", help="Disable cv2.imshow output")

    args = parser.parse_args()

    if not args.input and not args.camera:
        raise ValueError("Please set --input or --camera")

    metadata, resize_mode, class_names, model_classes, meta_multi_label = load_metadata(args.onnx)
    print(f"resize-mode: {resize_mode}")
    print(f"remapped-class-ids: {metadata.get('remapped-class-ids')}")
    render_config = load_yaml_render_config(args.yaml, model_classes)
    if render_config and render_config.get("label_names"):
        class_names = render_config.get("label_names")
    if class_names is None:
        print("[WARN] remapped-class-ids metadata missing; labels will be class indices")
    if render_config:
        class_map = render_config.get("class_map")
        mapping_source = render_config.get("mapping_source")
        if class_map:
            print(f"render class mapping source: {mapping_source}")
            print(f"render class mapping: {class_map}")
        else:
            print("render class mapping: (none)")
    if args.score_threshold is None:
        args.score_threshold = 0.001 if render_config else 0.70
    if meta_multi_label is None:
        args.multi_label_robust_mode = False
        print("[WARN] ONNX metadata missing multi_label_robust_mode; defaulting to False.")
    else:
        args.multi_label_robust_mode = bool(meta_multi_label)

    session = create_session(args.onnx, args.provider)
    input_name, input_shape = get_input_info(session, args.onnx)

    if args.camera:
        run_on_camera(args, session, input_name, input_shape, resize_mode, class_names, render_config)
    else:
        run_on_images(args, session, input_name, input_shape, resize_mode, class_names, render_config)
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
