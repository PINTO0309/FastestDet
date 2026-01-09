import os
import cv2
import numpy as np
import yaml

import torch
import random

from utils.augment import build_augmentation_pipeline
from utils.resize import (
    resize_image_numpy,
    normalize_resize_mode,
    is_yuv422_mode,
    is_y_only_mode,
    is_y_bin_mode,
    is_y_tri_mode,
)

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class TensorDataset():
    def __init__(self, path, img_width, img_height, aug=False, class_ids=None, resize_mode=None, aug_yaml=None):
        assert os.path.exists(path), "%s path is invalid or missing" % path

        self.aug = aug
        self.path = path
        self.data_list = []
        self.img_width = img_width
        self.img_height = img_height
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
        self.class_map = None
        self.resize_mode = normalize_resize_mode(resize_mode) if resize_mode is not None else None
        self.input_is_normalized = self.resize_mode is not None
        self.aug_yaml = aug_yaml
        self.aug_pipeline = None
        if class_ids is not None:
            class_ids = [int(c) for c in class_ids]
            self.class_map = {cid: idx for idx, cid in enumerate(class_ids)}

        # Dataset validation
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        self.data_list.append(data_path)
                else:
                    raise Exception("%s is not exist" % data_path)
        self._init_augmentation()

    def _init_augmentation(self):
        if not self.aug or not self.aug_yaml:
            return
        if not os.path.exists(self.aug_yaml):
            raise Exception("%s is not exist" % self.aug_yaml)
        with open(self.aug_yaml, 'r', encoding='utf-8') as f:
            aug_data = yaml.safe_load(f) or {}
        aug_cfg = aug_data.get("data_augment", aug_data) or {}
        class_swap_map = None
        if "HorizontalFlip" in aug_cfg and isinstance(aug_cfg["HorizontalFlip"], dict):
            class_swap_map = aug_cfg["HorizontalFlip"].get("class_swap_map")
        class_swap_map = self._normalize_class_swap_map(class_swap_map)
        self.aug_pipeline = build_augmentation_pipeline(
            aug_cfg,
            img_w=self.img_width,
            img_h=self.img_height,
            class_swap_map=class_swap_map,
            dataset=self,
        )

    def _normalize_class_swap_map(self, class_swap_map):
        if not class_swap_map:
            return None
        mapped = {}
        for key, value in class_swap_map.items():
            try:
                key = int(key)
                value = int(value)
            except (TypeError, ValueError):
                continue
            if self.class_map is not None:
                if key not in self.class_map or value not in self.class_map:
                    continue
                mapped[self.class_map[key]] = self.class_map[value]
            else:
                mapped[key] = value
        return mapped or None

    def _load_labels(self, label_path):
        if not os.path.exists(label_path):
            raise Exception("%s is not exist" % label_path)
        boxes = []
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                l = line.strip().split()
                if len(l) < 5:
                    continue
                class_id = int(float(l[0]))
                if self.class_map is not None:
                    if class_id not in self.class_map:
                        continue
                    class_id = self.class_map[class_id]
                labels.append(class_id)
                boxes.append([float(l[1]), float(l[2]), float(l[3]), float(l[4])])
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64)
        return boxes, labels

    def _build_label_array(self, boxes, labels):
        if labels.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        label = np.zeros((labels.shape[0], 6), dtype=np.float32)
        label[:, 1] = labels.astype(np.float32)
        label[:, 2:] = boxes.astype(np.float32)
        return label

    def _resize_for_output(self, img, img_is_rgb):
        if self.resize_mode is None:
            img = cv2.resize(img, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
            if img_is_rgb:
                img = np.round(img[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
            return img

        if img_is_rgb:
            if (
                is_yuv422_mode(self.resize_mode)
                or is_y_only_mode(self.resize_mode)
                or is_y_bin_mode(self.resize_mode)
                or is_y_tri_mode(self.resize_mode)
            ):
                img_float = img
            else:
                img_float = img[..., ::-1]
        else:
            img_float = img.astype(np.float32) / 255.0
            if (
                is_yuv422_mode(self.resize_mode)
                or is_y_only_mode(self.resize_mode)
                or is_y_bin_mode(self.resize_mode)
                or is_y_tri_mode(self.resize_mode)
            ):
                img_float = cv2.cvtColor(img_float, cv2.COLOR_BGR2RGB)
        return resize_image_numpy(img_float, (self.img_height, self.img_width), self.resize_mode)

    def _format_after_augmentation(self, img_rgb):
        if self.resize_mode is None:
            return np.round(img_rgb[..., ::-1] * 255.0).clip(0, 255).astype(np.uint8)
        if (
            is_yuv422_mode(self.resize_mode)
            or is_y_only_mode(self.resize_mode)
            or is_y_bin_mode(self.resize_mode)
            or is_y_tri_mode(self.resize_mode)
        ):
            return resize_image_numpy(img_rgb, (self.img_height, self.img_width), self.resize_mode)
        return img_rgb[..., ::-1]

    def resize_image(self, img, size):
        out_w, out_h = size
        if self.resize_mode is None:
            return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        mode = self.resize_mode
        if (
            is_yuv422_mode(mode)
            or is_y_only_mode(mode)
            or is_y_bin_mode(mode)
            or is_y_tri_mode(mode)
        ):
            mode = "opencv_inter_nearest"
        return resize_image_numpy(img, (out_h, out_w), mode)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        label_path = img_path.split(".")[0] + ".txt"

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            raise Exception("%s is not exist" % img_path)
        boxes, labels = self._load_labels(label_path)

        if self.aug_pipeline is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_rgb = self.resize_image(img_rgb, (self.img_width, self.img_height))
            img_rgb, boxes, labels = self.aug_pipeline(img_rgb, boxes, labels)
            img = self._format_after_augmentation(img_rgb)
        else:
            img = self._resize_for_output(img, img_is_rgb=False)

        label = self._build_label_array(boxes, labels)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        label = np.ascontiguousarray(label)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)

    def sample_random(self):
        index = random.randint(0, len(self.data_list) - 1)
        img_path = self.data_list[index]
        label_path = img_path.split(".")[0] + ".txt"
        img = cv2.imread(img_path)
        if img is None:
            raise Exception("%s is not exist" % img_path)
        boxes, labels = self._load_labels(label_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img, boxes, labels

if __name__ == "__main__":
    data = TensorDataset("/home/xuehao/Desktop/TMP/pytorch-yolo/widerface/train.txt")
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)
