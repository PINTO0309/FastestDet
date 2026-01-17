import torch
import torch.distributed as dist
import numpy as np
import io
import sys
from contextlib import redirect_stdout, redirect_stderr, contextmanager
from tqdm import tqdm
from utils.tool import *

from faster_coco_eval import COCO, COCOeval_faster

class CocoDetectionEvaluator():
    def __init__(self, names, device):
        self.device = device
        self.classes = []
        self.last_per_class_ap = None
        with open(names, 'r') as f:
            for line in f.readlines():
                self.classes.append(line.strip())

    @contextmanager
    def _console_only(self):
        stdout = sys.stdout
        stderr = sys.stderr
        real_stdout = getattr(stdout, "primary", stdout)
        real_stderr = getattr(stderr, "primary", stderr)
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        try:
            yield
        finally:
            sys.stdout = stdout
            sys.stderr = stderr

    def coco_evaluate(self, gts, preds):
        # Create Ground Truth
        coco_gt = COCO()
        coco_gt.dataset = {}
        coco_gt.dataset["images"] = []
        coco_gt.dataset["annotations"] = []
        k = 0
        for i, gt in enumerate(gts):
            for j in range(gt.shape[0]):
                k += 1
                coco_gt.dataset["images"].append({"id": i})
                coco_gt.dataset["annotations"].append({"image_id": i, "category_id": gt[j, 0],
                                                    "bbox": np.hstack([gt[j, 1:3], gt[j, 3:5] - gt[j, 1:3]]),
                                                    "area": np.prod(gt[j, 3:5] - gt[j, 1:3]),
                                                    "id": k, "iscrowd": 0})

        coco_gt.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            coco_gt.createIndex()

        # Create preadict
        coco_pred = COCO()
        coco_pred.dataset = {}
        coco_pred.dataset["images"] = []
        coco_pred.dataset["annotations"] = []
        k = 0
        for i, pred in enumerate(preds):
            for j in range(pred.shape[0]):
                k += 1
                coco_pred.dataset["images"].append({"id": i})
                coco_pred.dataset["annotations"].append({"image_id": i, "category_id": int(pred[j, 0]),
                                                        "score": pred[j, 1], "bbox": np.hstack([pred[j, 2:4], pred[j, 4:6] - pred[j, 2:4]]),
                                                        "area": np.prod(pred[j, 4:6] - pred[j, 2:4]),
                                                        "id": k})

        coco_pred.dataset["categories"] = [{"id": i, "supercategory": c, "name": c} for i, c in enumerate(self.classes)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            coco_pred.createIndex()
            coco_eval = COCOeval_faster(coco_gt, coco_pred, "bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
        with self._console_only():
            coco_eval.summarize()
        mAP05 = coco_eval.stats[1]
        self._print_per_class_ap(coco_eval)
        map_text = "nan" if np.isnan(mAP05) else f"{mAP05:.5f}"
        print(f"mAP@0.5: {map_text}")
        return mAP05

    def _print_per_class_ap(self, coco_eval):
        precisions = coco_eval.eval.get("precision")
        if precisions is None:
            return
        iou_thrs = coco_eval.params.iouThrs
        iou_index = np.where(np.isclose(iou_thrs, 0.5))[0]
        if iou_index.size == 0:
            iou_index = np.array([0])
        iou_index = int(iou_index[0])
        area_labels = getattr(coco_eval.params, "areaRngLbl", ["all"])
        area_index = area_labels.index("all") if "all" in area_labels else 0
        maxdet_index = len(coco_eval.params.maxDets) - 1

        title = "Per-class AP@0.5"
        name_width = max([len(name) for name in self.classes] + [5])
        ap_width = 7
        top = "+" + "-" * (name_width + 2) + "+" + "-" * (ap_width + 2) + "+"
        header = f"| {'Class'.ljust(name_width)} | {'AP'.rjust(ap_width)} |"
        print(title)
        print(top)
        print(header)
        print(top)
        results = []
        name: str
        for k, name in enumerate(self.classes):
            precision = precisions[iou_index, :, k, area_index, maxdet_index]
            precision = precision[precision > -1]
            ap = float(np.mean(precision)) if precision.size else float("nan")
            ap_text = "nan" if np.isnan(ap) else f"{ap:.5f}"
            results.append((name, ap))
            print(f"| {name.ljust(name_width)} | {ap_text.rjust(ap_width)} |")
        print(top)
        self.last_per_class_ap = results

    def _collect_gts_pts(self, val_dataloader, model, multi_label=False):
        gts, pts = [], []
        input_is_normalized = getattr(val_dataloader.dataset, "input_is_normalized", False)
        disable_pbar = dist.is_available() and dist.is_initialized() and dist.get_rank() != 0
        pbar = tqdm(val_dataloader, disable=disable_pbar)
        imgs: torch.Tensor
        targets: torch.Tensor
        for _, (imgs, targets) in enumerate(pbar):
            # Data preprocessing
            imgs = imgs.to(self.device).float()
            if not input_is_normalized:
                imgs = imgs / 255.0
            with torch.no_grad():
                # Model prediction
                preds = model(imgs)
                # Feature map post-processing
                output = handle_preds(preds, self.device, 0.001, multi_label=multi_label)

            # Detection results
            N, _, H, W = imgs.shape
            scale = np.array([W, H, W, H], dtype=np.float32)
            p: torch.Tensor
            for p in output:
                if p.numel() == 0:
                    pts.append(np.zeros((0, 6), dtype=np.float32))
                    continue
                p_np = p.cpu().numpy()
                coords = p_np[:, :4] * scale
                pbboxes = np.concatenate((p_np[:, 5:6], p_np[:, 4:5], coords), axis=1)
                pts.append(pbboxes)

            # Ground truth
            t_np = targets.cpu().numpy()
            for n in range(N):
                tn = t_np[t_np[:, 0] == n]
                if tn.size == 0:
                    gts.append(np.zeros((0, 5), dtype=np.float32))
                    continue
                bcx = tn[:, 2] * W
                bcy = tn[:, 3] * H
                bw = tn[:, 4] * W
                bh = tn[:, 5] * H
                x1 = bcx - 0.5 * bw
                y1 = bcy - 0.5 * bh
                x2 = bcx + 0.5 * bw
                y2 = bcy + 0.5 * bh
                tbboxes = np.stack((tn[:, 1], x1, y1, x2, y2), axis=1)
                gts.append(tbboxes)
        return gts, pts

    def compute_map(self, val_dataloader, model, multi_label=False):
        gts, pts = self._collect_gts_pts(val_dataloader, model, multi_label=multi_label)
        mAP05 = self.coco_evaluate(gts, pts)
        return mAP05

    def compute_map_distributed(self, val_dataloader, model, multi_label=False):
        gts, pts = self._collect_gts_pts(val_dataloader, model, multi_label=multi_label)
        if not dist.is_available() or not dist.is_initialized():
            return self.coco_evaluate(gts, pts)
        world_size = dist.get_world_size()
        gathered_gts = [None for _ in range(world_size)]
        gathered_pts = [None for _ in range(world_size)]
        dist.all_gather_object(gathered_gts, gts)
        dist.all_gather_object(gathered_pts, pts)
        if dist.get_rank() != 0:
            return None
        flat_gts = [item for sublist in gathered_gts for item in sublist]
        flat_pts = [item for sublist in gathered_pts for item in sublist]
        return self.coco_evaluate(flat_gts, flat_pts)
