import torch
import numpy as np
from tqdm import tqdm
from utils.tool import *

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

class CocoDetectionEvaluator():
    def __init__(self, names, device):
        self.device = device
        self.classes = []
        with open(names, 'r') as f:
            for line in f.readlines():
                self.classes.append(line.strip())
    
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
        coco_pred.createIndex()

        coco_eval = COCOeval(coco_gt, coco_pred, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        mAP05 = coco_eval.stats[1]
        self._print_per_class_ap(coco_eval)
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
        for k, name in enumerate(self.classes):
            precision = precisions[iou_index, :, k, area_index, maxdet_index]
            precision = precision[precision > -1]
            ap = float(np.mean(precision)) if precision.size else float("nan")
            ap_text = "nan" if np.isnan(ap) else f"{ap:.4f}"
            print(f"| {name.ljust(name_width)} | {ap_text.rjust(ap_width)} |")
        print(top)

    def compute_map(self, val_dataloader, model):
        gts, pts = [], []
        input_is_normalized = getattr(val_dataloader.dataset, "input_is_normalized", False)
        pbar = tqdm(val_dataloader)
        for i, (imgs, targets) in enumerate(pbar):
            # Data preprocessing
            imgs = imgs.to(self.device).float()
            if not input_is_normalized:
                imgs = imgs / 255.0
            with torch.no_grad():
                # Model prediction
                preds = model(imgs)
                # Feature map post-processing
                output = handle_preds(preds, self.device, 0.001)

            # Detection results
            N, _, H, W = imgs.shape
            for p in output:
                pbboxes = []
                for b in p:
                    b = b.cpu().numpy()
                    score = b[4]
                    category = b[5]
                    x1, y1, x2, y2 = b[:4] * [W, H, W, H]
                    pbboxes.append([category, score, x1, y1, x2, y2])
                pts.append(np.array(pbboxes))

            # Ground truth
            for n in range(N):
                tbboxes = []
                for t in targets:
                    if t[0] == n:
                        t = t.cpu().numpy()
                        category = t[1]
                        bcx, bcy, bw, bh = t[2:] * [W, H, W, H]
                        x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
                        x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh
                        tbboxes.append([category, x1, y1, x2, y2])
                gts.append(np.array(tbboxes))
                
        mAP05 = self.coco_evaluate(gts, pts)

        return mAP05
