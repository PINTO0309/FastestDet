import yaml
import torch
import torchvision

# Parse class id parameters
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

# Parse yaml config
class LoadYaml:
    def __init__(self, path):
        with open(path, encoding='utf8') as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        self.val_txt = data["DATASET"]["VAL"]
        self.train_txt = data["DATASET"]["TRAIN"]
        self.names = data["DATASET"]["NAMES"]

        self.learn_rate = data["TRAIN"]["LR"]
        self.batch_size = data["TRAIN"]["BATCH_SIZE"]
        self.milestones = data["TRAIN"]["MILESTIONES"]
        self.end_epoch = data["TRAIN"]["END_EPOCH"]
        self.classes = parse_classes(data["TRAIN"].get("CLASSES"))
        
        self.input_width = data["MODEL"]["INPUT_WIDTH"]
        self.input_height = data["MODEL"]["INPUT_HEIGHT"]
        
        self.category_num = data["MODEL"]["NC"]
        
        print("Load yaml sucess...")

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# Post-processing (normalized coordinates)
def handle_preds(preds, device, conf_thresh=0.25, nms_thresh=0.45):
    total_bboxes, output_bboxes  = [], []
    # Convert feature map to bounding box coordinates
    N, C, H, W = preds.shape
    bboxes = torch.zeros((N, H, W, 6))
    pred = preds.permute(0, 2, 3, 1)
    # Objectness branch
    pobj = pred[:, :, :, 0].unsqueeze(dim=-1)
    # Box regression branch
    preg = pred[:, :, :, 1:5]
    # Class prediction branch
    pcls = pred[:, :, :, 5:]

    # Bounding box confidence
    bboxes[..., 4] = (pobj.squeeze(-1) ** 0.6) * (pcls.max(dim=-1)[0] ** 0.4)
    bboxes[..., 5] = pcls.argmax(dim=-1)

    # Bounding box coordinates
    gy, gx = torch.meshgrid([torch.arange(H), torch.arange(W)], indexing="ij")
    bw, bh = preg[..., 2].sigmoid(), preg[..., 3].sigmoid() 
    bcx = (preg[..., 0].tanh() + gx.to(device)) / W
    bcy = (preg[..., 1].tanh() + gy.to(device)) / H

    # cx,cy,w,h = > x1,y1,x2,y1
    x1, y1 = bcx - 0.5 * bw, bcy - 0.5 * bh
    x2, y2 = bcx + 0.5 * bw, bcy + 0.5 * bh

    bboxes[..., 0], bboxes[..., 1] = x1, y1
    bboxes[..., 2], bboxes[..., 3] = x2, y2
    bboxes = bboxes.reshape(N, H*W, 6)
    total_bboxes.append(bboxes)
        
    batch_bboxes = torch.cat(total_bboxes, 1)

    # Apply NMS to bounding boxes
    for p in batch_bboxes:
        output, temp = [], []
        b, s, c = [], [], []
        # Threshold filtering
        t = p[:, 4] > conf_thresh
        pb = p[t]
        for bbox in pb:
            obj_score = bbox[4]
            category = bbox[5]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[2], bbox[3]
            s.append([obj_score])
            c.append([category])
            b.append([x1, y1, x2, y2])
            temp.append([x1, y1, x2, y2, obj_score, category])
        # Torchvision NMS
        if len(b) > 0:
            b = torch.Tensor(b).to(device)
            c = torch.Tensor(c).squeeze(1).to(device)
            s = torch.Tensor(s).squeeze(1).to(device)
            keep = torchvision.ops.batched_nms(b, s, c, nms_thresh)
            for i in keep:
                output.append(temp[i])
        output_bboxes.append(torch.Tensor(output))
    return output_bboxes
