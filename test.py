import os
import cv2
import onnx
import time
import argparse
import warnings
from onnxsim import simplify

import torch
from utils.tool import *
from module.detector import Detector, normalize_pyramid_levels

BASE_STAGE_REPEATS = [4, 8, 4]
BASE_STAGE_OUT_CHANNELS = [-1, 24, 48, 96, 192]

# Suppress noisy future warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)

def _validate_half_step(value, name):
    if value is None:
        return
    if abs(value * 2 - round(value * 2)) > 1e-6:
        raise ValueError(f"{name} must be in 0.5 increments.")

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
    parser.add_argument('--img', type=str, default='', help='The path of test image')
    parser.add_argument('--thresh', type=float, default=0.65, help='The path of test image')
    parser.add_argument('--onnx', action="store_true", default=False, help='Export onnx file')
    parser.add_argument('--torchscript', action="store_true", default=False, help='Export torchscript file')
    parser.add_argument('--cpu', action="store_true", default=False, help='Run on cpu')
    parser.add_argument('--stage-out-channels', type=float, default=1.0, help='stage_out_channels multiplier (0.5 step)')
    parser.add_argument('--stage-repeats', type=float, default=1.0, help='stage_repeats multiplier (0.5 step)')
    parser.add_argument('--pyramid-levels', type=str, default="P1,P2,P3", help='comma-separated pyramid levels to fuse (P1,P2,P3)')
    parser.add_argument('--multi-label-robust-mode', action='store_true', default=False, help='enable multi-label robust inference')
    se_group = parser.add_mutually_exclusive_group()
    se_group.add_argument('--use-se', action='store_true', default=False, help='enable SE on shared features')
    se_group.add_argument('--use-ese', action='store_true', default=False, help='enable eSE on shared features')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "Please provide a valid config file path."
    assert os.path.exists(opt.weight), "Please provide a valid model path."
    assert os.path.exists(opt.img), "Please provide a valid test image path."

    # Select inference backend
    if opt.cpu:
        print("run on cpu...")
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            print("run on gpu...")
            device = torch.device("cuda")
        else:
            print("run on cpu...")
            device = torch.device("cpu")

    # Parse yaml config
    cfg = LoadYaml(opt.yaml)
    print(cfg)

    # Load model
    print("load weight from:%s"%opt.weight)
    _validate_half_step(opt.stage_out_channels, "stage_out_channels")
    _validate_half_step(opt.stage_repeats, "stage_repeats")
    stage_out_channels = _scale_stage_list(BASE_STAGE_OUT_CHANNELS, opt.stage_out_channels, keep_first=True)
    stage_repeats = _scale_stage_list(BASE_STAGE_REPEATS, opt.stage_repeats)
    pyramid_levels = normalize_pyramid_levels(opt.pyramid_levels)
    model = Detector(
        cfg.category_num,
        True,
        stage_repeats=stage_repeats,
        stage_out_channels=stage_out_channels,
        use_ese=opt.use_ese,
        use_se=opt.use_se,
        multi_label=opt.multi_label_robust_mode,
        pyramid_levels=pyramid_levels,
    ).to(device)
    model.load_state_dict(torch.load(opt.weight, map_location=device))
    #sets the module in eval node
    model.eval()

    # Data preprocessing
    ori_img = cv2.imread(opt.img)
    res_img = cv2.resize(ori_img, (cfg.input_width, cfg.input_height), interpolation = cv2.INTER_LINEAR)
    img = res_img.reshape(1, cfg.input_height, cfg.input_width, 3)
    img = torch.from_numpy(img.transpose(0, 3, 1, 2))
    img = img.to(device).float() / 255.0

    # Export ONNX
    if opt.onnx:
        torch.onnx.export(model,                     # model being run
                          img,                       # model input (or a tuple for multiple inputs)
                          "./FastestDet.onnx",       # where to save the model (can be a file or file-like object)
                          export_params=True,        # store the trained parameter weights inside the model file
                          opset_version=11,          # the ONNX version to export the model to
                          do_constant_folding=True)  # whether to execute constant folding for optimization
        # onnx-sim
        onnx_model = onnx.load("./FastestDet.onnx")  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        print("onnx sim sucess...")
        onnx.save(model_simp, "./FastestDet.onnx")

    # Export TorchScript
    if opt.torchscript:
        import copy
        model_cpu = copy.deepcopy(model).cpu()
        x = torch.rand(1, 3, cfg.input_height, cfg.input_width)
        mod = torch.jit.trace(model_cpu, x)
        mod.save("./FastestDet.pt")
        print("to convert torchscript to pnnx/ncnn: ./pnnx FastestDet.pt inputshape=[1,3,%d,%d]" % (cfg.input_height, cfg.input_height))

    # Model inference
    start = time.perf_counter()
    preds = model(img)
    end = time.perf_counter()
    time = (end - start) * 1000.
    print("forward time:%fms"%time)

    # Feature map post-processing
    output = handle_preds(preds, device, opt.thresh, multi_label=opt.multi_label_robust_mode)

    # Load label names
    LABEL_NAMES = []
    with open(cfg.names, 'r') as f:
        for line in f.readlines():
            LABEL_NAMES.append(line.strip())

    H, W, _ = ori_img.shape
    scale_h, scale_w = H / cfg.input_height, W / cfg.input_width

    # Draw predicted boxes
    for box in output[0]:
        print(box)
        box = box.tolist()

        obj_score = box[4]
        category = LABEL_NAMES[int(box[5])]

        x1, y1 = int(box[0] * W), int(box[1] * H)
        x2, y2 = int(box[2] * W), int(box[3] * H)

        cv2.rectangle(ori_img, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(ori_img, '%.2f' % obj_score, (x1, y1 - 5), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(ori_img, category, (x1, y1 - 25), 0, 0.7, (0, 255, 0), 2)

    cv2.imwrite("result.png", ori_img)
