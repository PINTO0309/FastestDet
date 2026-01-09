import os
import torch
import argparse
import warnings
from torchsummary import summary

from utils.tool import *
from utils.datasets import *
from utils.evaluation import CocoDetectionEvaluator

from module.detector import Detector

# Suppress noisy future warnings from dependencies.
warnings.filterwarnings("ignore", category=FutureWarning)

# Select backend device: CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Training config
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, default="", help='.yaml config')
    parser.add_argument('--weight', type=str, default=None, help='.weight config')

    opt = parser.parse_args()
    assert os.path.exists(opt.yaml), "Please provide a valid config file path."
    assert os.path.exists(opt.weight), "Please provide a valid weight file path."

    # Parse yaml config
    cfg = LoadYaml(opt.yaml)
    print(cfg)

    # Load model weights
    print("load weight from:%s"%opt.weight)
    model = Detector(cfg.category_num, True).to(device)
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
    evaluation.compute_map(val_dataloader, model)
