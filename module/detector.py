import re
import torch
import torch.nn as nn

from .shufflenetv2 import ShuffleNetV2
from .custom_layers import DetectHead, SPP, ESE, SE

VALID_PYRAMID_LEVELS = ("P1", "P2", "P3")

def normalize_pyramid_levels(levels):
    if levels is None:
        return VALID_PYRAMID_LEVELS
    if isinstance(levels, (list, tuple)):
        raw_levels = [str(item).strip() for item in levels]
    elif isinstance(levels, str):
        raw_levels = re.split(r"[,\s]+", levels.strip())
    else:
        raise TypeError("pyramid_levels must be a string, list, tuple, or None.")
    normalized = []
    for level in raw_levels:
        if not level:
            continue
        key = level.upper()
        if key not in VALID_PYRAMID_LEVELS:
            raise ValueError(f"Invalid pyramid level: {level}")
        if key not in normalized:
            normalized.append(key)
    if not normalized:
        raise ValueError("pyramid_levels must include at least one of P1, P2, or P3.")
    return tuple(normalized)

class Detector(nn.Module):
    def __init__(
        self,
        category_num,
        load_param,
        input_channels=3,
        stage_repeats=None,
        stage_out_channels=None,
        use_skip_residual=False,
        use_ese=False,
        use_se=False,
        pyramid_levels=None,
    ):
        super(Detector, self).__init__()

        self.stage_repeats = stage_repeats or [4, 8, 4]
        self.stage_out_channels = stage_out_channels or [-1, 24, 48, 96, 192]
        self.backbone = ShuffleNetV2(
            self.stage_repeats,
            self.stage_out_channels,
            load_param,
            in_channels=input_channels,
            use_skip_residual=use_skip_residual,
        )

        self.pyramid_levels = normalize_pyramid_levels(pyramid_levels)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        level_channels = {
            "P1": self.stage_out_channels[-3],
            "P2": self.stage_out_channels[-2],
            "P3": self.stage_out_channels[-1],
        }
        spp_in_channels = sum(level_channels[level] for level in self.pyramid_levels)
        self.SPP = SPP(spp_in_channels, self.stage_out_channels[-2])
        self.ese = ESE(self.stage_out_channels[-2]) if use_ese else None
        self.se = SE(self.stage_out_channels[-2]) if use_se else None
         
        self.detect_head = DetectHead(self.stage_out_channels[-2], category_num)

    def forward(self, x):
        P1, P2, P3 = self.backbone(x)
        if len(self.pyramid_levels) > 1:
            if "P1" in self.pyramid_levels:
                P1 = self.avg_pool(P1)
            if "P3" in self.pyramid_levels:
                P3 = self.upsample(P3)
        level_map = {"P1": P1, "P2": P2, "P3": P3}
        features = [level_map[level] for level in self.pyramid_levels]
        if len(features) == 1:
            P = features[0]
        else:
            P = torch.cat(features, dim=1)

        y = self.SPP(P)
        if self.se is not None:
            y = self.se(y)
        elif self.ese is not None:
            y = self.ese(y)

        return self.detect_head(y)

if __name__ == "__main__":
    model = Detector(80, False, input_channels=3)
    test_data = torch.rand(1, 3, 352, 352)
    torch.onnx.export(model,                    #model being run
                     test_data,                 # model input (or a tuple for multiple inputs)
                     "./test.onnx",             # where to save the model (can be a file or file-like object)
                     export_params=True,        # store the trained parameter weights inside the model file
                     opset_version=11,          # the ONNX version to export the model to
                     do_constant_folding=True)  # whether to execute constant folding for optimization
