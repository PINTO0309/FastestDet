import torch
import torch.nn as nn

class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride, use_residual=False):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_residual = use_residual

        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(mid_channels, mid_channels, ksize, stride, pad, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            # pw-linear
            nn.Conv2d(mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        ]
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
            ]
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride==1:
            x_proj, x = self.channel_shuffle(old_x)
            out = torch.cat((x_proj, self.branch_main(x)), 1)
            if self.use_residual:
                out = out + old_x
            return out
        elif self.stride==2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)), 1)

    def channel_shuffle(self, x: torch.Tensor):
        batchsize, num_channels, height, width = x.shape
        x = x.reshape(batchsize, num_channels // 2, 2, height, width)
        return x[:, :, 0], x[:, :, 1]



DEFAULT_STAGE_REPEATS = [4, 8, 4]
DEFAULT_STAGE_OUT_CHANNELS = [-1, 24, 48, 96, 192]
DEFAULT_P1_STRIDE = 8


def _resolve_stem_strides(p1_stride):
    if p1_stride == 8:
        return 2, 2
    if p1_stride == 4:
        return 2, 1
    if p1_stride == 2:
        return 1, 1
    raise ValueError("p1_stride must be one of 2, 4, or 8.")

class ShuffleNetV2(nn.Module):
    def __init__(
        self,
        stage_repeats,
        stage_out_channels,
        load_param,
        in_channels=3,
        use_skip_residual=False,
        p1_stride=DEFAULT_P1_STRIDE,
    ):
        super(ShuffleNetV2, self).__init__()

        self.stage_repeats = stage_repeats
        self.stage_out_channels = stage_out_channels
        self.in_channels = in_channels
        self.use_skip_residual = use_skip_residual
        self.p1_stride = DEFAULT_P1_STRIDE if p1_stride is None else int(p1_stride)
        self.stem_stride, self.maxpool_stride = _resolve_stem_strides(self.p1_stride)

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels, input_channel, 3, self.stem_stride, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.maxpool_stride, padding=1)

        stage_names = ["stage2", "stage3", "stage4"]
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            stageSeq = []
            for i in range(numrepeat):
                if i == 0:
                    stageSeq.append(ShuffleV2Block(input_channel, output_channel,
                                                mid_channels=output_channel // 2, ksize=3, stride=2,
                                                use_residual=self.use_skip_residual))
                else:
                    stageSeq.append(ShuffleV2Block(input_channel // 2, output_channel,
                                                mid_channels=output_channel // 2, ksize=3, stride=1,
                                                use_residual=self.use_skip_residual))
                input_channel = output_channel
            setattr(self, stage_names[idxstage], nn.Sequential(*stageSeq))

        if load_param == False:
            self._initialize_weights()
        else:
            print("load param...")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        P1 = self.stage2(x)
        P2 = self.stage3(P1)
        P3 = self.stage4(P2)

        return P1, P2, P3

    def _initialize_weights(self):
        if (
            self.in_channels != 3
            or self.stage_repeats != DEFAULT_STAGE_REPEATS
            or self.stage_out_channels != DEFAULT_STAGE_OUT_CHANNELS
            or self.p1_stride != DEFAULT_P1_STRIDE
        ):
            print("Skip loading shufflenetv2.pth due to non-default backbone configuration.")
            return
        print("Initialize params from:%s"%"./module/shufflenetv2.pth")
        self.load_state_dict(torch.load("./module/shufflenetv2.pth"), strict = True)
