import torch.nn as nn


class _PreProcess(nn.Sequential):
    def __init__(self, num_input_channels, num_init_features=64, small_input=True):
        super(_PreProcess, self).__init__()
        if small_input:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=3, stride=1, padding=1,
                                      bias=True))
        else:
            self.add_module('conv0',
                            nn.Conv2d(num_input_channels, num_init_features, kernel_size=7, stride=2, padding=3,
                                      bias=True))
            self.add_module('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1,
                                                  ceil_mode=False))


class _PreActUnit(nn.Module):
    """
    Pre Activation of the Basic Block
    """

    def __init__(self, num_input_features, num_output_features, expansion=1, stride=1, drop_rate=0.0):
        super(_PreActUnit, self).__init__()
        self._expansion = expansion
        self.f_block = nn.Sequential()
        if self._expansion == 1:
            self.f_block.add_module('norm1', nn.BatchNorm2d(num_input_features))
            self.f_block.add_module('relu1', nn.ReLU(inplace=True))
            self.f_block.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                                       kernel_size=3, stride=stride, padding=1, bias=False))
            self.f_block.add_module('dropout', nn.Dropout(drop_rate))
            self.f_block.add_module('norm2', nn.BatchNorm2d(num_output_features))
            self.f_block.add_module('relu2', nn.ReLU(inplace=True))
            self.f_block.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                                       kernel_size=3, stride=1, padding=1, bias=False))
        else:
            self.f_block.add_module('norm1', nn.BatchNorm2d(num_input_features))
            self.f_block.add_module('relu1', nn.ReLU(inplace=True))
            self.f_block.add_module('conv1', nn.Conv2d(num_input_features, num_output_features,
                                                       kernel_size=1, bias=False))
            self.f_block.add_module('norm2', nn.BatchNorm2d(num_output_features))
            self.f_block.add_module('relu2', nn.ReLU(inplace=True))
            self.f_block.add_module('conv2', nn.Conv2d(num_output_features, num_output_features,
                                                       kernel_size=3, stride=stride, padding=1, bias=False))
            self.f_block.add_module('dropout', nn.Dropout(drop_rate))
            self.f_block.add_module('norm3', nn.BatchNorm2d(num_output_features))
            self.f_block.add_module('relu3', nn.ReLU(inplace=True))
            self.f_block.add_module('conv3', nn.Conv2d(num_output_features, self._expansion * num_output_features,
                                                       kernel_size=1, bias=False))
        if stride != 1 or num_input_features != self._expansion * num_output_features:
            self.i_block = nn.Sequential()
            self.i_block.add_module('norm', nn.BatchNorm2d(num_input_features))
            # self.i_block.add_module('relu', nn.ReLU(inplace=True))
            self.i_block.add_module('conv',
                                    nn.Conv2d(num_input_features, self._expansion * num_output_features, kernel_size=1,
                                              stride=stride,
                                              bias=False))

    def forward(self, x):
        new_features = self.f_block(x)
        if hasattr(self, "i_block"):
            x = self.i_block(x)
        return new_features + x


class _PreActBlock(nn.Module):
    def __init__(self, input_channel, output_channel, expansion, block_depth, down_sample=False, drop_rate=0.0):
        super(_PreActBlock, self).__init__()
        self.preact_block = nn.Sequential()
        for i in range(block_depth):
            if i == 0:
                unit = _PreActUnit(input_channel, output_channel, expansion, stride=int(1 + down_sample),
                                   drop_rate=drop_rate)
            else:
                unit = _PreActUnit(input_channel, output_channel, expansion, drop_rate=drop_rate)
            self.preact_block.add_module("unit%d" % (i + 1), unit)
            input_channel = output_channel * expansion

    def forward(self, x):
        return self.preact_block(x)


class PreActResNet(nn.Module):
    def __init__(self, expansion, block_config, num_input_channels=1, num_init_features=64,
                 data_parallel=True, small_input=False, drop_rate=0.0):
        super(PreActResNet, self).__init__()
        self._input_channels = num_init_features
        self._output_channels = num_init_features
        self.encoder = nn.Sequential()
        self._block_config = block_config
        pre_process = _PreProcess(num_input_channels, num_init_features, small_input=small_input)
        if data_parallel:
            pre_process = nn.DataParallel(pre_process)
        self.encoder.add_module("pre_process", pre_process)
        for idx, block_depth in enumerate(block_config):
            block = _PreActBlock(self._input_channels, self._output_channels, expansion, block_depth,
                                 down_sample=(idx != 0), drop_rate=drop_rate)
            # update the channel num
            self._input_channels = self._output_channels * expansion
            self._output_channels = self._output_channels * 2
            if data_parallel:
                block = nn.DataParallel(block)
            self.encoder.add_module("block%d" % (idx + 1), block)
        # we may use norm and relu before the global avg. Standard implementation doesn't use
        trans = nn.Sequential()
        trans.add_module("norm", nn.BatchNorm2d(int(num_init_features * (2 ** (len(block_config) - 1)) * expansion)))
        trans.add_module("relu", nn.ReLU(inplace=True))
        if data_parallel:
            trans = nn.DataParallel(trans)
        self.encoder.add_module('transition', trans)
        self.num_feature_channel = int(num_init_features * (2 ** (len(block_config) - 1)) * expansion)

    def forward(self, input_img):
        features = self.encoder(input_img)
        return features


preactresnet_dict = {
    "preactresnet18": {"expansion": 1, "block_config": [2, 2, 2, 2]},
    "preactresnet34": {"expansion": 1, "block_config": [3, 4, 6, 3]},
    "preactresnet50": {"expansion": 4, "block_config": [3, 4, 6, 3]},
    "preactresnet101": {"expansion": 4, "block_config": [3, 4, 23, 3]},
    "preactresnet152": {"expansion": 4, "block_config": [3, 8, 36, 3]}
}


def get_preact_resnet(name, drop_rate, input_channels=1, small_input=False, data_parallel=True):
    return PreActResNet(num_input_channels=input_channels,
                        expansion=preactresnet_dict[name]["expansion"],
                        block_config=preactresnet_dict[name]["block_config"], drop_rate=drop_rate,
                        data_parallel=data_parallel, small_input=small_input)
