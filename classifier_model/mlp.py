import torch
import torch.nn as nn
import numpy as np
import random


class MLP(nn.Module):
    def __init__(self, num_input_channels=784, hidden_unit=(256, 128), num_classes=10,
                 data_parallel=True, drop_rate=0.0):
        super(MLP, self).__init__()
        # self.classifier = nn.Sequential(
        #     nn.Linear(784, hidden_unit[0]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(hidden_unit[0], hidden_unit[1]),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(hidden_unit[1], num_classes)
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (4, 4), stride=2, padding=1),
            nn.ReLU()
        )
        self.encoder = nn.DataParallel(self.encoder)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )
        self.classifier = nn.DataParallel(self.classifier)
        # for name, param in self.named_parameters():
        #     if 'fc' in name and 'weight' in name:
        #         nn.init.xavier_uniform_(param.data)
        #     elif 'fc' in name and 'bias' in name:
        #         param.data.fill_(0)

    def forward(self, input_img):
        batch_size = input_img.size(0)
        features = self.encoder(input_img)
        features = features.view(batch_size, -1)
        cls_result = self.classifier(features)
        return cls_result
