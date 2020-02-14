from torch import nn


class Decoder(nn.Module):
    def __init__(self,
                 latent_dim=100,
                 num_feature=64,
                 num_channel=1,
                 data_parallel=True,
                 kernel_size=(5, 6)):
        super(Decoder, self).__init__()
        decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim,
                               num_feature * 16,
                               kernel_size,
                               1,
                               0,
                               bias=False),
            nn.BatchNorm2d(num_feature * 16),
            nn.ReLU(True),
            # state size. (num_feature*32) x 5 x 6
            nn.ConvTranspose2d(num_feature * 16,
                               num_feature * 8,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(num_feature * 8),
            nn.ReLU(True),
            # state size. (num_feature*16) x 10 x 12
            nn.ConvTranspose2d(num_feature * 8,
                               num_feature * 4,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(num_feature * 4),
            nn.ReLU(True),
            # state size. (num_feature*8) x 20 x 24
            nn.ConvTranspose2d(num_feature * 4,
                               num_feature * 2,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(num_feature * 2),
            nn.ReLU(True),
            # state size. (num_feature*4) x 40 x 48
            nn.ConvTranspose2d(num_feature * 2,
                               num_feature,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(num_feature),
            nn.ReLU(True),
            # state size. (num_feature*2) x 80 x 96
            nn.ConvTranspose2d(num_feature, num_channel, 4, 2, 1, bias=False),
            # nn.Tanh()
            # nn.Sigmoid()
            # state size. num_channel x 160 x 192 without use the sigmoid function
        )
        if data_parallel:
            self.decoder = nn.DataParallel(decoder)
        else:
            self.decoder = decoder

    def forward(self, latent_sample):
        return self.decoder(latent_sample)
