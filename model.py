import torch
import torch.nn as nn
import torch.nn.functional as F


nfc_base = {4:32, 8:32, 16:16, 32:16, 64:8, 128:4, 256:2, 512:1}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def UpBlock(in_ch, out_ch):
    main = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_ch, out_ch*2, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch*2),
        nn.GLU(dim=1),
    )
    main.in_ch = in_ch
    main.out_ch = out_ch
    return main

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        nz = config['nz']
        ngf = config['ngf']

        nfc = {k:int(v*ngf) for k,v in nfc_base.items()}

        self.init = nn.Sequential(
            nn.ConvTranspose2d(nz, nfc[4]*2, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nfc[4]*2),
            nn.GLU(dim=1),
        )

        self.up_8 = UpBlock(nfc[4], nfc[8])
        self.up_16 = UpBlock(nfc[8], nfc[16])
        self.up_32 = UpBlock(nfc[16], nfc[32])
        self.up_64 = UpBlock(nfc[32], nfc[64])
        self.up_128 = UpBlock(nfc[64], nfc[128])

        self.to_128 = nn.Sequential(
            nn.Conv2d(nfc[128], nc, 3, 1, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, noise):
        noise = noise.view(noise.shape[0], -1, 1, 1)
        feat_4 = self.init(noise)

        feat_8 = self.up_8(feat_4)
        feat_16 = self.up_16(feat_8)
        feat_32 = self.up_32(feat_16)
        feat_64 = self.up_64(feat_32)
        feat_128 = self.up_128(feat_64)

        return self.to_128(feat_128)


def DownBlock(in_planes, out_planes, dropout=0.0):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout2d(dropout),
    )

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        nc = config['nc']
        ndf = config['ndf']

        dropout = config['d_dropout']

        nfc = {k:int(v*ndf) for k,v in nfc_base.items()}

        self.start_block = nn.Sequential(
            nn.Conv2d(nc, nfc[64], 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout),
        )

        self.down_to_32 = DownBlock(nfc[64], nfc[32], dropout)
        self.down_to_16 = DownBlock(nfc[32], nfc[16], dropout)
        self.down_to_8 = DownBlock(nfc[16], nfc[8], dropout)

        self.rf_main = nn.Sequential(
            nn.Conv2d(nfc[8], 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

        self.apply(weights_init)


    def forward(self, image):

        feat_64 = self.start_block(image)
        feat_32 = self.down_to_32(feat_64)
        feat_16 = self.down_to_16(feat_32)
        feat_8 = self.down_to_8(feat_16)

        rf = self.rf_main(feat_8)

        return rf

