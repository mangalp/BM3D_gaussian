import torch
from torch import nn
from torch.nn import functional as F

from .unet import conv3x3, conv1x1, upconv2x2


class DownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 2x2 convolution with stride 2 for downsampling.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool


class UpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', up_mode='pixelshuffle', blur=False):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.out_channels,
                                mode=self.up_mode, blur=blur)

        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(
                2 * self.out_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:
            from_down: tensor from the encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class Encoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 depth=5,
                 start_filts=64,
                 z_dim=4):
        super(Encoder, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.start_filts = start_filts
        self.z_dim = z_dim

        self.down_convs = []

        ins = self.in_channels
        outs = self.start_filts

        down_conv = DownConv(ins, outs, pooling=False)
        self.down_convs.append(down_conv)

        self.convmu = conv3x3(outs, self.z_dim)
        self.convlogvar = conv3x3(outs, self.z_dim)

        self.down_convs = nn.ModuleList(self.down_convs)

    def forward(self, x):
        for module in self.down_convs:
            x, _ = module(x)

        mu = self.convmu(x)
        logvar = self.convlogvar(x)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 depth=5,
                 start_filts=64,
                 up_mode='transpose',
                 blur=True,
                 z_dim=4):
        super(Decoder, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.depth = depth
        self.start_filts = start_filts
        self.up_mode = up_mode
        self.blur = blur
        self.z_dim = z_dim

        self.up_convs = []
        ins = self.z_dim
        outs = self.z_dim
        up_conv = DownConv(ins, outs, pooling=False)
        self.up_convs.append(up_conv)

        self.up_convs = nn.ModuleList(self.up_convs)
        self.conv_final = conv1x1(outs, self.num_classes)

        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)

    def forward(self, x):
        for module in self.up_convs:
            x, _ = module(x)

        x = self.conv_final(x)
        return self.blur(self.pad(x))
    
    
class VAE(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 depth=5,
                 start_filts=64,
                 up_mode='transpose',
                 blur=True,
                 z_dim=4):
        super(VAE, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.start_filts = start_filts
        self.up_mode = up_mode
        self.blur = blur
        self.z_dim = z_dim

        self.encoder = Encoder(in_channels=self.in_channels,
                               num_classes=self.num_classes,
                               depth=self.depth,
                               start_filts=self.start_filts,
                               z_dim=self.z_dim)
        self.decoder = Decoder(in_channels=self.in_channels,
                               num_classes=self.num_classes,
                               depth=self.depth,
                               start_filts=self.start_filts,
                               up_mode=self.up_mode,
                               blur=self.blur,
                               z_dim=self.z_dim)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def reparameterize(self, mu, logvar):
        """
        Uses reparametrization trick as mentioned in the paper https://arxiv.org/abs/1312.6114
        to draw a sample from a normal distribution given its mean and log variance.
        """
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(mu)
        z = mu + epsilon * std
        return z

    def reparameterize_eps(self, mu, logvar, epsilon):
        std = torch.exp(0.5 * logvar)
        z = mu + epsilon * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


class MultiVAE(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=1,
                 depth=3,
                 start_filts=10,
                 blur=True):
        super(MultiVAE, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depth = depth
        self.start_filts = start_filts
        self.blur = blur

        self.vaes = []
        self.down_convs = []
        self.up_convs = []
        for d in range(self.depth):
            ins = self.in_channels if d == 0 else outs
            outs = self.start_filts * (2 ** d)
            self.down_convs.append(DownConv(ins, outs, pooling=d < (self.depth - 1)))

        self.middle_vae = VAE(in_channels=outs,
                              num_classes=outs,
                              depth=0,
                              start_filts=2 * outs,
                              z_dim=outs)

        for i in range(depth - 1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode='pixelshuffle',
                             merge_mode='concat', blur=blur)
            self.up_convs.append(up_conv)
            self.vaes.append(VAE(in_channels=outs,
                                 num_classes=outs,
                                 depth=0,
                                 start_filts=2 * outs,
                                 z_dim=outs))

        self.vaes = nn.ModuleList(self.vaes)
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)

        self.conv_final = conv1x1(outs, self.num_classes)
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)

        self.reset_params()
        self.cuda()
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):

        mus = []
        logvars = []
        conv_skips = []
        for i, conv in enumerate(self.down_convs):
            x, before_pool = conv(x)
            if i < len(self.down_convs) - 1:
                conv_skips.append(before_pool)
        x, mu, logvar = self.middle_vae.forward(x)
        mus.append(mu)
        logvars.append(logvar)

        for up_conv, conv_skip, vae in zip(self.up_convs, reversed(conv_skips), self.vaes):
            x = up_conv.forward(conv_skip, x)
            x, mu, logvar = vae.forward(x)
            mus.append(mu)
            logvars.append(logvar)

        x = self.conv_final(x)
        x = self.blur(self.pad(x))

        return x, mus, logvars