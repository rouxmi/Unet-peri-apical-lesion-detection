import torch
import torch.nn as nn

class CustomUNet(nn.Module):
    """
    Custom implementation of the U-Net architecture for image segmentation.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (tuple): Number of channels in each layer of the encoder and decoder.
        strides (tuple): Stride values for max pooling layers in the encoder.
        num_res_units (int): Number of residual units in each convolutional block.
        norm (nn.Module): Normalization layer to be used (default: nn.BatchNorm2d).
        act (nn.Module): Activation function to be used (default: nn.LeakyReLU).

    Attributes:
        encoder1 (nn.Sequential): Encoder block 1.
        pool1 (nn.MaxPool2d): Max pooling layer 1.
        encoder2 (nn.Sequential): Encoder block 2.
        pool2 (nn.MaxPool2d): Max pooling layer 2.
        encoder3 (nn.Sequential): Encoder block 3.
        pool3 (nn.MaxPool2d): Max pooling layer 3.
        encoder4 (nn.Sequential): Encoder block 4.
        pool4 (nn.MaxPool2d): Max pooling layer 4.
        bottleneck (nn.Sequential): Bottleneck block.
        upconv4 (nn.ConvTranspose2d): Upsampling convolutional layer 4.
        decoder4 (nn.Sequential): Decoder block 4.
        upconv3 (nn.ConvTranspose2d): Upsampling convolutional layer 3.
        decoder3 (nn.Sequential): Decoder block 3.
        upconv2 (nn.ConvTranspose2d): Upsampling convolutional layer 2.
        decoder2 (nn.Sequential): Decoder block 2.
        upconv1 (nn.ConvTranspose2d): Upsampling convolutional layer 1.
        decoder1 (nn.Sequential): Decoder block 1.
        conv (nn.Conv2d): Convolutional layer for final output.

    """

    def __init__(self, in_channels, out_channels, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2), num_res_units=2, norm=nn.BatchNorm2d, act=nn.LeakyReLU):
        super(CustomUNet, self).__init__()

        self.encoder1 = self.conv_block(in_channels, channels[0])
        self.pool1 = nn.MaxPool2d(kernel_size=strides[0], stride=strides[0])
        self.encoder2 = self.conv_block(channels[0], channels[1])
        self.pool2 = nn.MaxPool2d(kernel_size=strides[1], stride=strides[1])
        self.encoder3 = self.conv_block(channels[1], channels[2])
        self.pool3 = nn.MaxPool2d(kernel_size=strides[2], stride=strides[2])
        self.encoder4 = self.conv_block(channels[2], channels[3])
        self.pool4 = nn.MaxPool2d(kernel_size=strides[3], stride=strides[3])

        self.bottleneck = self.conv_block(channels[3], channels[4])

        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(channels[3] * 2, channels[3])
        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(channels[2] * 2, channels[2])
        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(channels[1] * 2, channels[1])
        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(channels[0] * 2, channels[0])

        self.conv = nn.Conv2d(in_channels=channels[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )


