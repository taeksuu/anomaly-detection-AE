import torch.nn as nn


class LargeAE(nn.Module):
    # User-defined autoencoder model for better image reconstruction
    def __init__(self, color_mode, latent_space_dim=2048):
        super(LargeAE, self).__init__()

        self.color_mode = color_mode
        self.latent_space_dim = latent_space_dim

        if color_mode == "rgb":
            channels = 3
        else:
            channels = 1

        # encoder
        self.encoder = nn.Sequential(
            # Additional Conv: 256 * 256 * 1
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv1
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv5
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv6
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv7
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv8
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv9
            nn.Conv2d(in_channels=256, out_channels=self.latent_space_dim, kernel_size=8, stride=1, padding=0)

        )

        # decoder
        self.decoder = nn.Sequential(

            # Conv9 reversed
            nn.ConvTranspose2d(in_channels=self.latent_space_dim, out_channels=256, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv8 reversed
            nn.ConvTranspose2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv7 reversed
            nn.ConvTranspose2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv6 reversed
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv5 reversed
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv4 reversed
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv3 reversed
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv2 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Conv1 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),

            # Additional Conv reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1),

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
