class AutoEncoder(nn.Module):
    def __init__(self, color_mode):
        super(AutoEncoder, self).__init__()
        
        if color_mode == "rgb":
            channels = 3
        else: channels = 1

        self.encoder = nn.Sequential(
            #Additional Conv: 256 * 256 * 1
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.02),
          
            #Conv1
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            
            #Conv4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            
            #Conv8
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02),
            
            #Conv9
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=8, stride=1, padding=0)
        )
        
        self.decoder = nn.Sequential(
            #Conv9 reversed
            nn.ConvTranspose2d(in_channels=1, out_channels=32, kernel_size=8, stride=1, padding=0),
            nn.LeakyReLU(0.02),
            
            #Conv8 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.02),
            
            #Conv4 reversed
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            
            #Conv1 reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.02),
            
            #Additional Conv reversed
            nn.ConvTranspose2d(in_channels=32, out_channels=channels, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
