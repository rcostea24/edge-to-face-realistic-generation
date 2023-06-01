import torch
import torch.nn as nn

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect')
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        print(x.shape)
        x = self.maxpool(self.conv1(x))
        print(x.shape)
        x = self.maxpool(self.conv2(x))
        print(x.shape)
        x = self.maxpool(self.conv3(x))
        print(x.shape)
        x = self.conv4(x)
        print(x.shape)
        x = self.conv5(x)
        print(x.shape)
        return x

# Generator
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels) 

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x
    
class AttentionGate(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(n_channels)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(n_channels)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(n_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(1),
        )
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, g, x):
        """
        params: g - last layer output
                x - skip connection
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        t = self.relu(torch.add(g1, x1))
        t = self.sigmoid(self.psi(t))
        out = torch.mul(t, x)
        return out    

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_dropout = False, bn=False):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) 
        self.double_conv = DoubleConv(out_channels, out_channels)
            
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x1, x2=None):
        """
        params: x1 - last layer output
                x2 - skip connection
        """
        if x2 != None:
            x = torch.cat([x1, x2], dim=1)
        else:
            x = x1
        x = self.up(x)
        out = self.double_conv(x)
        return self.dropout(out) if self.use_dropout else out

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()
        
        self.down0 = DoubleConv(in_channels, out_channels) # 256x256x1 -> 256x256x32
        
        self.down1 = Down(out_channels, out_channels * 2) # 256x256x32 -> 128x128x64
        self.down2 = Down(out_channels * 2, out_channels * 4) # 128x128x64 -> 64x64x128
        self.down3 = Down(out_channels * 4, out_channels * 8) # 64x64x128 -> 32x32x256
        self.down4 = Down(out_channels * 8, out_channels * 16) # 32x32x256 -> 16x16x512
        self.down5 = Down(out_channels * 16, out_channels * 16) # 16x16x512 -> 8x8x512
        self.down6 = Down(out_channels * 16, out_channels * 16) # 8x8x512 -> 4x4x512
        self.down7 = Down(out_channels * 16, out_channels * 16) # 4x4x512 -> 2x2x512
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(out_channels * 16, out_channels * 16, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        ) # 2x2x512 -> 1x1x512

        self.up1 = Up(out_channels * 16, out_channels * 16, use_dropout=True, bn=True) # 1x1x512 -> 2x2x512

        self.att1 = AttentionGate(out_channels * 16) 
        self.up2 = Up(out_channels * 16 * 2, out_channels * 16, use_dropout=True) # 2x2x512 + 2x2x512 -> 2x2x1024 -> 4x4x512

        self.att2 = AttentionGate(out_channels * 16)
        self.up3 = Up(out_channels * 16 * 2, out_channels * 16, use_dropout=True) # 4x4x512 + 4x4x512 -> 4x4x1024 -> 8x8x512

        self.att3 = AttentionGate(out_channels * 16)
        self.up4 = Up(out_channels * 16 * 2, out_channels * 16, use_dropout=False) # 8x8x512 + 8x8x512 -> 8x8x1024 -> 16x16x512

        self.att4 = AttentionGate(out_channels * 16)
        self.up5 = Up(out_channels * 16 * 2, out_channels * 8, use_dropout=False) # 16x16x512 + 16x16x512 -> 16x16x1024 -> 32x32x256

        self.att5 = AttentionGate(out_channels * 8)
        self.up6 = Up(out_channels * 8 * 2, out_channels * 4, use_dropout=False) # 32x32x256 + 32x32x256 -> 32x32x512 -> 64x64x512 -> 64x64x128

        self.att6 = AttentionGate(out_channels * 4)
        self.up7 = Up(out_channels * 4 * 2, out_channels * 2, use_dropout=False) # 64x64x128 + 64x64x128 -> 64x64x256 -> 128x128x256 -> 128x128x64

        self.att7 = AttentionGate(out_channels * 2)
        self.up8 = Up(out_channels * 2 * 2, out_channels, use_dropout=False) # 128x128x64 + 128x128x64 -> 128x128x128 -> 256x256x32
        
        self.att8 = AttentionGate(out_channels)
        self.up9 = nn.Sequential(nn.Conv2d(out_channels * 2, 3, kernel_size=3, stride=1, padding=1), nn.Tanh()) # 256x256x32 + 256x256x32 -> 256x256x64 -> 256x256x3
        
    def forward(self, x):
        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bn = self.bottleneck(d7)

        u1 = self.up1(bn)

        d7 = self.att1(u1, d7)
        u2 = self.up2(u1, d7)

        d6 = self.att2(u2, d6)
        u3 = self.up3(u2, d6)
        
        d5 = self.att3(u3, d5)
        u4 = self.up4(u3, d5)
        
        d4 = self.att4(u4, d4)
        u5 = self.up5(u4, d4)

        d3 = self.att5(u5, d3)
        u6 = self.up6(u5, d3)

        d2 = self.att6(u6, d2)
        u7 = self.up7(u6, d2)

        d1 = self.att7(u7, d1)
        u8 = self.up8(u7, d1)

        d0 = self.att8(u8, d0)
        out = self.up9(torch.cat([u8, d0], dim=1))
        
        return out
        
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
if __name__ == '__main__':
    x = torch.rand([1, 1, 256, 256])
    #z = torch.randn([1, 1, 256, 256])
    #x = torch.cat([x, z], dim=1)
    #gen = Generator()
    #gen(x)
    #count_parameters(gen)
    
    x = torch.rand([1, 1, 256, 256])
    y = torch.rand([1, 3, 256, 256])
    disc = Discriminator()
    disc(x,y)
    #count_parameters(disc)

    
