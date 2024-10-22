
# Ghost Module
class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, silu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if silu else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.SiLU(inplace=True) if silu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


# Squeeze Excitation Block
class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y



# Ghost Bottleneck
class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, silu=True),
            # dw
            DWConv(hidden_dim, hidden_dim, kernel_size, stride, silu=False) if stride == 2 else nn.Identity(),
            # squeeze-and-excite
            SELayer(hidden_dim) if use_se else nn.Identity(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, silu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, kernel_size, stride, silu=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup)
            )
    
    def forward(self, x) :
        return self.conv(x) + self.shortcut(x)


# Global Attention Mechanism (GAM) module
class GAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GAM, self).__init__()
        self.channels = channels

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels // reduction, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        chn_att = self.channel_attention(x)
        x = x * chn_att

        # Spatial attention
        spa_att = self.spatial_attention(x)
        x = x * spa_att

        return x