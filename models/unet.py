import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512,1024]):
        super(UNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 添加初始卷积层
        self.initial_conv = DoubleConv(in_channels, features[0])  # 3→64

        # 构建encoder_layers，从features[0]开始作为输入通道
        current_channels = features[0]  # 64
        for feature in features[1:]:  # 从128开始
            self.encoder_layers.append(DoubleConv(current_channels, feature))
            current_channels = feature

    def forward(self, x):
        # 使用初始卷积
        x = self.initial_conv(x)  # [batch, 64, H, W]

        skip_connections = []
        for down in self.encoder_layers:
            skip_connections.append(x)  # 保存当前特征图
            x = self.pool(x)  # 下采样
            x = down(x)  # 双卷积

        encoder_output = x
        return encoder_output, skip_connections
class UNetDecoder(nn.Module):
    def __init__(self, features=[512, 256, 128, 64], bottleneck_channels=1024):
        super(UNetDecoder, self).__init__()

        self.decoder_layers = nn.ModuleList()
        self.up_convs = nn.ModuleList()


        # 构建上采样层：从瓶颈层通道数开始
        in_channels = bottleneck_channels
        for feature in features:
            self.up_convs.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )

            self.decoder_layers.append(
                DoubleConv(feature + feature, feature)
            )
            in_channels = feature

    def forward(self, x, skip_connections):
        # 关键修正：只使用前4个跳跃连接，去掉最后一个
        skip_connections = skip_connections[::-1]  # 去掉最后一个，然后反转

        for idx, (up_conv, double_conv) in enumerate(zip(self.up_convs, self.decoder_layers)):
            # 上采样
            x = up_conv(x)

            # 获取对应的跳跃连接
            skip_connection = skip_connections[idx]

            # 尺寸对齐
            # if x.shape[2:] != skip_connection.shape[2:]:
            #     target_height = skip_connection.shape[2]
            #     target_width = skip_connection.shape[3]
            #     x = torch.nn.functional.interpolate(
            #         x,
            #         size=(target_height, target_width),
            #         mode='bilinear',
            #         align_corners=True
            #     )

            # 通道拼接
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # 双卷积
            x = double_conv(concat_skip)

        return x

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, features=[64, 128, 256, 512,1024]):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.encoder = UNetEncoder(n_channels, features)
        decoder_features = features[:-1][::-1]  # [512, 256, 128, 64]
        self.decoder = UNetDecoder(
            features=decoder_features,
            bottleneck_channels=features[-1]
        )
        self.final_conv = nn.Conv2d(decoder_features[-1], n_classes, kernel_size=1)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        x = self.decoder(bottleneck, skip_connections)
        return self.final_conv(x)


def test_complete_unet():
    print("\n" + "=" * 50)
    print("=== 测试完整UNet模型 ===")

    # 创建测试数据
    x = torch.randn((1, 3, 512, 512))
    print(f"输入尺寸: {x.shape}")

    # 创建完整UNet
    model = UNet(n_channels=3, n_classes=2, features=[64, 128, 256, 512,1024])

    # 前向传播
    with torch.no_grad():
        output = model(x)

    print(f"UNet输出尺寸: {output.shape}")

    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params:,}")

    return output


if __name__ == "__main__":
    # 运行测试
    test_complete_unet()
    # try:
    #     test_decoder()
    #     output = test_complete_unet()
    #     print("\n🎉 UNet实现成功！所有测试通过！")
    #     print(f"最终输出范围: [{output.min():.3f}, {output.max():.3f}]")
    # except Exception as e:
    #     print(f"\n❌ 测试失败: {e}")
    #     import traceback
    #
    #     traceback.print_exc()