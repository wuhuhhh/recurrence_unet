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
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
        super(UNetEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder_layers.append(DoubleConv(in_channels, feature))
            in_channels = feature

    def forward(self, x):
        skip_connections = []
        for down in self.encoder_layers:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
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
            # 关键修正：双卷积的输入通道数 = feature + 对应编码器层的输出通道数
            # 对于第一个解码层：512 + 512 = 1024
            self.decoder_layers.append(
                DoubleConv(feature + feature, feature)  # 修正：feature + feature
            )
            in_channels = feature

    def forward(self, x, skip_connections):
        # 关键修正：只使用前4个跳跃连接，去掉最后一个
        skip_connections = skip_connections[:-1][::-1]  # 去掉最后一个，然后反转

        for idx, (up_conv, double_conv) in enumerate(zip(self.up_convs, self.decoder_layers)):
            # 上采样
            x = up_conv(x)

            # 获取对应的跳跃连接
            skip_connection = skip_connections[idx]

            # 尺寸对齐
            if x.shape[2:] != skip_connection.shape[2:]:
                target_height = skip_connection.shape[2]
                target_width = skip_connection.shape[3]
                x = torch.nn.functional.interpolate(
                    x,
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=True
                )

            # 通道拼接
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # 双卷积
            x = double_conv(concat_skip)

        return x

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(in_channels, features)
        decoder_features = features[:-1][::-1]  # [512, 256, 128, 64]
        self.decoder = UNetDecoder(
            features=decoder_features,
            bottleneck_channels=features[-1]  # 1024
        )
        self.final_conv = nn.Conv2d(decoder_features[-1], out_channels, kernel_size=1)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        x = self.decoder(bottleneck, skip_connections)
        return self.final_conv(x)


def test_decoder():
    print("=== 测试UNet解码器 ===")

    # 创建测试数据
    batch_size, channels, height, width = 1, 3, 512, 512
    x = torch.randn((batch_size, channels, height, width))
    print(f"输入图像尺寸: {x.shape}")

    # 测试编码器
    encoder = UNetEncoder(in_channels=3, features=[64, 128, 256, 512, 1024])
    encoder_output, skip_connections = encoder(x)

    print("\n=== 编码器输出 ===")
    print(f"瓶颈层输出尺寸: {encoder_output.shape}")
    for i, skip in enumerate(skip_connections):
        print(f"Skip connection {i}: {skip.shape}")

    # 测试解码器
    print("\n=== 解码器测试 ===")
    decoder_features = [512, 256, 128, 64]
    decoder = UNetDecoder(features=decoder_features, bottleneck_channels=1024)

    # 解码器前向传播
    decoder_output = decoder(encoder_output, skip_connections)
    print(f"解码器输出尺寸: {decoder_output.shape}")


def test_complete_unet():
    print("\n" + "=" * 50)
    print("=== 测试完整UNet模型 ===")

    # 创建测试数据
    x = torch.randn((1, 3, 512, 512))
    print(f"输入尺寸: {x.shape}")

    # 创建完整UNet
    model = UNet(in_channels=3, out_channels=1, features=[64, 128, 256, 512, 1024])

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
    try:
        test_decoder()
        output = test_complete_unet()
        print("\n🎉 UNet实现成功！所有测试通过！")
        print(f"最终输出范围: [{output.min():.3f}, {output.max():.3f}]")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()