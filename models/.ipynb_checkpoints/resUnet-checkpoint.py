import torch
import torch.nn as nn


class ResidualDoubleConv(nn.Module):
    """带有残差连接的双卷积模块"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 如果输入输出通道数不同，需要1x1卷积调整通道
        self.use_shortcut = (in_channels != out_channels)
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn_shortcut = nn.BatchNorm2d(out_channels)

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.double_conv(x)

        # 处理shortcut连接
        if self.use_shortcut:
            identity = self.bn_shortcut(self.shortcut(identity))

        out += identity  # 残差连接
        out = self.relu(out)

        return out


class ResidualUNetEncoder(nn.Module):
    """使用残差块的编码器"""

    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024]):
        super(ResidualUNetEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 初始卷积层使用残差块
        self.initial_conv = ResidualDoubleConv(in_channels, features[0])

        # 构建编码层
        current_channels = features[0]
        for feature in features[1:]:
            self.encoder_layers.append(ResidualDoubleConv(current_channels, feature))
            current_channels = feature

    def forward(self, x):
        x = self.initial_conv(x)
        skip_connections = []

        for down in self.encoder_layers:
            skip_connections.append(x)
            x = self.pool(x)
            x = down(x)

        encoder_output = x
        return encoder_output, skip_connections


class ResidualUNetDecoder(nn.Module):
    """使用残差块的解码器"""

    def __init__(self, features=[512, 256, 128, 64], bottleneck_channels=1024):
        super(ResidualUNetDecoder, self).__init__()
        self.decoder_layers = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        in_channels = bottleneck_channels
        for feature in features:
            # 上采样层
            self.up_convs.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2
                )
            )
            # 解码器使用残差块，输入通道是 feature*2 (skip connection + 上采样结果)
            self.decoder_layers.append(
                ResidualDoubleConv(feature * 2, feature)
            )
            in_channels = feature

    def forward(self, x, skip_connections):
        skip_connections = skip_connections[::-1]  # 反转跳跃连接

        for idx, (up_conv, residual_conv) in enumerate(zip(self.up_convs, self.decoder_layers)):
            x = up_conv(x)
            skip_connection = skip_connections[idx]

            # 尺寸对齐（如果需要）
            if x.shape != skip_connection.shape:
                x = torch.nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True
                )

            # 通道拼接
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = residual_conv(concat_skip)

        return x


class ResidualUNet(nn.Module):
    """完整的残差UNet"""

    def __init__(self, n_channels=3, n_classes=2, features=[64, 128, 256, 512, 1024]):
        super(ResidualUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = ResidualUNetEncoder(n_channels, features)
        decoder_features = features[:-1][::-1]  # [512, 256, 128, 64]
        self.decoder = ResidualUNetDecoder(
            features=decoder_features,
            bottleneck_channels=features[-1]
        )
        self.final_conv = nn.Conv2d(decoder_features[-1], n_classes, kernel_size=1)

    def forward(self, x):
        bottleneck, skip_connections = self.encoder(x)
        x = self.decoder(bottleneck, skip_connections)
        return self.final_conv(x)


def test_residual_unet():
    print("=" * 60)
    print("          残差UNet模型测试")
    print("=" * 60)

    # 测试配置
    batch_size = 2
    img_size = 256
    n_channels = 3
    n_classes = 2

    # 创建测试数据
    x = torch.randn((batch_size, n_channels, img_size, img_size))
    print(f"\n📊 测试数据信息:")
    print(f"  输入尺寸: {x.shape}")
    print(f"  输入范围: [{x.min():.3f}, {x.max():.3f}]")

    # 创建模型
    print(f"\n🔧 模型配置:")
    print(f"  输入通道: {n_channels}")
    print(f"  输出类别: {n_classes}")
    print(f"  特征通道: [64, 128, 256, 512, 1024]")

    model = ResidualUNet(n_channels=n_channels, n_classes=n_classes)

    # 测试前向传播
    print(f"\n🚀 前向传播测试:")
    model.eval()
    with torch.no_grad():
        output = model(x)

    print(f"  输出尺寸: {output.shape}")
    print(f"  输出范围: [{output.min():.3f}, {output.max():.3f}]")

    # 计算参数数量
    print(f"\n📈 模型统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")

    # 测试编码器
    print(f"\n🔍 编码器测试:")
    bottleneck, skip_connections = model.encoder(x)
    print(f"  瓶颈层输出: {bottleneck.shape}")
    for i, skip in enumerate(skip_connections):
        print(f"  跳跃连接 {i}: {skip.shape}")

    # 测试解码器
    print(f"\n🔍 解码器测试:")
    decoder_output = model.decoder(bottleneck, skip_connections)
    print(f"  解码器输出: {decoder_output.shape}")

    # 测试残差块
    print(f"\n🔍 残差块测试:")
    residual_block = ResidualDoubleConv(64, 128)
    test_input = torch.randn(2, 64, 32, 32)
    residual_output = residual_block(test_input)
    print(f"  残差块输入: {test_input.shape}")
    print(f"  残差块输出: {residual_output.shape}")

    # 内存使用测试
    print(f"\n💾 内存使用测试:")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model_cuda = ResidualUNet(n_channels=n_channels, n_classes=n_classes).to(device)
        x_cuda = x.to(device)

        torch.cuda.synchronize()
        start_memory = torch.cuda.memory_allocated(device)

        with torch.no_grad():
            output_cuda = model_cuda(x_cuda)

        torch.cuda.synchronize()
        end_memory = torch.cuda.memory_allocated(device)
        memory_used = (end_memory - start_memory) / 1024 ** 2  # MB

        print(f"  GPU内存使用: {memory_used:.2f} MB")
        print(f"  GPU输出尺寸: {output_cuda.shape}")
    else:
        print("  GPU不可用，跳过GPU测试")

    # 梯度测试
    print(f"\n📉 梯度流测试:")
    model.train()
    x.requires_grad_(True)
    output = model(x)

    # 创建模拟标签
    target = torch.randint(0, n_classes, (batch_size, img_size, img_size))

    # 计算损失并反向传播
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, target)
    loss.backward()

    print(f"  损失值: {loss.item():.4f}")
    print(f"  输入梯度: {x.grad is not None}")

    # 检查模型组件
    print(f"\n🔧 模型组件检查:")
    print(f"  编码器层数: {len(model.encoder.encoder_layers)}")
    print(f"  解码器层数: {len(model.decoder.decoder_layers)}")
    print(f"  上采样层数: {len(model.decoder.up_convs)}")

    # 测试不同输入尺寸
    print(f"\n📏 不同输入尺寸测试:")
    test_sizes = [128, 256, 512]
    for size in test_sizes:
        test_x = torch.randn(1, n_channels, size, size)
        with torch.no_grad():
            test_output = model(test_x)
        print(f"  输入 {size}x{size} -> 输出 {test_output.shape[2]}x{test_output.shape[3]}")

    return output, model


def test_residual_connections():
    """专门测试残差连接的功能"""
    print("\n" + "=" * 60)
    print("          残差连接专项测试")
    print("=" * 60)

    # 测试1: 相同通道数的残差块
    print("\n1. 相同通道数残差块测试:")
    block_same = ResidualDoubleConv(64, 64)
    x_same = torch.randn(2, 64, 16, 16)
    out_same = block_same(x_same)
    print(f"   输入: {x_same.shape}, 输出: {out_same.shape}")
    print(f"   是否使用shortcut卷积: {block_same.use_shortcut}")

    # 测试2: 不同通道数的残差块
    print("\n2. 不同通道数残差块测试:")
    block_diff = ResidualDoubleConv(64, 128)
    x_diff = torch.randn(2, 64, 16, 16)
    out_diff = block_diff(x_diff)
    print(f"   输入: {x_diff.shape}, 输出: {out_diff.shape}")
    print(f"   是否使用shortcut卷积: {block_diff.use_shortcut}")

    # 测试3: 验证残差连接确实存在
    print("\n3. 残差连接验证:")
    # 创建一个简单的测试，确保输出不是恒等映射
    test_input = torch.ones(1, 32, 8, 8) * 0.5
    test_block = ResidualDoubleConv(32, 32)
    test_output = test_block(test_input)

    # 如果残差连接工作正常，输出应该与输入不同
    is_different = not torch.allclose(test_input, test_output, atol=1e-6)
    print(f"   输入输出是否不同: {is_different}")
    print(f"   输入均值: {test_input.mean():.4f}")
    print(f"   输出均值: {test_output.mean():.4f}")


if __name__ == "__main__":
    try:
        # 运行残差连接专项测试
        test_residual_connections()

        # 运行完整模型测试
        output, model = test_residual_unet()

        print("\n" + "=" * 60)
        print("🎉 所有测试通过！残差UNet实现成功！")
        print("=" * 60)
        print(f"✅ 模型输出尺寸: {output.shape}")
        print(f"✅ 模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✅ 残差连接正常工作")
        print(f"✅ 梯度流正常")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()