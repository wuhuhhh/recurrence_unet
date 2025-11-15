# scripts/check_model.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from models import UNet

def check_model_performance(model_path, device='cpu'):
    """检查模型性能"""
    print("🧪 检查模型性能...")
    
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 加载模型
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 创建测试输入
    print("\n1. 测试随机输入:")
    random_input = torch.randn(1, 3, 512, 512).to(device)
    with torch.no_grad():
        random_output = model(random_input)
        random_pred = torch.sigmoid(random_output)
    
    print(f"   随机输入输出范围: [{random_pred.min():.3f}, {random_pred.max():.3f}]")
    print(f"   均值: {random_pred.mean():.3f}")
    
    # 测试全零输入
    print("\n2. 测试全零输入:")
    zeros_input = torch.zeros(1, 3, 512, 512).to(device)
    with torch.no_grad():
        zeros_output = model(zeros_input)
        zeros_pred = torch.sigmoid(zeros_output)
    
    print(f"   全零输入输出范围: [{zeros_pred.min():.3f}, {zeros_pred.max():.3f}]")
    print(f"   均值: {zeros_pred.mean():.3f}")
    
    # 测试全一输入
    print("\n3. 测试全一输入:")
    ones_input = torch.ones(1, 3, 512, 512).to(device)
    with torch.no_grad():
        ones_output = model(ones_input)
        ones_pred = torch.sigmoid(ones_output)
    
    print(f"   全一输入输出范围: [{ones_pred.min():.3f}, {ones_pred.max():.3f}]")
    print(f"   均值: {ones_pred.mean():.3f}")
    
    # 分析模型权重
    print("\n4. 模型权重分析:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"   {name:30} 形状: {tuple(param.shape)} 范围: [{param.data.min():.3f}, {param.data.max():.3f}]")
    
    print(f"\n📊 总参数量: {total_params:,}")

if __name__ == "__main__":
    model_path = "best_landslide_unet.pth"
    check_model_performance(model_path)