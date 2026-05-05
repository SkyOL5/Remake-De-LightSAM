import torch
import torch.nn as nn


class Modality_Inspector(nn.Module):
    
    def __init__(self, num_classes=6):
        
        super(Modality_Inspector, self).__init__()
        
        # 输入形状: (batch_size, 256, 64, 64)
        self.feature_extractor = nn.Sequential(
            # 第一卷积层
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),  # 输出: (128, 16, 16)
            
            # 第二卷积层
            nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),  # 输出: (32, 4, 4)
            
            # # 第三卷积层
            # nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=4),  # 输出: (32, 8, 8)
        )
        
        # 计算全连接层输入特征数
        self.flatten = nn.Flatten()
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        features = self.feature_extractor(x)  # 提取特征
        flattened = self.flatten(features)    # 展平
        output = self.classifier(flattened)   # 分类
        return output