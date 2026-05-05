import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalizedBBoxRegressor(nn.Module):
    """
    归一化边界框回归模型
    输入: (batch_size, 64, 64, 256) 的特征图
    输出: 归一化边界框 (x, y, w, h) 范围[0, 1]
    """
    
    def __init__(self, input_channels=256, img_width=1024, img_height=1024):
        super(NormalizedBBoxRegressor, self).__init__()
        
        self.input_channels = input_channels
        self.img_width = img_width
        self.img_height = img_height
        
        # 特征处理器 - 一层卷积
        self.feature_processor = nn.Sequential(
            # 输入: (batch_size, 256, 64, 64)
            nn.Conv2d(input_channels, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
            
            # 可选的第二层
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        )
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 回归头 - 输出归一化坐标
        self.regressor = nn.Sequential(
            nn.Linear(256, 128),  # 输入维度从512改为256
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 4)  # 输出: x, y, w, h (归一化坐标 0-1)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征图, shape: (batch_size, 64, 64, 256) 或 (batch_size, 256, 64, 64)
        
        返回:
            bbox: 归一化边界框, shape: (batch_size, 4)
                  格式: [x_center, y_center, width, height] (归一化坐标 0-1)
        """
        # 确保输入形状正确 (NCHW格式)
        if x.dim() == 4:
            if x.shape[-1] == 256:  # NHWC格式
                x = x.permute(0, 3, 1, 2)  # 转换为NCHW: (batch, 256, 64, 64)
        else:
            raise ValueError(f"输入维度错误: {x.shape}")
        
        # 特征处理
        features = self.feature_processor(x)  # (batch, 256, 16, 16)
        
        # 全局池化
        features = self.global_pool(features)  # (batch, 256, 1, 1)
        
        # 展平
        features = features.view(features.size(0), -1)  # (batch, 256)
        
        # 回归
        bbox = self.regressor(features)  # (batch, 4)
        
        # 使用sigmoid确保输出在0-1之间
        bbox = torch.sigmoid(bbox)
        
        return bbox
    
    def normalize_coords(self, pixel_coords):
        """
        将像素坐标归一化
        
        参数:
            pixel_coords: 像素坐标 [x, y, w, h]
        
        返回:
            normalized_coords: 归一化坐标 [x, y, w, h] 范围[0, 1]
        """
        normalized = pixel_coords.clone()
        if len(normalized.shape) == 1:
            normalized = normalized.unsqueeze(0)
        
        # 归一化
        normalized[:, 0] = pixel_coords[:, 0] / self.img_width   # x
        normalized[:, 1] = pixel_coords[:, 1] / self.img_height  # y
        normalized[:, 2] = pixel_coords[:, 2] / self.img_width   # w
        normalized[:, 3] = pixel_coords[:, 3] / self.img_height  # h
        
        return normalized.squeeze()
    
    def denormalize_coords(self, normalized_coords):
        """
        将归一化坐标转换为像素坐标
        
        参数:
            normalized_coords: 归一化坐标 [x, y, w, h] 范围[0, 1]
        
        返回:
            pixel_coords: 像素坐标
        """
        pixel = normalized_coords.clone()
        if len(pixel.shape) == 1:
            pixel = pixel.unsqueeze(0)
        
        # 反归一化
        pixel[:, 0] = normalized_coords[:, 0] * self.img_width   # x
        pixel[:, 1] = normalized_coords[:, 1] * self.img_height  # y
        pixel[:, 2] = normalized_coords[:, 2] * self.img_width   # w
        pixel[:, 3] = normalized_coords[:, 3] * self.img_height  # h
        
        return pixel.squeeze()

class NormalizedBBoxLoss(nn.Module):
    """
    归一化坐标的边界框损失函数
    """
    
    def __init__(self, lambda_l1=1.0, lambda_iou=3.0):
        super(NormalizedBBoxLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_iou = lambda_iou
        
    def forward(self, pred, target):
        """
        计算损失
        
        参数:
            pred: 预测归一化坐标 [x, y, w, h] (0-1)
            target: 真实归一化坐标 [x, y, w, h] (0-1)
        
        返回:
            loss_dict: 各种损失组成的字典
        """
        # 1. L1损失
        l1_loss = F.l1_loss(pred, target)
        
        # 2. IoU损失
        iou_loss = self.iou_loss(pred, target)
        
        # 4. 中心点约束损失
        center_loss = self.center_constraint_loss(pred)
        
        # 总损失
        total_loss = (
            self.lambda_l1 * l1_loss +
            self.lambda_iou * iou_loss +
            0.1 * center_loss
        )
        
        return {
            'total_loss': total_loss,
            'l1_loss': l1_loss,
            'iou_loss': iou_loss,
            'center_loss': center_loss
        }
    
    def iou_loss(self, pred, target):
        """归一化坐标的IoU损失"""
        # 转换为角点坐标
        pred_corners = self.to_corners(pred)
        target_corners = self.to_corners(target)
        
        # 计算交集
        inter_x1 = torch.max(pred_corners[:, 0], target_corners[:, 0])
        inter_y1 = torch.max(pred_corners[:, 1], target_corners[:, 1])
        inter_x2 = torch.min(pred_corners[:, 2], target_corners[:, 2])
        inter_y2 = torch.min(pred_corners[:, 3], target_corners[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # 计算并集
        pred_area = (pred_corners[:, 2] - pred_corners[:, 0]) * (pred_corners[:, 3] - pred_corners[:, 1])
        target_area = (target_corners[:, 2] - target_corners[:, 0]) * (target_corners[:, 3] - target_corners[:, 1])
        union_area = pred_area + target_area - inter_area + 1e-6
        
        iou = inter_area / union_area
        iou_loss = 1.0 - iou
        
        return iou_loss.mean()
    
    def to_corners(self, boxes):
        """归一化坐标转角点坐标"""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        
        return torch.stack([x1, y1, x2, y2], dim=1)
    
    def center_constraint_loss(self, pred):
        """中心点约束损失"""
        x, y = pred[:, 0], pred[:, 1]
        
        # 鼓励中心点在图像内
        center_loss = torch.relu(-x) + torch.relu(x - 1) + torch.relu(-y) + torch.relu(y - 1)
        
        return center_loss.mean()

class NormalizedDataset(Dataset):
    """
    归一化坐标数据集
    """
    
    def __init__(self, feature_files, box_files, feature_folder, img_width=1024, img_height=1024):
        """
        初始化
        
        参数:
            feature_files: 特征文件列表
            box_files: 边界框文件列表
            feature_folder: 特征文件夹路径
            img_width: 图像宽度
            img_height: 图像高度
        """
        self.feature_files = feature_files
        self.box_files = box_files
        self.feature_folder = feature_folder
        self.img_width = img_width
        self.img_height = img_height
        
        # 预加载边界框
        self.boxes = []
        for box_file in box_files:
            with open(box_file, "r") as f:
                cor = list(map(int, f.readline().strip().split(" ")))
                if len(cor) == 4:  # 确保是x,y,w,h格式
                    self.boxes.append(cor)
                elif len(cor) == 8:  # 如果是x1,y1,x2,y2,x3,y3,x4,y4格式
                    # 转换为x,y,w,h
                    xs = [cor[0], cor[2], cor[4], cor[6]]
                    ys = [cor[1], cor[3], cor[5], cor[7]]
                    x_center = sum(xs) / 4
                    y_center = sum(ys) / 4
                    width = max(xs) - min(xs)
                    height = max(ys) - min(ys)
                    self.boxes.append([x_center, y_center, width, height])
        
        # 确保文件数量匹配
        assert len(self.feature_files) == len(self.boxes), \
            f"特征文件数 ({len(self.feature_files)}) 不等于边界框数 ({len(self.boxes)})"
    
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        # 加载特征
        feature_path = os.path.join(self.feature_folder, self.feature_files[idx])
        features = torch.load(feature_path)
        
        # 确保特征形状
        if features.dim() == 3:  # (64, 64, 256)
            pass  # 保持原状
        elif features.dim() == 4:  # (1, 256, 64, 64)或其他
            if features.shape[0] == 1:
                features = features.squeeze(0)
            if features.shape[0] == 256:  # (256, 64, 64)
                features = features.permute(1, 2, 0)  # 转换为(64, 64, 256)
        
        # 获取边界框并归一化
        pixel_bbox = torch.tensor(self.boxes[idx], dtype=torch.float32)
        
        # 归一化
        normalized_bbox = torch.zeros_like(pixel_bbox)
        normalized_bbox[0] = pixel_bbox[0] / self.img_width   # x
        normalized_bbox[1] = pixel_bbox[1] / self.img_height  # y
        normalized_bbox[2] = pixel_bbox[2] / self.img_width   # w
        normalized_bbox[3] = pixel_bbox[3] / self.img_height  # h
        
        return features.float(), normalized_bbox