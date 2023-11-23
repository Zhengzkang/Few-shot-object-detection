import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeNet(nn.Module):
    def __init__(self, num_classes, feature_dim, backbone):
        super(PrototypeNet, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.backbone = backbone
        self.w0 = nn.Parameter(torch.randn(feature_dim, feature_dim), requires_grad=True)
        self.w1 = nn.Parameter(torch.randn(feature_dim, feature_dim), requires_grad=True)

    def forward(self, images, boxes):
        # 提取图像特征
        features = self.backbone(images)
        # 根据边界框提取对象特征
        object_features = []
        for i in range(len(boxes)):
            box = boxes[i]
            x1, y1, x2, y2 = box
            feature = features[i, :, y1:y2, x1:x2]
            object_features.append(feature)
        object_features = torch.stack(object_features, dim=0)
        # 计算类别原型
        prototypes = []
        for i in range(self.num_classes):
            class_features = object_features[object_features[:, 0] == i]
            prototype = torch.mean(class_features, dim=0)
            prototypes.append(prototype)
        prototypes = torch.stack(prototypes, dim=0)
        # 计算类别相似度和邻接矩阵
        similarity = torch.matmul(prototypes, self.w0)
        similarity = torch.matmul(similarity, prototypes.t())
        attention = F.softmax(similarity, dim=-1)
        # 传播类别关系信息和学习代表性类别知识
        representative_prototypes = torch.matmul(attention, prototypes)
        representative_prototypes = torch.matmul(representative_prototypes, self.w1)
        representative_prototypes = F.leaky_relu(representative_prototypes)
        return representative_prototypes