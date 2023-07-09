import torch
import torch.nn as nn
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
        self.conv1 = nn.Conv2d(3072, 256, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.backbone.extract_features(x)
        x = torch.cat([x[3], x[4]], dim=1)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x



