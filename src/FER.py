"""Based On: Stoychev, S., Gunes, H. (2023). 
The Effect of Model Compression on Fairness in Facial Expression Recognition. 
In: Rousseau, JJ., Kapralos, B. (eds) 
Pattern Recognition, Computer Vision, and Image Processing. 
ICPR 2022 International Workshops and Challenges. 
ICPR 2022. Lecture Notes in Computer Science, vol 13646. 
Springer, Cham. https://doi.org/10.1007/978-3-031-37745-7_9"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub

class FERModel(nn.Module):
    def __init__(self, input_size=48, input_channels=1, out_classes=8, q=False):
        super(FERModel, self).__init__()

        # 1st Convolution layer
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 2nd Convolution layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        
        # 3rd Convolution layer
        self.conv3 = nn.Conv2d(128, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        
        # 4th Convolution layer
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512 * (input_size // 16) * (input_size // 16), 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        
        self.fc3 = nn.Linear(512, out_classes)

        # Dropout layers
        self.dropout = nn.Dropout(0.25)
        self.q = q
        if q:
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        # Convolution layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout(x)

        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
