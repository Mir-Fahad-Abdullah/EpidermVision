import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, base_model, num_classes, num_ftrs = 1280):
        super(CustomModel, self).__init__()
        self.base_model_features = base_model.features
        self.base_model_avg_pool = base_model.avgpool
        
        # for param in self.base_model_features.parameters():
        #     param.requires_grad = False

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(num_ftrs, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_model_features(x)
        x = self.base_model_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
















