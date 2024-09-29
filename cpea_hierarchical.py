import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchicalCPEA(nn.Module):
    def __init__(self, base_model, base_model_output_dim, num_classes_level_1, num_classes_level_2):
        super(HierarchicalCPEA, self).__init__()
        self.base_model = base_model
        self.head_level_1 = nn.Linear(base_model_output_dim, num_classes_level_1)
        self.head_level_2 = nn.Linear(base_model_output_dim, num_classes_level_2)

    def forward(self, x):
        features = self.base_model(x)
        out_level_1 = self.head_level_1(features)
        out_level_2 = self.head_level_2(features)
        return out_level_1, out_level_2
