import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, max_moves):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.lin1 = nn.Linear(8*6*6+max_moves*2*64, 8*6*6)
        self.lin2 = nn.Linear(8*6*6, max_moves)
        self.soft = nn.Softmax(dim=1)

    def forward(self, input, possible_actions):
        x = F.relu(self.bn1(self.conv1(input.float())))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, possible_actions), 1)
        x = F.relu(self.lin1(x))
        out = self.soft(self.lin2(x))
        return out