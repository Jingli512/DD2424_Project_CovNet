import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, batch_norm = False, dropout = None):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, kernel_size=3, padding=1)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.batch_norm = batch_norm
        self.dropout = dropout

        if(self.batch_norm):
            self.bn1 = nn.BatchNorm2d(hidden_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        if(self.dropout is not None):
            self.dropout = nn.Dropout(p = dropout)

    def forward(self, x):

        x = self.conv1(x)
        if(self.batch_norm):
            x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        if(self.batch_norm):
            x = self.bn2(x)
        x = self.relu(x)

        x = self.max_pool2d(x)

        if(self.dropout is not None):
            x = self.dropout(x)

        return x


class VGG_3block(nn.Module):
    def __init__(self, dims = [3, 32, 64, 128], batch_norm = False, dropout = None):
        super(VGG_3block, self).__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        blocks = []
        for i in range(len(dims) - 1):
            if(self.dropout is not None):
                blocks.append(VGGBlock(input_dim = dims[i], hidden_dim = dims[i + 1], output_dim = dims[i + 1], batch_norm = self.batch_norm, dropout = self.dropout[i]))
            else:
                blocks.append(VGGBlock(input_dim = dims[i], hidden_dim = dims[i + 1], output_dim = dims[i + 1], batch_norm = self.batch_norm))
        self.blocks = nn.Sequential(*blocks)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.relu = nn.ReLU()
        if(self.dropout is not None):
            self.dropout = nn.Dropout(p = self.dropout[-1])
        self.fc2 = nn.Linear(128, 10)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.blocks(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        if(self.dropout is not None):
            x = self.dropout(x)
        x = self.fc2(x)
        return x

if(__name__ == '__main__'):
    model = VGG_3block()
    dummy = torch.randn((5, 3, 32, 32))
    out = model(dummy)
    print(out.shape)
