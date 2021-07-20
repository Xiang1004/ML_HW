import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    # 先定義Layer的參數
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO
        # Convolution 1 , input_shape=(1,28,28)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=1,padding=0)  # output_shape=(64,24,24)
        self.relu1 = nn.ReLU()  # Activation
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # output_shape=(64,12,12)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1,padding=0)  # output_shape=(128,8,8)
        self.relu2 = nn.ReLU()  # Activation
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # output_shape=(128,4,4)
        # Convolution 2
        self.cnn3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, stride=1,padding=0)  # output_shape=(128,8,8)
        self.relu3 = nn.ReLU()  # Activation
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)  # output_shape=(128,2,2)
        self.fc1 = nn.Linear(512 * 2 * 2, 10)

    # 再定義執行順序
    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        out = out.view(out.size(0), -1)  # flatten the tensor (樣本數量, -1自動確定維度)
        out = self.fc1(out)
        return out

    def name(self):
        return "ConvNet"

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # TODO
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1,padding=0)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1,padding=0)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out

    def name(self):
        return "MyNet"