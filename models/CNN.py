import torch.nn as nn
import torch

class ConvolutionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0):
        super(ConvolutionBlock, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convolution = nn.Conv2d(self.in_dim , self.out_dim , kernel_size, stride, padding)
    
    def forward(self, x):
        x = self.convolution(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CNN(nn.Module):
    def __init__(self, nblocks, in_dim, out_dim, kernel_size, stride, padding, layers_out, input_image_size=(224, 224)):
        super(CNN, self).__init__()
        layers = []
        current_in_dim = in_dim
        for layer_out in layers_out:
            layers.append(ConvolutionBlock(current_in_dim, layer_out, kernel_size, stride, padding))
            current_in_dim = layer_out
        self.convolutional_layers = nn.Sequential(*layers)

        # Compute flattened size dynamically by passing a dummy input through the conv layers
        with torch.no_grad():
            dummy = torch.zeros(1, in_dim, *input_image_size)
            out = self.convolutional_layers(dummy)
            flattened_size = out.view(1, -1).shape[1]

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(flattened_size, out_dim)

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x