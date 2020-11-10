import torch
import torch.nn as nn

cfgs = {
    'A': [1, 1, 2, 2, 2],   # vgg11
    'B': [2, 2, 2, 2, 2],   # vgg13
    'D': [2, 2, 3, 3, 3],   # vgg16
    'E': [2, 2, 4, 4, 4],   # vgg19
}
"""
cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
"""

class VGG(nn.Module):
    def __init__(self, num_layers=None, batch_norm=True, init_weights=True, num_classes=100):
        super(VGG, self).__init__()
        self.conv1 = self._make_block(3, 64, num_layer=num_layers[0], batch_norm=batch_norm)
        self.conv2 = self._make_block(64, 128, num_layer=num_layers[1], batch_norm=batch_norm)
        self.conv3 = self._make_block(128, 256, num_layer=num_layers[2], batch_norm=batch_norm)
        self.conv4 = self._make_block(256, 512, num_layer=num_layers[3], batch_norm=batch_norm)
        self.conv5 = self._make_block(512, 512, num_layer=num_layers[4], batch_norm=batch_norm)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes)
        )
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
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

    def _make_block(self, in_channels, out_channels, num_layer, batch_norm=False):
        """
        Each block builds layers containing repeated conv, bn and relu layers with num of layers in 'num_layer'
        And append a maxpool layers at the end of the block
        """
        block = []

        for _ in range(num_layer):
            block += [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
            
            if batch_norm:
                block += [nn.BatchNorm2d(out_channels)]
            
            block += [nn.ReLU(inplace=True)]
            in_channels = out_channels
        
        block += [nn.MaxPool2d(kernel_size=2, stride=2)]
        return nn.Sequential(*block)

def _vgg(arch, cfg, batch_norm, pretrained, **kwargs):
    return VGG(num_layers=cfgs[cfg], batch_norm=batch_norm, **kwargs)

def vgg16(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16', 'D', False, pretrained, **kwargs)

def vgg16_bn(pretrained=False, progress=True, **kwargs):
    return _vgg('vgg16_bn', 'D', True, pretrained, **kwargs)