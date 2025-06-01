import torch.nn as nn
import torch
from torch.autograd import Function

class MixStyle(nn.Module):
    """MixStyle module for domain generalization (Cross-Domain version)."""
    def __init__(self, p=0.5, alpha=0.1):
        """
        Args:
            p (float): Probability of applying MixStyle.
            alpha (float): Beta distribution parameter for sampling mix ratios.
        """
        super(MixStyle, self).__init__()
        self.p = p
        self.alpha = alpha
        self._activated = True

    def forward(self, x, domain_labels):
        if not self.training or not self._activated or torch.rand(1).item() > self.p:
            return x

        # Compute mean and standard deviation
        batch_size, channels, _ = x.size()
        mu = x.mean(dim=2, keepdim=True)
        sigma = x.std(dim=2, keepdim=True)

        # Normalize the input
        x_normed = (x - mu) / (sigma + 1e-6)

        # Shuffle statistics across different domains
        perm = torch.arange(batch_size)
        for i in range(batch_size):
            # Find a different domain for shuffling
            different_domain = (domain_labels != domain_labels[i])
            if different_domain.any():
                perm[i] = different_domain.nonzero(as_tuple=True)[0].tolist()[0]

        mu_mix = mu[perm]
        sigma_mix = sigma[perm]

        # Sample mixing coefficients
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size, 1, 1)).to(x.device)

        # Mix statistics and reconstruct features
        mu_mixed = lam * mu + (1 - lam) * mu_mix
        sigma_mixed = lam * sigma + (1 - lam) * sigma_mix
        x_mixed = x_normed * sigma_mixed + mu_mixed

        return x_mixed

    def activate(self, activate=True):
        """Activate or deactivate MixStyle."""
        self._activated = activate

    def deactivate(self):
        """Shortcut to deactivate MixStyle."""
        self.activate(False)

def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride,
                     padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class SELayer(nn.Module):
    """Squeeze-and-Excitation block."""
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class BasicBlock(nn.Module):
    expansion = 1  # Add this attribute

    def __init__(self, inplanes, planes, stride=1, downsample=None, mixstyle=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes)
        self.downsample = downsample
        self.stride = stride
        self.mixstyle = mixstyle

    def forward(self, x, domain_labels=None):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Apply MixStyle if defined
        if self.mixstyle is not None:
            out = self.mixstyle(out, domain_labels)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    """ResNet architecture with MixStyle."""
    def __init__(self, block, layers, in_channel=1, out_channel=10, mixstyle_p=0.5, mixstyle_alpha=0.1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], mixstyle=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, mixstyle=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, mixstyle=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, mixstyle=False)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(3, 10)  # Age and gender layer
        self.fc = nn.Linear(512 * block.expansion + 10, out_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, mixstyle=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.mixstyle if mixstyle else None))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, mixstyle=self.mixstyle if mixstyle else None))

        # Use nn.ModuleList to retain the domain_labels input across blocks
        return nn.ModuleList(layers)

    def forward_layer(self, x, layers, domain_labels=None):
        for layer in layers:
            x = layer(x, domain_labels)  # Forward with domain_labels
        return x

    def forward(self, x, ag, domain_labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.forward_layer(x, self.layer1, domain_labels)
        x = self.forward_layer(x, self.layer2, domain_labels)
        x = self.forward_layer(x, self.layer3)
        x = self.forward_layer(x, self.layer4)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        ag = self.fc1(ag)
        x = torch.cat((ag, x), dim=1)
        x = self.fc(x)

        return x

def resnet18(**kwargs):
    """Constructing a ResNet-18 model with MixStyle."""
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model