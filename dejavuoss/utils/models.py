import torch.nn as nn

class SSLNetwork(nn.Module):
    def __init__(
        self, arch, loss
    ):
        super().__init__()
        if "resnet" in arch:
            import torchvision.models.resnet as resnet
            self.net = resnet.__dict__[arch]()
        else:
            print("Arch not found")
            exit(0)

        self.num_features = self.net.fc.in_features
        self.loss = loss

    def forward(self, inputs):
        return self.net(inputs)