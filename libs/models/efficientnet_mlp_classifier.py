import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class SimpleClassifier(nn.Module):
    """
    EfficientNetB5 backbone + 4 MLP layers
    """
    def __init__(self, num_classes, dropout_rate:float=0.3, layers:list[int]=[512, 256, 128], num_layers_to_unfreeze=25):
        super(SimpleClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b7')

        # Freeze all layers except the last `num_layers_to_unfreeze`
        for param in self.model.parameters():
            param.requires_grad = False
        num_layers = len(list(self.model.children()))
        print('Num layers:',num_layers)
        for i, layer in enumerate(self.model.children()):
            if num_layers - i < num_layers_to_unfreeze:
                continue
            for param in layer.parameters():
                param.requires_grad = True

        assert isinstance(layers, list), "Argument 'layers' should be list of integers"
        self.layers = [self.model._fc.in_features]+layers
        
        self.classifier = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(self.layers[j], self.layers[j+1]),
                    nn.BatchNorm1d(self.layers[j+1]),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate) if j < len(self.layers)-2 else nn.Identity()
                )
                for j in range(len(self.layers)-1)
            ],
            nn.Linear(self.layers[-1], num_classes)
        )
        self.model._fc = self.classifier

    def forward(self, x):
        return self.model(x)