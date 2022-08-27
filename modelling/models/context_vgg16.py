from typing import Any, List, Union
import torch
import torch.nn as nn
import torchvision
from torchvision.models import vgg

cfg: List[Union[str, int]] = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
keys_to_replace: List[str] = ['classifier.0.weight', 'classifier.0.bias', 'classifier.3.weight', 'classifier.3.bias',
                              'classifier.6.weight', 'classifier.6.bias']
LEN_CONTEXT_VECTOR = 15


class ContextVGG(torchvision.models.VGG):
    """
    VGG model that combines a context vector
    """
    def __init__(
            self, features: nn.Module,
            num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5) -> None:
        super(ContextVGG, self).__init__(features, num_classes, init_weights)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7 + LEN_CONTEXT_VECTOR, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_image: torch.Tensor, x_context: torch.Tensor) -> torch.Tensor:
        x_image = self.features(x_image)
        x_image = self.avgpool(x_image)
        x_image = torch.flatten(x_image, 1)
        x = torch.cat([x_image, x_context], 1)
        x = self.classifier(x)
        return x


def context_vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ContextVGG:
    """
    Returns the ContextVGG model
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = ContextVGG(vgg.make_layers(cfg, batch_norm=True), **kwargs)
    if pretrained:
        state_dict = vgg.load_state_dict_from_url(vgg.model_urls["vgg16_bn"], progress=progress)
        for key in keys_to_replace:
            state_dict.pop(key, None)
        model.load_state_dict(state_dict, strict=False)
    return model

