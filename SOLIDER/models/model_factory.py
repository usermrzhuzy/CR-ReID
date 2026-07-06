from torch import nn

from SOLIDER.models.registry import BACKBONE
from SOLIDER.models.registry import CLASSIFIER
from SOLIDER.models.registry import LOSSES


model_dict = {
    'swin_t': 768,
    'swin_s': 768,
    'swin_b': 1024,

}


class LinearClassifier(nn.Module):
    def __init__(self, nattr, c_in, bn=True, pool='avg', scale=1):
        super(LinearClassifier, self).__init__()
        self.nattr = nattr
        self.c_in = c_in
        self.bn = bn
        self.pool = pool
        self.scale = scale
        # 选择池化层
        if self.pool == 'avg':
            self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        elif self.pool == 'max':
            self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        else:
            raise ValueError(f"Unsupported pooling type: {self.pool}")

        # 选择 BatchNorm
        self.bn_layer = nn.BatchNorm1d(c_in) if self.bn else nn.Identity()

        # 线性分类器
        self.fc = nn.Linear(c_in, nattr)

    def forward(self, x):
        # 进行池化
        x = self.pooling(x).view(x.shape[0], -1)

        # 归一化（如果使用 BatchNorm）
        x = self.bn_layer(x)

        # 线性分类
        x = self.fc(x) * self.scale  # 乘以 scale 进行缩放
        return x

CLASSIFIER.register("linear", LinearClassifier)


def build_backbone(key, multi_scale=False):


    model = 'swin_b'
    output_d = 1024

    return model, output_d


def build_classifier(key):

    return CLASSIFIER[key]


def build_loss(key):

    return LOSSES[key]

