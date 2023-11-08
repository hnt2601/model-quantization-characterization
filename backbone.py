import torch.nn as nn
import torch
import timm


class EfficientnetV2(nn.Module):
    def __init__(self, class_num=10, pretrained=False):
        super(EfficientnetV2, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained
        self.name = "efficientnetv2_rw_t"

        # load model #####################################
        self.model = timm.create_model(self.name, pretrained=self.pretrained)

        # set last layer ################################
        if self.model.classifier.out_features != self.class_num:
            self.model.classifier = nn.Linear(1024, self.class_num)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out


class MobileNetV2(nn.Module):
    def __init__(self, class_num=10, pretrained=False):
        super(MobileNetV2, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained
        self.name = "mobilenetv2"

        # load model #####################################
        self.model = torch.hub.load(
            "pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=self.pretrained
        )

        # set last layer ################################
        if self.model.classifier[1].out_features != self.class_num:
            self.model.classifier[1] = nn.Linear(1280, self.class_num)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet18(nn.Module):
    def __init__(self, class_num=10, pretrained=True):
        super(ResNet18, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained
        self.name = "resnet18"

        # load model #####################################
        self.model = timm.create_model(self.name, pretrained=self.pretrained)
        # self.model = timm.create_model('resnet18', pretrained=False)
        # set last layer ################################
        if self.model.fc.out_features != self.class_num:
            self.model.fc = nn.Linear(512, self.class_num)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        # print(x.shape)
        out = self.model(x)
        return out


class ResNet50(nn.Module):
    def __init__(self, class_num=10, pretrained=False):
        super(ResNet50, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained
        self.name = "resnet50"

        # load model #####################################
        self.model = timm.create_model(self.name, pretrained=self.pretrained)

        # set last layer ################################
        if self.model.fc.out_features != self.class_num:
            self.model.fc = nn.Linear(2048, self.class_num)

    def forward(self, x):
        out = self.model(x)
        return out


class ResNet101(nn.Module):
    def __init__(self, class_num=10, pretrained=False):
        super(ResNet101, self).__init__()
        self.class_num = class_num
        self.pretrained = pretrained
        self.name = "resnet101"

        # load model #####################################
        self.model = timm.create_model(self.name, pretrained=self.pretrained)

        # set last layer ################################
        if self.model.fc.out_features != self.class_num:
            self.model.fc = nn.Linear(2048, self.class_num)

        # # Normal initialization #########################
        # for params, a in self.model.named_parameters():
        #     torch.nn.init.normal(a.data)

    def forward(self, x):
        out = self.model(x)
        return out
