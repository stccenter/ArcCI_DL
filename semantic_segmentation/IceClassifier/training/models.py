from torchvision import models

backbone_dict = {
    'fcn_resnet50': models.segmentation.fcn_resnet50,
    'fcn_resnet101': models.segmentation.fcn_resnet101,
    'deeplabv3_resnet50': models.segmentation.deeplabv3_resnet50,
    'deeplabv3_resnet101': models.segmentation.deeplabv3_resnet101,
    'deeplabv3_mobilenet_v3_large': models.segmentation.deeplabv3_mobilenet_v3_large,
    'lraspp_mobilenet_v3_large': models.segmentation.lraspp_mobilenet_v3_large,
}


def fetch_backbone_model(model_name, num_classes):
    model = backbone_dict[model_name]
    return model(
        pretrained=False,
        progress=True,
        num_classes=num_classes
    )
