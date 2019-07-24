from torchvision import transforms, datasets, models

def get_model(model):
    if model == 'alexnet': 
        features = 4096
        return models.alexnet(pretrained = True), features
    elif model == 'densenet121': 
        features = 1024
        return models.densenet121(pretrained = True), features
    elif model == 'densenet161': 
        features = 2208
        return models.densenet161(pretrained = True), features
    elif model == 'densenet169': 
        features = 1664
        return models.densenet169(pretrained = True), features
    elif model == 'densenet201': 
        features = 1920
        return models.densenet201(pretrained = True), features
    elif model == 'inception_v3':
        features = 2048
        return models.inception_v3(pretrained = True), features
    elif model == 'resnet101': 
        features = 2048
        return models.resnet101(pretrained = True), features
    elif model == 'resnet152': 
        features = 2048
        return models.resnet152(pretrained = True), features
    elif model == 'resnet18': 
        features = 512
        return models.resnet18(pretrained = True), features
    elif model == 'resnet34': 
        features = 512
        return models.resnet34(pretrained = True), features
    elif model == 'resnet50': 
        features = 2048
        return models.resnet50(pretrained = True), features
    elif model == 'squeezenet1_0':
        features = 512
        return models.squeezenet1_0(num_classes = 102), features
    elif model == 'squeezenet1_1':
        features = 512
        return models.squeezenet1_1(num_classes = 102), features
    elif model == 'vgg11':
        features = 4096
        return models.vgg11(pretrained = True), features
    elif model == 'vgg11_bn':
        features = 4096
        return models.vgg11_bn(pretrained = True), features
    elif model == 'vgg13':
        features = 4096
        return models.vgg13(pretrained = True), features
    elif model == 'vgg13_bn':
        features = 4096
        return models.vgg13_bn(pretrained = True), features
    elif model == 'vgg16':
        features = 4096
        return models.vgg16(pretrained = True), features
    elif model =='vgg19':
        features = 4096
        return models.vgg19(pretrained = True), features
    elif model =='vgg19_bn':
        features = 4096
        return models.vgg19_bn(pretrained = True), features
    elif model =='vgg16_bn':
        features = 4096
        return models.vgg16_bn(pretrained = True), features