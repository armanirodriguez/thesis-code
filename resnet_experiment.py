import torch
from resnet50 import ResNet50

CLASSIFIER_IMAGENET_SAVE_PATH = './checkpoints/resnet_imagenet.pt'

if __name__ == '__main__':
    classifier = torch.load(CLASSIFIER_IMAGENET_SAVE_PATH)
    print(classifier)