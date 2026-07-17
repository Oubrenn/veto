"""Backbone编码器"""
from .inception_time import InceptionTime
from .resnet import ResNet1D
from .fcn import FCN

__all__ = ['InceptionTime', 'ResNet1D', 'FCN']
