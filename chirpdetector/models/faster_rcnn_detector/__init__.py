"""Faster R-CNN implementation from torchvision."""

from .loaders import load_finetuned_faster_rcnn, load_pretrained_faster_rcnn

__all__ = ["load_pretrained_faster_rcnn", "load_finetuned_faster_rcnn"]
