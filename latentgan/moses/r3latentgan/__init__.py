from .config import get_parser as R3latentGAN_parser
from .model import R3LatentGAN
from .trainer import R3LatentGANTrainer

__all__ = ['R3latentGAN_parser', 'R3LatentGAN', 'R3LatentGANTrainer']
