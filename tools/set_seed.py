import torch
import numpy as np
import random


def set_seed(seed: int) -> None:
    """
    设置随机种子，便于复现训练结果，节省空间存储
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
