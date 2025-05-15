import logging

import torch


def get_gpu_info() -> tuple[int, bool]:
    """Helper function to see if we are training on GPU, XPU or CPU"""
    try:
        # get nr of XPUs if training on Intel GPUs
        n_gpus = torch.xpu.device_count()
        logging.info(f"Nr of XPU devices available: {n_gpus}")
        if n_gpus > 0:
            return n_gpus, True
        n_gpus = torch.cuda.device_count()
        logging.info(f"Nr of CUDA devices available: {n_gpus}")
        if n_gpus > 0:
            return n_gpus, False
    except AttributeError:
        n_gpus = torch.cuda.device_count()
        logging.info(f"Nr of GPU devices available: {n_gpus}")
        return n_gpus, False
