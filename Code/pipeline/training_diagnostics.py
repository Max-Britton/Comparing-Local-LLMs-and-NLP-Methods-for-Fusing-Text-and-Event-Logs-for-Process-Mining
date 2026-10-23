from contextlib import contextmanager
import random

import numpy as np


@contextmanager
def isolated_inference(model, torch, device):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    accelerator_state = None
    if device.type == 'cuda':
        accelerator_state = torch.cuda.get_rng_state(device)
    elif device.type == 'mps':
        accelerator_state = torch.mps.get_rng_state()
    was_training = model.training
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        model.train(was_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if device.type == 'cuda':
            torch.cuda.set_rng_state(accelerator_state, device)
        elif device.type == 'mps':
            torch.mps.set_rng_state(accelerator_state)
