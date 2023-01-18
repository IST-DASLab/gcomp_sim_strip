import torch


class Compressor:
    def __init__(self, enable_error_correction=False, warmup_steps=None):
        self.enable_error_correction = enable_error_correction
        self.states = {}
        self.warmup_steps = warmup_steps

    def compress(self, grad, state):
        if "step" not in state:
            state["step"] = 0
        step = state["step"]
        if self.warmup_steps and step < self.warmup_steps:
            return grad, None
        grad_ = grad

        if self.enable_error_correction:
            if "error_correction" not in state:
                state["error_correction"] = torch.zeros_like(grad_)
            e_c = state["error_correction"]
            # update error correction before subtraction
            e_c.add_(grad_)
            grad_.copy_(e_c)

        self.compress_(grad_, state)

        if self.enable_error_correction:
            e_c.sub_(grad_)

    # Compression has to be done in-place
    def compress_(self, grad, state):
        raise NotImplementedError


class NoneCompressor(Compressor):
    def __init__(self):
        super().__init__()

    def compress_(self, grad, state):
        pass


class SanityCompressor(Compressor):
    def __init__(self):
        super().__init__()

    def compress_(self, grad, state):
        grad.zero_()
