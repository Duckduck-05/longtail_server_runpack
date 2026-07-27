"""No-op compatibility shim required by the public CORAL main.py.

The pinned CORAL commit imports LossTracker but does not ship it.  The class is
only used for optional local logging, so this shim intentionally has no effect
on model, optimizer, data, loss, EMA, sampling, or evaluation.
"""


class LossTracker:
    def __init__(self, logdir, save_interval=10):
        self.logdir = logdir
        self.save_interval = save_interval

    def update(self, *args, **kwargs):
        return None

    def save(self, *args, **kwargs):
        return None
