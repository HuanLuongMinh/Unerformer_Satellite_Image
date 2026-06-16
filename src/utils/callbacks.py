class EarlyStopping:
    """Stops training when mIoU does not improve for `patience` evaluations."""

    def __init__(self, patience: int = 2, delta: float = 1e-4):
        self.patience  = patience
        self.delta     = delta
        self.best      = -float('inf')
        self.counter   = 0
        self.triggered = False

    def step(self, miou: float) -> bool:
        """Call after each validation. Returns True when training should stop."""
        if miou > self.best + self.delta:
            self.best    = miou
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered
