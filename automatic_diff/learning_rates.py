import numpy as np


class LearningRate:

    @classmethod
    def create(cls, *arg):
        if isinstance(arg[0], cls):
            return arg[0]
        else:
            return cls(*arg)

    def __init__(self, lr=0.1):
        self._init_lr = lr
        self.lr = self._init_lr

    @property
    def lr(self):
        return self.__lr

    @lr.setter
    def lr(self, lr):
        self.__lr = lr

    def update(self, grad_descent):
        return np.array(grad_descent.dy) * self.lr


class TimeDecayLearningRate(LearningRate):

    def __init__(self, decay_rate=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.decay_rate = decay_rate

    def update(self, grad_descent):
        self.lr = self._init_lr / (1 + self.decay_rate * grad_descent.num_iter)
        return np.array(grad_descent.dy) * self.lr


class GradDecayLearningRate(LearningRate):

    def __init__(self, patience=5, **kwargs):
        super().__init__(**kwargs)
        self.patience = patience
        self.this_dy = None
        self.prev_dy = None
        self.this_x = None
        self.prev_x = None
        self.prev_lr = []

    def update(self, grad_descent):
        if grad_descent.num_iter > 0:
            self.prev_dy = self.this_dy.copy()
            self.prev_x = self.this_x.copy()
        self.this_dy = np.array(grad_descent.dy)
        self.this_x = np.array(grad_descent.x)
        if grad_descent.num_iter > 0:
            delta_dy = self.this_dy - self.prev_dy
            delta_x = self.this_x - self.prev_x
            self.lr = np.dot(delta_x, delta_dy) / np.dot(delta_dy, delta_dy)
            self.lr = min(abs(self.lr), abs(self._init_lr))
            self.prev_lr.append(self.lr)
            if len(self.prev_lr) > self.patience:
                self.prev_lr.pop(0)
            self.lr = np.mean(self.prev_lr)
        return np.array(grad_descent.dy) * self.lr


class MomentumLearningRate(LearningRate):

    def __init__(self, momentum_rate=0.9, decay_rate=1e-1, **kwargs):
        super().__init__(**kwargs)
        self.momentum_rate = momentum_rate
        self.decay_rate = decay_rate
        self.nu = None

    def update(self, grad_descent):
        self.lr = self._init_lr / (1 + self.decay_rate * grad_descent.num_iter)
        if grad_descent.num_iter == 0:
            self.nu = np.zeros_like(grad_descent.x)
        self.nu = self.momentum_rate * self.nu + np.array(grad_descent.dy) * self.lr
        print(self.nu)
        return self.nu
