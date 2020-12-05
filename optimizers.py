import tensorflow as tf


class optimizer_adam:
    """Wrap the tensorflow optimizers, for easy handling and gradient clipping."""
    def __init__(self, learning_rate=1e-4, beta_1=0.1, clip_norm=2e1):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1)
        self.optimizer = optimizer
        self.clip_norm = clip_norm

    def optimize(self, g, variables):
        """Optimize the variables with the gradient in g."""
        if self.clip_norm is not None:
            g = [tf.clip_by_norm(x, self.clip_norm) for x in g]
        self.optimizer.apply_gradients(zip(g, variables))
