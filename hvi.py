import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow.keras.layers import PReLU


class IdActivation:
    """Represent a naive identity activation layer."""
    def jacobian(self, x=None, transformed=None):
        return 0.0

    def inverse(self, y):
        return y

    def inverse_jacobian(self, x=None, transformed=None):
        return 0.0


class PReluActivation:
    """Represent a PRelu activation layer, enhanced with the Jacobian computation of the activation"""
    def __init__(self, a=0.01):
        self.a = a

    def jacobian_vector(self, transformed=None):
        log_jacobian = (-tf.math.sign(transformed) * 0.5 + 0.5) * tf.math.log(self.a)
        return log_jacobian

    def jacobian(self, transformed=None):
        log_jacobian = self.jacobian_vector(transformed)
        log_jacobian = tf.reduce_sum(log_jacobian, axis=-1)
        return log_jacobian

    def __call__(self, x):
        g0 = tf.cast(x < 0, tf.float32)
        x = x * (g0 * self.a + (1 - g0))
        return x

    def inverse(self, y):
        g0 = tf.cast(y < 0, tf.float32)
        x = y * (g0 / self.a + (1 - g0))
        return x

    def inverse_jacobian(self, transformed=None):
        return -self.jacobian(transformed)


class SeluActivation:
    """Represent a Selu activation layer, enhanced with the Jacobian computation of the activation"""
    alpha = 1.67326324
    scale = 1.05070098

    def jacobian(self, transformed=None):
        log_jacobian = (
                (tf.math.sign(transformed) * 0.5 + 0.5) * tf.math.log(self.scale)
                + (-tf.math.sign(transformed) * 0.5 + 0.5) * tf.math.log(transformed + self.alpha * self.scale)
        )
        log_jacobian = tf.reduce_sum(log_jacobian, axis=-1)
        return log_jacobian

    def inverse(self, y):
        g0 = tf.cast(y < 0, tf.float32)
        x = tf.math.log(y / self.scale / self.alpha + 1) * g0 + y / self.scale * (1 - g0)
        return x

    def inverse_jacobian(self, transformed=None):
        g0 = tf.cast(transformed >= 0, tf.float32)
        log_jacobian = (
            tf.reduce_sum(
                -np.log(self.scale) * g0 + tf.math.log(
                    (transformed + self.scale * self.alpha) / self.scale / self.alpha) * (1 - g0)
                , axis=1)
        )
        return log_jacobian


class DenseInvertibleVI(tf.keras.layers.Dense):
    """
    Inherit from the keras Dense layer,
    but adds an extra method ('callVI') that computes not only the layer value,
    but also the Jacobian of the transformation
    """
    def __init__(self, output_dim=None, activation='linear', *args, **kwargs):
        if activation == 'selu':
            self.activationVI = SeluActivation()
        elif activation == 'linear':
            self.activationVI = IdActivation()
        elif activation == 'prelu':
            self.activationVI = PReluActivation()
            activation = PReLU(tf.initializers.constant(self.activationVI.a))
        else:
            raise Exception(f"Jacobian computation not available for the activation function: {activation}")
        tf.keras.layers.Dense.__init__(self, output_dim, activation, *args, **kwargs)

    def callVI(self, x: int, log_jacobian=0.0):
        y = self(x)
        log_jacobian += tf.math.log(tf.math.abs(tf.linalg.det(self.weights[0])))
        log_jacobian += self.activationVI.jacobian(y)
        return y, log_jacobian

    def callInverseVI(self, y: int, log_jacobian=0.0):
        m = self.weights[0]
        b = self.weights[1]
        im = tf.linalg.inv(m)
        x = self.activationVI.inverse(y)
        log_jacobian_activation = self.activationVI.jacobian(y)
        x = tf.matmul(x - b[tf.newaxis], im)
        log_jacobian -= tf.math.log(tf.math.abs(tf.linalg.det(m))) + log_jacobian_activation
        return x, log_jacobian


class SequentialVI(tf.keras.models.Sequential):
    """
    Inherit from the keras Sequential layer,
    but adds an extra method ('callVI') that computes not only the layer value,
    but also the Jacobian of the transformation,
    calling the methods 'callVI' of the single layers).
    """
    def callVI(self, x: int, log_jacobian=0.0):
        for layer in self.layers:
            x, log_jacobian = layer.callVI(x, log_jacobian)
        return x, log_jacobian

    def callInverseVI(self, y: int, log_jacobian=0.0):
        ll = len(self.layers)
        for i in range(ll):
            layer = self.layers[ll - 1 - i]
            y, log_jacobian = layer.callInverseVI(y, log_jacobian)
        return y, log_jacobian


class UniformSamplerVI:
    def __init__(self, d: int, min_val=-1.0, max_val=+1.0):
        '''
        Represents a density sampler with uniform distribution.
        :param d: dimension of the samples
        :type d: int
        :param min_val: minimum value of the d coordinates
        :type min_val: float
        :param max_val: maximum value of the d coordinates
        :type max_val: float
        '''
        assert max_val > min_val
        self.particles_density = d * np.log(np.float32(max_val - min_val))  # 0*tf.reduce_mean(particles,axis=1)
        self.d = d
        self.min_val = min_val
        self.max_val = max_val

    def sample(self, size: int) -> (tf.Tensor, tf.Tensor):
        '''
        Samples particles with the uniform distrution
        Returns a pair: samples, log density of samples
        :param size: the number of samples, first dimension of both the return values
        :type size: int
        :return: (samples,log_density)
        :rtype: tuple(tf.Tensor,tf.Tensor)
        '''
        eps = tf.random.uniform(
            [size, self.d],
            minval=self.min_val,
            maxval=self.max_val,
            dtype=tf.dtypes.float32
        )
        log_density = self.particles_density * np.ones(size)
        return eps, log_density

    def __call__(self, particles: np.array):
        """
        Compute the density over the input particles.
        :param particles: A tensor
        :return: The uniform density of the particles.
        """
        if (particles > self.min_val).numpy().all(0).all(0) & (particles < self.max_val).numpy().all(0).all(0):
            return self.particles_density * np.ones(particles.shape, dtype=np.float32)
        else:
            return np.zeros(particles.shape, dtype=np.float32)


class GaussianDiagonalSamplerVI:
    def __init__(self, d: int, loc=0.0, scale=1.0):
        '''
        Represents a density sampler with gaussian distribution.
        :param d: dimension of the samples
        :type d: int
        :param loc: mean of the distribution
        :type loc: float
        :param scale: standard deviaton of the distribution
        :type scale: float
        '''
        self.d = d
        self.pdf = tfp.distributions.Normal(loc=loc, scale=scale)

    def sample(self, size: int) -> (tf.Tensor, tf.Tensor):
        '''
        Samples particles with the given distribution
        Returns a pair: samples, log density of samples
        :param size: the number of samples, first dimension of both the return values
        :type size: int
        :return: (samples,log_density)
        :rtype: tuple(tf.Tensor,tf.Tensor)
        '''
        eps = self.pdf.sample([size, self.d])
        log_density = self(eps)
        return eps, log_density

    def __call__(self, particles: np.array):
        """
        Compute the density over the input particles.
        :param particles: A tensor
        :return: The gaussian density of the particles.
        """
        log_density = tf.reduce_mean(self.pdf.log_prob(particles), axis=-1)
        return log_density


class hierarchical_sampler:
    """
    Represent a complete HVI sampler of a fixed dimension,
    with Uniform density transformed with
    affine projection and selu activations.
    """
    def __init__(self, d: int, layers: int = 2, init_scale: float = 1.0):
        """
        Instance the object
        :param d: Dimension of the sampler
        :param layers: How many layers to include
        :param init_scale: The initial scale of the output.
        """
        eps = UniformSamplerVI(d)
        ##Hierarchical transformations
        reparametrization = SequentialVI(
            [DenseInvertibleVI(d, activation='selu') for i in range(layers - 1)] + [DenseInvertibleVI(d)]
        )
        self.eps = eps
        self.init_scale = init_scale
        self.reparametrization = reparametrization

    def get_variables(self):
        return self.reparametrization.weights

    def sample(self, MC: int):
        """
        Obtain samples for VI.
        :param MC: How many samples
        :return: The samples, the Jacobian of the transformation and the variables affecting the samples
        """
        theta0, log_p = self.eps.sample(MC)
        theta0, log_p = self.reparametrization.callVI(theta0, log_p)
        theta0 *= self.init_scale
        log_p -= tf.math.log(self.init_scale)
        if np.isnan(theta0.numpy()).any(0).any(0):
            print('ERROR! Nan in the HVI sampler')
        return theta0, log_p, self.get_variables()


if __name__ == '__main__':
    # Dimensions
    encoding_dim = 100
    _hidden_layer_dim = 10
    output_shape = 3

    # Transformations
    decoder = SequentialVI([
        DenseInvertibleVI(encoding_dim, activation='selu'),
        DenseInvertibleVI(encoding_dim, activation='selu'),
        # DenseInvertibleVI(encoding_dim, activation='selu'),
        # DenseInvertibleVI(encoding_dim, activation='selu'),
        # DenseInvertibleVI(encoding_dim, activation='selu'),
        # DenseInvertibleVI(encoding_dim, activation='prelu'),
        DenseInvertibleVI(encoding_dim),
    ])

    # Initial sampler
    # sam = UniformSamplerVI(encoding_dim)
    sam = GaussianDiagonalSamplerVI(encoding_dim)

    # Sample and transform
    x, lj = sam.sample(50)
    tx, tlj = decoder.callVI(x, lj)

    # Inverse transform
    x2, lj2 = decoder.callInverseVI(tx, tlj)
    import pdb

    pdb.set_trace()
    print(x - x2, lj - lj2)

    # Variables to optimize
    variables = decoder.weights

    # Wrapper for standard HVI sampler
    h = hierarchical_sampler(encoding_dim)
    h.sample(16)
