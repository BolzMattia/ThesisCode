import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

#Gaussian standard distribution
_pdf_standard_gaussian = tfp.distributions.Normal(0.0, 1.0)
#Default dimension for hidden layers
_hidden_layer_dim = 16


class DSamplerUniform:
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

    def sample(self, size: int) -> (tf.Tensor, tf.Tensor, list):
        '''
        Samples particles with the uniform distrution
        Returns a triplet: samples, log density of samples, variables affecting the density
        :param size: the number of samples, as first dimension of the return
        :type size: int
        :return: (samples,log_density,variables)
        :rtype: tuple(tf.Tensor,tf.Tensor,list)
        '''
        eps = tf.random.uniform(
            [size, self.d],
            minval=self.min_val,
            maxval=self.max_val,
            dtype=tf.dtypes.float32
        )
        log_density = self.particles_density * np.ones(size)
        return eps, log_density, []


def good_transformation(f):
    '''
    Decorator of a function that takes samples, transform them and returns the Jacobian of the transformation,
    with the variables that affects the transformation
    :param f: The function that transform the samples and returns (samples,log_Jacobian,variables)
    :type f: Function
    :return: The Function that adds the log_Jacobian and the variables to the input log_Jacobian and variables
    :rtype: Function
    '''

    def transformed(x, log_p, lambdas) -> (tf.Tensor, tf.Tensor, list):
        tx, j, l = f(x)
        return tx, log_p - j, lambdas + l

    return transformed


class AffineProjection:
    def __init__(self, d: int, initializer=None, scale=1.0):
        '''
        Represent an affine transformation, returns also the log-Jacobian and the variables.
        :param d: The dimension of the projection
        :type d: int
        :param initializer: The weights and bias initializer
        :type initializer: tf.initializers
        :param scale: Scales the output by this value
        :type scale: float
        '''
        if initializer is None:
            init = tf.initializers.lecun_normal()
        self.d = d
        scale = tf.constant(scale)
        A = tf.Variable(init(shape=[d, d], dtype=tf.float32))
        b = tf.Variable(init(shape=[d], dtype=tf.float32))
        j_correction = np.log(scale) * d

        # The decorated function that is used to apply the affine transformation
        @good_transformation
        def proj(x) -> (tf.Tensor, tf.Tensor, list):
            log_jacobian = tf.math.log(tf.math.abs(tf.linalg.det(A))) + j_correction
            variables = [A, b]
            y = (tf.matmul(x, A) + b) * scale
            return y, log_jacobian, variables

        self.proj = proj


@good_transformation
@tf.function
def selu_log_jacobian(particles) -> (tf.Tensor, tf.Tensor, list):
    '''
    Apply a Selu activation function to the particles, component by component.
    Return the log-Jacobian of the transformation and the variables that affects it.
    :param particles: The particles to transform
    :type particles: tf.Tensor
    :return: (Transformed particles, log-Jacobian, variables)
    :rtype: (tf.Tensor, tf.Tensor, list)
    '''
    alpha = 1.67326324
    scale = 1.05070098
    transformed = tf.nn.selu(particles)
    log_jacobian = (
            (tf.math.sign(transformed) * 0.5 + 0.5) * tf.math.log(scale)
            + (-tf.math.sign(transformed) * 0.5 + 0.5) * tf.math.log(transformed + alpha * scale)
    )
    log_jacobian = tf.reduce_sum(log_jacobian, axis=-1)
    return transformed, log_jacobian, []


class AffineProjectionLowDim:
    def __init__(self, d: int, d1: int, initializer=None, scale=1.0):
        '''
        Represent an affine projection from d dimension to d1 dimension.
        Adds d1-d random variables, uniformly distributed to preserve the density expression.
        :param d: The dimension of input
        :type d: int
        :param d1: The dimension of output
        :type d1: int
        :param initializer: The initializer for the weights and bias of the affine transformation
        :type initializer: tf.initializer
        :param scale: The output is scaled by this constant
        :type scale: float
        '''
        assert d1 < d, f"Final dimension ({d}) must be bigger than initial dimension ({d1})"
        if initializer is None:
            init = tf.initializers.lecun_normal()
        self.d1 = d1
        self.d = d
        scale = tf.constant(scale)
        j_correction = np.log(scale) * (d - d1)
        proj11 = AffineProjection(d1)
        A12 = tf.Variable(init(shape=[d1, d - d1], dtype=tf.float32))
        b2 = tf.Variable(init(shape=[d - d1], dtype=tf.float32))
        D22 = tf.Variable(init(shape=[d - d1], dtype=tf.float32))
        extra_sampler = DSamplerUniform(d - d1)

        # The function that applies the affine transformation
        @good_transformation
        def proj(x) -> (tf.Tensor, tf.Tensor, list):
            x2, log_p2, v2 = extra_sampler.sample(x.shape[0])
            x2 = (tf.matmul(x, A12) + b2 + x2 * D22) * scale
            x1, log_p1, v1 = proj11.proj(x, 0.0, [])
            log_jacobian = tf.reduce_sum(tf.math.log(tf.math.abs(D22))) - log_p1 - log_p2 + j_correction
            variables = [A12, b2, D22]
            y = tf.concat([x1, x2], axis=1)
            return y, log_jacobian, variables

        self.proj = proj


def layer_builder(kind, **config):
    """Return a single layer transformation with the given characteristics"""
    l = {
        'affine': AffineProjection,
        'affine_large_dim': AffineProjectionLowDim
    }[kind](**config)
    return l


class transformer_selu:
    """Represent a sequence of layers with selu activation function,
    that also compute the jacobian of the transformation"""
    def __init__(self, d: int, layers_config: dict, init_scale: float):
        layers = []
        for layer in layers_config:
            # If the layer dim is not explicit it is forced to the problem dim
            if 'd' not in layer:
                layer['d'] = d
            layers.append(layer_builder(**layer))
        final_affine = AffineProjection(d, scale=init_scale)
        self.final_affine = final_affine
        self.layers = layers

    def __call__(self, theta, log_p, variables):
        """
        Apply the transformation to theta.
        :param theta: The input tensor dim=batch_size, d
        :param log_p: The Jacobian of the transformation
        :param variables: The variables affecting the transformation
        :return: (tensor dim=(batch_size,d), tensor dim=(batch_size,1),list of variables)
        """
        for layer in self.layers:
            theta, log_p, variables = selu_log_jacobian(
                *layer.proj(theta, log_p, variables)
            )
        theta, log_p, variables = self.final_affine.proj(theta, log_p, variables)
        return theta, log_p, variables


class hierarchical_sampler:
    """Represent an HVI sampler with uniform sampling distribution
    and selu activation function beween layers"""
    def __init__(self, d: int, epsilon_config: dict, layers_config: dict, init_scale: float):
        ##Epsilon sampling
        if 'd' not in epsilon_config:
            epsilon_config['d'] = d
        eps = DSamplerUniform(**epsilon_config)

        ##Hierarchical reparametrization
        reparametrization = transformer_selu(d, layers_config, init_scale)

        self.eps = eps
        self.reparametrization = reparametrization

    def sample(self, MC: int):
        """
        Sample particles with known distribution.
        :param MC: How many particles
        :return: (tensor dim=(batch_size,d), tensor dim=(batch_size,1),list of variables)
        """
        theta0, log_p, variables = self.eps.sample(MC)
        theta0, log_p, variables = self.reparametrization(theta0, log_p, variables)
        if np.isnan(theta0.numpy()).any(0).any(0):
            print('ERROR! Nan in the HVI sampler')
        return theta0, log_p, variables


class transformer_mlp:
    """DEPRECATED"""
    def __init__(self, out_dim, input_shape=None, example=None, init_scale=1.0):
        if input_shape is None:
            input_shape = example.shape[1:]

        mlp = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(out_dim)
        ])

        self.f = mlp
        self.init_scale = init_scale

    def __call__(self, x, particles_global=None):
        z = self.f(x) * self.init_scale
        log_p = tf.constant(0.0, dtype=tf.float32)
        return z, log_p, self.f.weights


class dense_Bayesian_selu_stdPrior:
    """DEPRECATED"""
    def __init__(self, input_dim, output_dim, init_scale=1.0):
        A_norm = np.float32(np.sqrt(input_dim + output_dim))
        b_norm = np.float32(output_dim)
        self.b_mu = tf.Variable(np.zeros(output_dim), dtype=tf.float32)
        self.b_sigma = tf.Variable(np.ones(output_dim) / b_norm, dtype=tf.float32)
        self.A_mu = tf.Variable(np.zeros([output_dim, input_dim]), dtype=tf.float32)
        self.A_sigma = tf.Variable(np.ones([output_dim, input_dim]) / A_norm, dtype=tf.float32)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_A = tfp.distributions.Normal(0.0, A_norm)
        self.prior_b = tfp.distributions.Normal(0.0, b_norm)
        self.pdf_b = tfp.distributions.Uniform(self.b_mu, self.b_sigma)
        self.pdf_A = tfp.distributions.Uniform(self.A_mu, self.A_sigma)
        self.variables = [self.b_mu, self.b_sigma, self.A_mu, self.A_sigma]

    def __call__(self, x):
        eps_b = tf.random.uniform([self.output_dim])
        eps_A = tf.random.uniform([self.output_dim, self.input_dim])
        b = self.b_mu + eps_b * self.b_sigma
        A = self.A_mu + eps_A * self.A_sigma
        transformed = b[tf.newaxis] + tf.matmul(A, x[:, :, tf.newaxis])[:, 0, :]
        transformed, _, _ = selu_log_jacobian(transformed, 0.0, [])
        # Prior
        log_p = -(tf.reduce_sum(self.prior_b.log_prob(self.b_sigma))
                  + tf.reduce_sum(self.prior_A.log_prob(self.A_sigma)))
        # Entropy
        log_p -= (tf.reduce_sum(self.pdf_b.entropy()) + tf.reduce_sum(self.pdf_A.entropy()))
        return transformed, log_p, self.variables


class transformer_mlp_Bayesian:
    """DEPRECATED"""
    def __init__(self, out_dim, input_shape=None, example=None, init_scale=1.0):
        if input_shape is None:
            input_shape = example.shape[1]
        mlp = [
            dense_Bayesian_selu_stdPrior(input_shape, _hidden_layer_dim),
            dense_Bayesian_selu_stdPrior(_hidden_layer_dim, _hidden_layer_dim),
            dense_Bayesian_selu_stdPrior(_hidden_layer_dim, _hidden_layer_dim),
            dense_Bayesian_selu_stdPrior(_hidden_layer_dim, out_dim),
        ]
        self.mlp = mlp
        self.init_scale = init_scale

    def __call__(self, x, particles_global=None):
        z = x
        log_p = tf.math.log(self.init_scale)
        variables = []
        for l in self.mlp:
            z, prior, v = l(z)
            log_p += prior
            variables += v
        return z, log_p, variables


class transformer_variational_autoencoder:
    """DEPRECATED"""
    def __init__(self, out_dim, encoding_dim, input_shape=None, example=None, init_scale=1.0):
        if input_shape is None:
            input_shape = example.shape[1:]

        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(encoding_dim * 2)
        ])

        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(encoding_dim,)),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(input_shape[0])
        ])

        forecaster = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(encoding_dim,)),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(out_dim)
        ])
        self.encoder = encoder
        self.decoder = decoder
        self.forecaster = forecaster
        self.variables = self.encoder.weights + self.decoder.weights + self.forecaster.weights
        self.init_scale = init_scale
        self.encoding_dim = encoding_dim

    def __call__(self, x, particles_global=None):
        e = self.encoder(x)
        # Gaussian variational distribution with mean and standard deviation depending on the encoded values
        m, s = e[:, :self.encoding_dim], tf.nn.softplus(e[:, self.encoding_dim:])
        # Decode using a single samples from the Gaussian variational distribution
        e = m + s * np.random.randn(x.shape[0], self.encoding_dim)
        x_hat = self.decoder(e)
        z = self.forecaster(e) * self.init_scale
        log_p = tf.keras.losses.MSE(x, x_hat) - tf.reduce_sum(tf.math.log(s), axis=1) * 0.5
        return z, log_p, self.variables


class transformer_autoencoder:
    """DEPRECATED"""
    def __init__(self, out_dim, encoding_dim, input_shape=None, example=None, init_scale=1.0):
        if input_shape is None:
            input_shape = example.shape[1:]

        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(input_shape)),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(encoding_dim)
        ])

        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(encoding_dim,)),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(input_shape[0])
        ])

        forecaster = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(encoding_dim,)),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(out_dim)
        ])

        self.encoder = encoder
        self.decoder = decoder
        self.forecaster = forecaster
        self.variables = self.encoder.weights + self.decoder.weights + self.forecaster.weights
        self.init_scale = init_scale

    def __call__(self, x, particles_global=None):
        e = self.encoder(x)
        x_hat = self.decoder(e)
        z = self.forecaster(e) * self.init_scale
        log_p = tf.keras.losses.MSE(x, x_hat)
        return z, log_p, self.variables


class transformer_lstm:
    """DEPRECATED"""
    def __init__(self, out_dim, input_shape=None, example=None, init_scale=1.0):
        if input_shape is None:
            input_shape = example.shape[1:]

        lstm = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(_hidden_layer_dim, activation='selu'),
            tf.keras.layers.Dense(out_dim)
        ])

        self.f = lstm
        self.init_scale = init_scale

    def __call__(self, x, particles_global=None):
        z = self.f(x) * self.init_scale
        log_p = tf.constant(0.0, dtype=tf.float32)
        return z, log_p, self.f.weights


class factory:
    def get(d: int, **sampler_config):
        sampler = hierarchical_sampler(d, **sampler_config)
        return sampler


if __name__ == '__main__':
    d = 2000
    config = {
        "init_scale": 0.1,
        "epsilon_config": {},
        "layers_config": [
            {
                "kind": "affine"
            },
            {
                "kind": "affine"
            },
            {
                "kind": "affine"
            }
        ]
    }
    s = hierarchical_sampler(d, **config)
    import pdb

    pdb.set_trace()
    print(s.sample(100))
