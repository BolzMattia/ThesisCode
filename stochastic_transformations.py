import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import mean_covariance_models as M
from utils import matrix2line_diagFunc, matrix2line_diagFunc_timeseries, indexes_librarian

_standard_gaussian = tfp.distributions.Normal(np.float32(0.0),np.float32(1.0))

class LinearProjector:
    """Represent an affine projection"""

    def __init__(self, input_dim, output_dim, jacobian_term=False):
        """
        Instantiate the projection parameters.
        :param input_dim: The input dimension
        :param output_dim: The output dimension
        :param jacobian_term: Set to True to return a regularization =log(A).
                              Use only if input_dim == output_dim
        """
        self.A = tf.Variable(initial_value=tf.keras.initializers.lecun_normal()(shape=[1, output_dim, input_dim]))
        self.b = tf.Variable(initial_value=tf.zeros(shape=[1, output_dim]))
        self.jacobian_term = jacobian_term
        if jacobian_term:
            assert input_dim == output_dim

    def callVI(self, x, regularization_penalty=0.0, variables=[]):
        """
        Apply the projection to x.
        :param x: The input tensor with dim=batch_size,input_dim
        :param regularization_penalty: The regularization term with dim=batch_size,1
        :param variables: The variables affecting the transformation
        :return: Output tensor, regularization tensor, list of variables
        """
        y = tf.matmul(self.A, x[:, :, tf.newaxis])[:, :, 0] + self.b
        if self.jacobian_term:
            log_jacobian = tf.math.log(tf.math.abs(tf.linalg.det(self.A)))
            regularization_penalty -= log_jacobian
            variables += v
        variables = [self.A, self.b]
        return y, regularization_penalty, variables


class LinearProjectorGaussianPosterior:
    """Same as LinearProjector but uses VI with a gaussian diagonal posterior distribution,
    to estimate the affine projection parameters"""
    stddev_A_mean = 1e-4
    stddev_A_std = 1e-2
    mean_A_std = -3.0
    stddev_b_std = 1e-1
    mean_b_std = -10.0

    def __init__(self, input_dim, output_dim):
        A_shape = [1, output_dim, input_dim]
        self.A_mean = tf.Variable(
            initial_value=tf.keras.initializers.RandomNormal(stddev=self.stddev_A_mean)(shape=A_shape))
        self._A_std = tf.Variable(
            initial_value=tf.keras.initializers.RandomNormal(mean=self.mean_A_std, stddev=self.stddev_A_std)(
                shape=A_shape))
        # self.A_std = tf.math.softplus(self._A_std)
        b_shape = [1, output_dim]
        self.b_mean = tf.Variable(initial_value=tf.zeros(shape=b_shape))
        self._b_std = tf.Variable(
            initial_value=tf.keras.initializers.RandomNormal(
                mean=self.mean_b_std, stddev=self.stddev_b_std)(shape=b_shape))
        # self.b_std = tf.math.softplus(self._b_std)
        self.pdf_regularization_b = tfp.distributions.Normal(
            loc=0.0,
            scale=tf.math.softplus(self.mean_b_std) + self.stddev_b_std)
        self.pdf_regularization_A = tfp.distributions.Normal(
            loc=0.0,
            scale=tf.math.softplus(self.mean_A_std) + self.stddev_A_std)
        # self.pdf_regularization_b = tfp.distributions.Normal(self.b_mean, tf.math.softplus(self._b_std))
        self.is_jacobian_singular = input_dim != output_dim

    def callVI(self, x, penalty=0.0, variables=[]):
        """
        Samples parameters form the posterior and then apply the projection to x.
        :param x: The input tensor with dim=batch_size,input_dim
        :param regularization_penalty: The regularization term with dim=batch_size,1
        :param variables: The variables affecting the transformation
        :return: Output tensor, regularization tensor, list of variables
        """
        y, log_jacobian, v = self._callVI(x)
        penalty -= log_jacobian + self.entropy()
        variables += v
        return y, penalty, variables

    def _callVI(self, x):
        """
        Internal function that samples the parameters from the posterior and then apply the transformation.
        :param x: the input tensor
        :return: Output tensor, regularization tensor, list of variables
        """
        A = self.A_mean + tf.math.softplus(self._A_std) * tf.random.normal(self.A_mean.shape)
        b = self.b_mean + tf.math.softplus(self._b_std) * tf.random.normal(self.b_mean.shape)
        y = tf.matmul(A, x[:, :, tf.newaxis])[:, :, 0] + b
        if self.is_jacobian_singular:
            log_jacobian = tf.zeros(1, dtype=tf.float32)
        else:
            log_jacobian = tf.math.log(tf.math.abs(tf.linalg.det(A)))
        #Regularization term, analogous to a non-informative prior on the net parameters
        # log_jacobian += (tf.reduce_sum(self.pdf_regularization_A.log_prob(A))
        #                  + tf.reduce_sum(self.pdf_regularization_b.log_prob(b)))
        variables = [self.A_mean, self._A_std, self.b_mean, self._b_std]
        return y, log_jacobian, variables

    def entropy(self):
        """Compute the entropy of the posterior."""
        entropy_A = tfp.distributions.Normal(self.A_mean, tf.math.softplus(self._A_std)).entropy()
        entropy_b = tfp.distributions.Normal(self.b_mean, tf.math.softplus(self._b_std)).entropy()
        entropy = tf.reduce_sum(entropy_A) + tf.reduce_sum(entropy_b)
        return entropy


def get_selu(jacobian_penalty=False):
    if jacobian_penalty:
        def selu(x, regularization_penalty=0.0, variables=[]) -> (tf.Tensor, tf.Tensor, list):
            '''
            Apply a Selu activation function to the particles, component by component.
            Return the log-Jacobian of the transformation and the variables that affects it.
            :param x: The particles to transform
            :type x: tf.Tensor
            :return: (Transformed particles, log-Jacobian, variables)
            :rtype: (tf.Tensor, tf.Tensor, list)
            '''
            alpha = 1.67326324
            scale = 1.05070098
            transformed = tf.nn.selu(x)
            log_jacobian = (
                    (tf.math.sign(transformed) * 0.5 + 0.5) * tf.math.log(scale)
                    # A constant is added into the logarithm to avoid numerical underflow
                    + (-tf.math.sign(transformed) * 0.5 + 0.5) * tf.math.log(1e-6 + transformed + alpha * scale)
            )
            regularization_penalty -= tf.reduce_sum(log_jacobian, axis=-1)
            if np.isinf(np.sum(regularization_penalty.numpy())):
                print(f'Selu: {regularization_penalty}, selu(x): {transformed}')
                import pdb
                pdb.set_trace()

            return transformed, regularization_penalty, variables
    else:
        def selu(x, regularization_penalty=0.0, variables=[]) -> (tf.Tensor, tf.Tensor, list):
            return tf.nn.selu(x), regularization_penalty, variables

    return selu


def get_activation(kind='selu', jacobian_penalty=False):
    """Access point of the managed activation functions."""
    return {
        'selu': get_selu
    }[kind](jacobian_penalty=jacobian_penalty)


def compose(*funcs):
    """Takes a list of functions and combine them into a single function.
    Passes the output of the previous as input the next one, ordered."""

    def composition(*args):
        for f in funcs:
            args = f(*args)
        return args

    return composition


def scaler(scale):
    """A layer that represent a fixed scaling of the input and adjust the regularization."""

    def f(x, regularization_penalty=0.0, variables=[]):
        x *= scale
        log_j = tf.math.log(tf.math.abs(scale))
        regularization_penalty -= log_j
        return x, regularization_penalty, variables

    return f


def MLP(input_dim: int, layers_dim: list, out_dim: int, init_scale=1.0, linear_projector=LinearProjector):
    """
    An MLP transformation with different layers and selu activation function.
    :param input_dim: The input dim
    :param layers_dim: The number of layers
    :param out_dim: The output dim
    :param init_scale: The scale of the output
    :param linear_projector: The class to use for the affine projection
    :return: Transformation function, List of the single layers objects
    """
    layers = []
    d1 = input_dim
    for d2 in layers_dim:
        layers.append(linear_projector(d1, d2).callVI)
        layers.append(get_activation('selu'))
        d1 = d2
    layers.append(linear_projector(d2, out_dim).callVI)
    layers.append(scaler(init_scale))
    mlp = compose(*layers)
    return mlp, layers


def reconstruction_loss_quadratic(x, x_hat):
    return tf.reduce_mean(tf.square(x - x_hat), axis=-1)


class AutoEncoder:
    """Represent an autoencoder"""

    def __init__(self,
                 input_dim: int,
                 encoder_layers_dim: list,
                 encode_dim: int,
                 decoder_layers_dim: list):
        self.encoder, _ = MLP(input_dim, encoder_layers_dim, encode_dim)
        self.decoder, _ = MLP(encode_dim, decoder_layers_dim, input_dim)

    def __call__(self, x, regularization_penalty=0.0, variables=[]):
        """
        Encode decode the values and return the results
        :param x: The input features
        :param regularization_penalty: Add the reconstruction loss to the regularization
        :param variables: The list of variables affecting the encoding-decoding
        :return: :rtype: (encoded tf.Tensor,regularization term tf.Tensor,variables list)
        """
        z, regularization_penalty, variables = self.encoder(
            x, regularization_penalty, variables)
        x_hat, regularization_penalty, variables = self.decoder(
            z, regularization_penalty, variables)
        regularization_penalty += reconstruction_loss_quadratic(x, x_hat)
        return z, regularization_penalty, variables


class LSTM:
    """Represent an LSTM layer followed by different MLP layers."""

    def __init__(self,
                 input_dim: int,
                 recurrent_dim: int,
                 post_recurrent_layers_dim: list,
                 out_dim: list,
                 gaussian_posterior=False,
                 init_scale=1.0):
        if gaussian_posterior:
            linear_projector = LinearProjectorGaussianPosterior
        else:
            linear_projector = LinearProjector

        lstm = tf.keras.layers.LSTM(recurrent_dim)

        def lstm_call(x, penalty=0.0, variables=[]):
            return lstm(x), penalty, variables

        mlp_post_recurrent, _ = MLP(recurrent_dim, post_recurrent_layers_dim, out_dim,
                                    linear_projector=linear_projector, init_scale=init_scale)
        shaper = compose(lstm_call, mlp_post_recurrent)
        self.lstm = lstm
        self.mlp_post = mlp_post_recurrent
        self.shaper = shaper

    def __call__(self, x, penalty=0.0, variables=[]):
        """
        Apply the LSTM and the MLP layers to the input.
        :param x: The input features
        :param regularization_penalty: Pass the regularization term
        :param variables: The list of variables affecting the encoding-decoding
        :return: :rtype: (encoded tf.Tensor,regularization term tf.Tensor,variables list)
        """
        return self.shaper(x, penalty, variables)


def GaussianReparametrizationSampler(x, n):
    """Map a line vector into mu,vcv states.
    Keep the first dimension intact as a batch."""
    mu, vcv, cholesky, log_jacobian = M.shape_mu_vcv(x, n)
    pdf = tfp.distributions.MultivariateNormalTriL(mu, cholesky)
    z = pdf.sample()
    prior_regularization = tf.reduce_sum(_standard_gaussian.log_prob(z), axis=1)
    entropy = pdf.entropy()
    return z, entropy + prior_regularization


def GaussianDiagonalReparametrizationSampler(x, n):
    """Map a line vector into mu,vcv states with the constraint of vcv being diagonal.
    Keep the first dimension intact as a batch."""
    mu = x[:, :n]
    sigma = tf.math.softplus(x[:, n:])
    pdf = tfp.distributions.Normal(mu, sigma)
    z = pdf.sample()
    prior_regularization = tf.reduce_sum( _standard_gaussian.log_prob(z), axis=1)
    entropy = tf.reduce_sum(pdf.entropy(), axis=1)
    return z, entropy + prior_regularization


class VariationalAutoEncoder:
    """Represent a Variational autoencoder"""

    def __init__(self,
                 input_dim: int,
                 encoder_layers_dim: list,
                 encode_dim: int,
                 decoder_layers_dim: list,
                 diagonal_covariance=True):
        self.encode_dim = encode_dim
        if diagonal_covariance:
            encode_parameters_dim = 2 * encode_dim
            self.reparametrization_trick = GaussianDiagonalReparametrizationSampler
        else:
            encode_parameters_dim = M.num_parameters_mean_cholesky(encode_dim)
            self.reparametrization_trick = GaussianReparametrizationSampler
        self.encoder, _ = MLP(input_dim, encoder_layers_dim, encode_parameters_dim, init_scale=1e-2)
        self.decoder, _ = MLP(encode_dim, decoder_layers_dim, input_dim)

    def __call__(self, x, regularization_penalty=0.0, variables=[]):
        """
        Encode-decode the values with the reparametrization trick and return the results.
        :param x: The input features
        :param regularization_penalty: Add the reconstruction loss to the regularization and the encoded values entropy.
        :param variables: The list of variables affecting the encoding-decoding
        :return: :rtype: (encoded tf.Tensor,regularization term tf.Tensor,variables list)
        """
        z_parameters, regularization_penalty, variables = self.encoder(
            x, regularization_penalty, variables)
        z, entropy = self.reparametrization_trick(z_parameters, self.encode_dim)
        x_hat, regularization_penalty, variables = self.decoder(z, regularization_penalty, variables)
        regularization_penalty += reconstruction_loss_quadratic(x, x_hat) - entropy
        return z, regularization_penalty, variables


if __name__ == '__main__':
    N = 3
    d = int(N * (N + 1) / 2) + N
    mc = 7
    x = np.float32(np.random.randn(mc, d))
    l1 = LinearProjectorGaussianPosterior(d, d)
    s = scaler(1e-4)
    c = compose(get_activation('selu'), l1.callVI, get_activation('selu'), s)
    lx, lp, v = c(x)
    print([x.shape for x in v])

    lstm = LSTM(d, d, [d * 2, d * 2], d)

    x_recurrent = np.float32(np.random.randn(mc, 5, d))
    y, lp, v = lstm(x_recurrent)
    print(lp, y)
    print(lp.shape, y.shape, len(v))

    mlp, _ = MLP(d, [d * 2, d * 2], d, linear_projector=LinearProjectorGaussianPosterior)
    y, lp_mlp, v_mlp = mlp(x)

    ae = AutoEncoder(d, [d * 2, d * 2], 2, [4, 4])
    z, regularization_penalty, v = ae(x)
    print(regularization_penalty, z)
    print(regularization_penalty.shape, z.shape)

    ae = VariationalAutoEncoder(d, [d * 2, d * 2], 3, [4, 4])
    z, regularization_penalty, v = ae(x)
    print(regularization_penalty, z)
    print(regularization_penalty.shape, z.shape)
