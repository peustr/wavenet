from keras.layers import (
    Activation, Add, Conv1D, Dense, Flatten, Input, Multiply)
from keras.models import Model


def WaveNetResidualConv1D(num_filters, kernel_size, dilation_rate):
    """ Function that creates a residual block for the WaveNet with gated
        activation units, skip connections and residual output, as described
        in Sections 2.3 and 2.4 of the paper [1].

        Args:
            num_filters (int): Number of filters used for convolution.
            kernel_size (int): The size of the convolution.
            dilation_rate (int): The dilation rate for the dilated convolution.

        Returns:
            A layer wrapper compatible to the Keras functional API.

        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    def build_residual_block(l_input):
        # Gated activation.
        l_sigmoid_conv1d = Conv1D(
            num_filters, kernel_size, dilation_rate=dilation_rate,
            padding="same", activation="sigmoid")(l_input)
        l_tanh_conv1d = Conv1D(
            num_filters, kernel_size, dilation_rate=dilation_rate,
            padding="same", activation="tanh")(l_input)
        l_mul = Multiply()([l_sigmoid_conv1d, l_tanh_conv1d])
        # Branches out to skip unit and residual output.
        l_skip_connection = Conv1D(1, 1)(l_mul)
        l_residual = Add()([l_input, l_skip_connection])
        return l_residual, l_skip_connection
    return build_residual_block


def build_wavenet_model(input_size, num_filters, kernel_size,
                        num_residual_blocks):
    """ Returns an implementation of WaveNet, as described in Section 2
        of the paper [1].

        Args:
            input_size (int): The size of the waveform the network will
                consider as input.
            num_filters (int): Number of filters used for convolution.
            kernel_size (int): The size of the convolution.
            num_residual_blocks (int): How many residual blocks to generate
                between input and output. Residual block i will have a dilation
                rate of 2^(i+1), i starting from zero.

        Returns:
            A Keras model representing the WaveNet.

        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    l_input = Input(batch_shape=(None, input_size, 1))
    l_stack_conv1d = Conv1D(num_filters, kernel_size, padding="same")(l_input)
    l_skip_connections = []
    for i in range(num_residual_blocks):
        l_stack_conv1d, l_skip_connection = WaveNetResidualConv1D(
            num_filters, kernel_size, 2 ** (i + 1))(l_stack_conv1d)
        l_skip_connections.append(l_skip_connection)
    l_sum = Add()(l_skip_connections)
    relu = Activation("relu")(l_sum)
    l1_conv1d = Conv1D(1, 1, activation="relu")(relu)
    l2_conv1d = Conv1D(1, 1)(l1_conv1d)
    l_flatten = Flatten()(l2_conv1d)
    l_output = Dense(256, activation="softmax")(l_flatten)
    model = Model(inputs=[l_input], outputs=[l_output])
    model.compile(
        loss="categorical_crossentropy", optimizer="adam",
        metrics=["accuracy"])
    return model
