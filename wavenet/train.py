from wavenet.model import build_wavenet_model
from wavenet.preprocess import get_audio_sample_batches


def train_wavenet(path_to_audio_train, path_to_audio_valid, input_size,
                  num_filters, kernel_size, num_residual_blocks,
                  batch_size, num_epochs):
    """ A wrapper function that, given the path to audio files for training
        and validation, trains the WaveNet and returns a trained Keras model
        with its training history metadata.

        Args:
            path_to_audio_train (str): Path to the directory containin the
                audio files for training.
            path_to_audio_valid (str): Path to the directory containin the
                audio files for validation.
            input_size (int): The size of the input layer of the network,
                and the receptive field during the construction of the data
                samples.
            num_filters (int): Number of filters used for convolution in the
                causal and dilated convolution layers.
            kernel_size (int): Convolution window size for the causal and
                dilated convolution layers.
            num_residual_blocks (int): How many residual blocks to generate
                between input and output. Residual block i will have a dilation
                rate of 2^(i+1), i starting from zero.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
    """
    wavenet = build_wavenet_model(
        input_size, num_filters, kernel_size, num_residual_blocks)
    print("Generating training data...")
    X_train, y_train = get_audio_sample_batches(
        path_to_audio_train, input_size)
    print("Generating validation data...")
    X_test, y_test = get_audio_sample_batches(
        path_to_audio_valid, input_size)
    print("Training model...")
    history = wavenet.fit(
        X_train, y_train, batch_size=batch_size, epochs=num_epochs,
        validation_data=(X_test, y_test))
    return wavenet, history
