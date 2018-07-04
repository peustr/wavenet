import os

import numpy as np
import scipy.io.wavfile as wav


def read_audio(filename):
    """ Reads a ".wav" file and returns the waveform as a series of signed
        16-bit integers.

        Args:
            filename (str): The path to a wav file.
    """
    _, audio = wav.read(filename)
    return audio


# The following 'scale_*' functions implement the known formula:
# f(x) = (b - a) * ((x - min) / (max - min)) + a
# For scaling a given value x between the range [a, b].


def scale_audio_uint8_to_float64(arr):
    """ Scales an array of 8-bit unsigned integers [0-255] to [-1, 1].
    """
    vmax = np.iinfo(np.uint8).max
    vmin = np.iinfo(np.uint8).min
    arr = arr.astype(np.float64)
    return 2 * ((arr - vmin) / (vmax - vmin)) - 1


def scale_audio_int16_to_float64(arr):
    """ Scales an array of 16-bit integers [-2^15, 2^15 - 1] to [-1, 1].
    """
    vmax = np.iinfo(np.int16).max
    vmin = np.iinfo(np.int16).min
    arr = arr.astype(np.float64)
    return 2 * ((arr - vmin) / (vmax - vmin)) - 1


def scale_audio_float64_to_uint8(arr):
    """ Scales an array of float values between [-1, 1] to an 8-bit
        unsigned integer value [0, 255].
        Inverse of `scale_audio_uint8_to_float64`.
    """
    vmax = np.iinfo(np.uint8).max
    arr = ((arr + 1) / 2) * vmax
    arr = arr.astype(np.uint8)
    return arr


def scale_audio_float64_to_int16(arr):
    """ Scales an array of float values between [-1, 1] to a 16-bit
        integer value [-2^15, 2^15 - 1].
        Inverse of `scale_audio_int16_to_float64`.
    """
    vmax = np.iinfo(np.int16).max
    vmin = np.iinfo(np.int16).min
    arr = ((arr + 1) / 2) * (vmax - vmin) + vmin
    arr = arr.astype(np.int16)
    return arr


def mu_law(xt, mu=255):
    """ Transforms the normalized audio signal to values between [-1, 1]
        so that it can be quantized in range [0, 255] for the softmax output.
        See Section 2.2 of the paper [1].

        See:
            [1] Oord, Aaron van den, et al. "Wavenet: A generative model for
                raw audio." arXiv preprint arXiv:1609.03499 (2016).
    """
    return np.sign(xt) * (np.log(1 + mu * np.absolute(xt)) / np.log(1 + mu))


def mu_law_inverse(yt, mu=255):
    """ The inverse transformation of mu-law that expands the input back to
        the original space.
    """
    return np.sign(yt) * (1 / mu) * (((1 + mu) ** np.abs(yt)) - 1)


def to_one_hot(xt):
    """ Converts an integer value between 0 and 255 to its one-hot
        representation.
    """
    return np.eye(256, dtype="uint8")[xt]


def get_audio_sample_batches(path_to_audio, receptive_field_size,
                             stride_step=32):
    """ Provides the audio data in batches for training and validation.

        Note: This function used to be a generator, but when experimenting
        with little data, a function makes more sense.

        Args:
            path_to_audio_train (str): Path to the directory containin the
                audio files for training.
            receptive_field_size (int): The size of the sliding window that
                passes over the data and collects training samples.
            stride_step (int, default:32): The step by which the window slides.
    """
    audio_files = [
        path_to_audio + fn for fn in os.listdir(path_to_audio)
        if fn.endswith(".wav")]
    X = []
    y = []
    for audio_file in audio_files:
        audio = read_audio(audio_file)
        # Scales audio from 16 bit int to (-1, 1).
        audio = scale_audio_int16_to_float64(audio)
        offset = 0
        while offset + receptive_field_size - 1 < len(audio):
            X.append(
                audio[
                    offset:offset + receptive_field_size
                ].reshape(receptive_field_size, 1))
            # For y apply mu-law, scale to 0-255 and make one-hot vector.
            y_cur = audio[receptive_field_size]
            y_cur = mu_law(y_cur)
            y_cur = scale_audio_float64_to_uint8(y_cur)
            y.append(to_one_hot(y_cur))
            offset += stride_step
    return np.array(X), np.array(y)


def prediction_to_waveform_value(probability_distribution, random=False):
    """ Accepts the output of the WaveNet as input (a probability vector of
        size 256) and outputs a 16-bit integer that that corresponds to the
        position selected in the expanded space.

        Args:
            probability_distribution (np.array(256)): An 1-dimensional vector
                that represents the probability distribution over the next
                value of the generated waveform.
            random (bool, default:False): If true, a random value between 0
                and 256 will be used to reconstruct the signal, drawn according
                to the provided distribution. Otherwise, the most probable
                value will be selected.
    """
    if random:
        choice = np.random.choice(range(256), p=probability_distribution)
    else:
        choice = np.argmax(probability_distribution)
    # Project the predicted [0, 255] integer value back to [-2^15, 2^15 - 1].
    y_cur = scale_audio_uint8_to_float64(choice)
    y_cur = mu_law_inverse(y_cur)
    y_cur = scale_audio_float64_to_int16(y_cur)
    return y_cur
