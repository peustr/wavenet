import numpy as np
from keras.models import load_model

from wavenet.preprocess import prediction_to_waveform_value


def generate_audio_from_model_output(
        path_to_model, path_to_output_wav, input_audio_size, generated_frames,
        sample_rate):
    wavenet = load_model(path_to_model)
    # We initialize with zeros, can also use a proper seed.
    generated_audio = np.zeros(input_audio_size, dtype=np.int16)
    cur_frame = 0
    while cur_frame < generated_frames:
        # Frame is always shifting by `cur_frame`, so that we can always
        # get the last `input_audio_size` values.
        probability_distribution = wavenet.predict(
            generated_audio[cur_frame:].reshape(
                1, input_audio_size, 1)).flatten()
        cur_sample = prediction_to_waveform_value(probability_distribution)
        generated_audio = np.append(generated_audio, cur_sample)
        cur_frame += 1
    return generated_audio
