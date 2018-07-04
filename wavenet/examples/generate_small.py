import scipy.io.wavfile as wav

from wavenet.generate import generate_audio_from_model_output


def main():
    path_to_model = "wavenet_in1024_nf16_k2_nres9_bat32_e20.h5"
    path_to_output_wav = "wavenet_sample_output.wav"
    generated_frames = 2 ** 17  # About 5 seconds of speech (or noise!).
    input_audio_size = 1024
    # Sample rate same as the files used to train the model.
    sample_rate = 22050

    generated_audio = generate_audio_from_model_output(
        path_to_model, path_to_output_wav, input_audio_size,
        generated_frames, sample_rate)
    wav.write(path_to_output_wav, sample_rate, generated_audio)


if __name__ == "__main__":
    main()
