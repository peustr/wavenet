from wavenet.train import train_wavenet


def main():
    path_to_audio_train = "data/train/"
    path_to_audio_valid = "data/valid/"
    input_size = 1024
    num_filters = 16
    kernel_size = 2
    num_residual_blocks = 9
    batch_size = 32
    num_epochs = 20

    distinct_filename = "wavenet_in{}_nf{}_k{}_nres{}_bat{}_e{}.h5".format(
        input_size,
        num_filters,
        kernel_size,
        num_residual_blocks,
        batch_size,
        num_epochs)

    wavenet, history = train_wavenet(
        path_to_audio_train,
        path_to_audio_valid,
        input_size,
        num_filters,
        kernel_size,
        num_residual_blocks,
        batch_size,
        num_epochs)

    # Persist model with a distinct name to remember the training parameters.
    print("Saving {}...".format(distinct_filename))
    wavenet.save(distinct_filename)


if __name__ == "__main__":
    main()
