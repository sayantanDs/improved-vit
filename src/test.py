import os
import numpy as np
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf
from tensorflow.keras.models import model_from_json

from models import vit, cvit



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset_path', default="speech_commands_v2_spectrograms")
    parser.add_argument('-m', '--model_json', default="../trained_models/cvit/model.json")
    parser.add_argument('-w', '--model_weight', default="../trained_models/cvit/cvit_100-0.952.h5")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    model_json_path = args.model_json
    model_weight_path = args.model_weight

    # -------------------------------------------------------------------------------------
    test_dir = os.path.join(dataset_path, "test")

    label_names = np.array([x for x in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir,x))])
    label_names.sort()
    num_labels = len(label_names)

    print(num_labels, "labels:\n", label_names)

    # -------------------------------------------------------------------------------------
    BATCH_SIZE = 32
    IMG_SIZE = (128, 101)

    input_shape = IMG_SIZE + (1,)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=False,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    rescale = tf.keras.layers.Rescaling(scale=1./255)
    def rescale_ds(x, y):
        return rescale(x), y

    test_ds = test_ds.map(rescale_ds)

    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    # -------------------------------------------------------------------------------------
    
    # model_weight_path = os.path.join("..", "trained_models", "cvit", "cvit_100-0.952.h5")
    # model_json_path = os.path.join("..", "trained_models", "cvit", "model.json")

    # load model
    with open(model_json_path, "r") as json_file:
        model_json = json_file.read()

    model = model_from_json(model_json)
    print("Model loaded from json ...")

    # compile model
    LEARNING_RATE = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # load weights
    model.load_weights(model_weight_path)
    print("Model weights loaded ...")

    model.summary()

    # evaluate
    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    print(results)