import os
import shutil
import json
import argparse
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'
import tensorflow as tf

from models import vit, cvit, vgg16
from utils import CustomModelCheckPoint, get_prev_save_file_name, get_prev_best_save_file_name




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dataset_path', default="speech_commands_v2_spectrograms")
    parser.add_argument('-o', '--output_path', default="save_files")
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('--save_folder')    # same as model if not specified

    parser.add_argument('-e', '--epochs', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=32, type=int)
    parser.add_argument('-l', '--learning_rate', default=0.001, type=float)
    parser.add_argument('--early_stopping', default=10, type=float)
    parser.add_argument('-c', '--cpu', action='store_true')  # on/off flag
    parser.add_argument('--set_gpu_growth', action='store_true')  # on/off flag
    args = parser.parse_args()

    dataset_path = args.dataset_path        # "speech_commands_v2_spectrograms"
    output_path = args.output_path          # "save_files"
    model_choice = args.model               # model choice
    save_folder = args.save_folder          # will be same as model name
    
    BATCH_SIZE = args.batch_size            # 32
    NUM_EPOCHS = args.epochs                # 100
    LEARNING_RATE = args.learning_rate      # 0.001
    EARLY_STOPPING = args.early_stopping    # 10
    EARLY_STOPPING = None if EARLY_STOPPING==0 else EARLY_STOPPING
    
    USE_CPU = args.cpu                      # False
    SET_GPU_GROWTH = args.set_gpu_growth    # False

    save_folder = model_choice.lower() if save_folder is None else save_folder

    print("dataset_path =", dataset_path)
    print("output_path =", output_path)
    print("model_choice =", model_choice)
    print("save_folder =", save_folder)
    print()
    print("BATCH_SIZE =", BATCH_SIZE)
    print("NUM_EPOCHS =", NUM_EPOCHS)
    print("LEARNING_RATE =", LEARNING_RATE)
    print("EARLY_STOPPING =", EARLY_STOPPING)
    print()
    print("USE_CPU =", USE_CPU)
    print()
    print("Tensorflow version ", tf. __version__)
    print()


    # ============================== [ GPU SETTINGS ] ======================================

    gpu_devices = tf.config.list_physical_devices('GPU')
    if(len(gpu_devices)>0):
        print(gpu_devices)
        if SET_GPU_GROWTH:
            tf.config.experimental.set_memory_growth(gpu_devices[0], True)
    elif not USE_CPU:
        print("No GPU devices found!")
        exit(1)

    
    # ============================== [ CREATE DATASET ] ======================================
    
    train_dir = os.path.join(dataset_path, "train")
    test_dir = os.path.join(dataset_path, "test")
    val_dir = os.path.join(dataset_path, "val")

    # -------------------------------------------------------------------------------------
    label_names = np.array([x for x in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,x))])
    label_names.sort()
    num_labels = len(label_names)

    print(num_labels, "labels:\n", label_names)

    # -------------------------------------------------------------------------------------
    # BATCH_SIZE = 32
    IMG_SIZE = (128, 101)

    input_shape = IMG_SIZE + (1,)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        shuffle=True,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        shuffle=True,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        shuffle=True,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )

    # -------------------------------------------------------------------------------------
    rescale = tf.keras.layers.Rescaling(scale=1./255)
    def rescale_ds(x, y):
        return rescale(x), y

    train_ds = train_ds.map(rescale_ds)
    test_ds = test_ds.map(rescale_ds)
    val_ds = val_ds.map(rescale_ds)

    # -------------------------------------------------------------------------------------

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    # test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    # -------------------------------------------------------------------------------------

    model_save_loc =  os.path.join(output_path, save_folder)
    print("Model save location:", model_save_loc)
    os.makedirs(model_save_loc, exist_ok=True)

    # ============================== [ CREATE MODEL ] ======================================

    if(model_choice.lower() == "vit"):
        model = vit(input_shape, num_labels)
    elif(model_choice.lower() == "cvit"):
        model = cvit(input_shape, num_labels)
    elif(model_choice.lower() == "modified_vgg16"):
        model = vgg16(input_shape, num_labels)
    else:
        model_func = getattr(tf.keras.applications, model_choice)
        model = model_func(weights=None, input_shape=input_shape, classes=num_labels)

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    if not os.path.isfile(os.path.join(model_save_loc, "model.json")):
        model_json = model.to_json()
        with open(os.path.join(model_save_loc, "model.json"), "w") as json_file:
            json_file.write(model_json)
        print("Saved model json ...")

    if not os.path.isfile(os.path.join(model_save_loc, "model_summary.txt")):
        with open(os.path.join(model_save_loc, "model_summary.txt"), 'w') as f:
            with redirect_stdout(f):
                model.summary()
        print("Saved model summary ...")
        
    # ============================== [ TRAIN MOEL ] ======================================
    
    # To continue from previous run
    prev_save_file = get_prev_save_file_name(model_save_loc)
    prev_best_file = get_prev_best_save_file_name(model_save_loc)
    prev_epoch = 0
    prev_best_acc = 0

    if prev_save_file:
        print("Last best save file: ", prev_best_file)
        prev_best_acc = float(prev_best_file[-13:-8])
        print("Last best acc: ", prev_best_acc)

        print("Last save file: ", prev_save_file)
        prev_epoch = int(prev_save_file[-12:-9])
        print("prev epoch:", prev_epoch)

        print("Loading weights...")
        load_status = model.load_weights(os.path.join(model_save_loc,prev_save_file))
        # load_status.assert_consumed()

    custom_checkpoint_callback = CustomModelCheckPoint(
                                    model_save_loc,
                                    prev_save_file=prev_save_file,
                                    prev_best_file=prev_best_file,
                                    prev_best_acc=prev_best_acc,
                                    model_name=model_choice.lower()
                                )
    
    csv_logger_callback = tf.keras.callbacks.CSVLogger(
        os.path.join(model_save_loc, "logs.csv"), separator=',', append=True
    )

    # ---------------------- MODEL FIT --------------------------------    
    num_epochs = NUM_EPOCHS

    callbacks=[
            custom_checkpoint_callback,
            csv_logger_callback
        ]
    if EARLY_STOPPING is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping( monitor='val_loss', patience=EARLY_STOPPING, verbose=1))
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs, 
        initial_epoch=prev_epoch,
        callbacks=callbacks,
    )
    

    # ============================== [ EVALUATE MOEL ] ======================================

    results = model.evaluate(test_ds, verbose=1, return_dict=True)
    print(results)

    with open(os.path.join(model_save_loc, "evaluate.json"), 'w') as f:
        json.dump(results, f)