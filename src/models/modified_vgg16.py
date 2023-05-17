import tensorflow as tf

def vgg16(input_shape, classes):
    
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    block_config = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
    for block_num in range(len(block_config)):
        for filter_num in range(len(block_config[block_num])):
            filter_size = block_config[block_num][filter_num]
            x = tf.keras.layers.Conv2D(filters=filter_size, kernel_size=(3,3), padding="same", activation="relu", name=f"block{block_num+1}_conv{filter_num+1}")(x)

        x = tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2), name=f"block{block_num+1}_pool")(x)
        x = tf.keras.layers.BatchNormalization(name=f"block{block_num+1}_batchnorm")(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096,activation='relu', name="fc1")(x)
    x = tf.keras.layers.Dense(units=4096,activation='relu', name="fc2")(x)


    predictions = tf.keras.layers.Dense(units=classes, activation="softmax", name="predictions")(x)

    # final model
    model = tf.keras.Model(inputs, outputs=predictions)


    return model

