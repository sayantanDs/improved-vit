import tensorflow as tf
from .layers import generate_patches, TokenLayer, Conv_TransformerEncoder


def cvit(
        input_shape,
        classes,
        patch_size=16, 
        patch_overlap=8, 
        hidden_size=64, 
        num_transformer_layers=12,
        num_heads=12,
        mlp_dim=256,
        dropout_rate=0.5, 
        attention_dropout_rate=0.2,
        conv_dropout_rate=0.2,
        mlp_activation=tf.nn.gelu,
        conv_activation='swish'
    ):

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    x = tf.keras.layers.Conv2D(
        filters=8, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation="relu",
        name="conv_features_1"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"conv_features_1_batchnorm")(x)
    
    x = tf.keras.layers.Conv2D(
        filters=8, 
        kernel_size=(3, 3), 
        strides=(1, 1), 
        padding='same',
        activation="relu",
        name="conv_features_2"
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"conv_features_2_batchnorm")(x)
    

    # Create patches.
    x = generate_patches(
        x,
        patch_size, 
        patch_overlap,
        hidden_size
    )

    # Add CLS token
    x = TokenLayer(name='cls')(x)

    # Transformer encoder blocks
    x = Conv_TransformerEncoder(
        x,
        num_transformer_layers, 
        mlp_dim, 
        num_heads, 
        dropout_rate, 
        attention_dropout_rate,
        conv_dropout_rate,
        mlp_activation,
        conv_activation
    )

    # take only the CLS token output
    x = x[:, 0]


    predictions = tf.keras.layers.Dense(classes, name='predictions', activation='softmax')(x)

    # final model
    model = tf.keras.Model(inputs=inputs, outputs=predictions)   
    
    return model