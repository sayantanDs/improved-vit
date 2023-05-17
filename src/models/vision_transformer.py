import tensorflow as tf
from .layers import generate_patches, TokenLayer, Encoder


def vit(
        input_shape,
        classes,
        patch_size=16, 
        patch_overlap=8, 
        hidden_size=64, 
        num_transformer_layers=12,
        num_heads=12,
        mlp_dim=256,
        dropout_rate=0.2, 
        attention_dropout_rate=0.2
    ):

    inputs = tf.keras.layers.Input(shape=input_shape)

    # Create patches.
    x = generate_patches(
        inputs,
        patch_size, 
        patch_overlap,
        hidden_size
    )

    # Add CLS token
    x = TokenLayer(name='cls')(x)

    # Transformer encoder blocks
    x = Encoder(
        x,
        num_transformer_layers, 
        mlp_dim, 
        num_heads, 
        dropout_rate, 
        attention_dropout_rate
    )

    # take only the CLS token output
    x = x[:, 0]


    predictions = tf.keras.layers.Dense(classes, name='predictions', activation='softmax')(x)

    # final model
    model = tf.keras.Model(inputs=inputs, outputs=predictions)   
    
    return model