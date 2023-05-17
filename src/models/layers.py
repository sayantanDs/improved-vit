import tensorflow as tf

def generate_patches(inputs, patch_size, patch_overlap=0, hidden_size=None):
    patch_stride = patch_size - patch_overlap
    if hidden_size is None:
        hidden_size = patch_stride * patch_stride

    patches = tf.keras.layers.Conv2D(
        filters=hidden_size, 
        kernel_size=patch_size, 
        strides=patch_stride, 
        padding='valid',
        name='embedding'
    )(inputs)
    
    _, w, h, _ = patches.shape

    # seq_len = (inputs.shape[1] // patch_size) * (inputs.shape[2] // patch_size)
    seq_len = w*h
    x = tf.reshape(patches, [-1, seq_len, hidden_size])
    return x


# taken from https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/vit.py
@tf.keras.utils.register_keras_serializable()
class TokenLayer(tf.keras.layers.Layer):
    """A simple layer to wrap token parameters."""

    def build(self, inputs_shape):
        self.cls = self.add_weight(
            'cls', (1, 1, inputs_shape[-1]), initializer='zeros')

    def call(self, inputs):
        cls = tf.cast(self.cls, inputs.dtype)
        cls = cls + tf.zeros_like(inputs[:, 0:1])  # A hacky way to tile.
        x = tf.concat([cls, inputs], axis=1)
        return x
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
  

@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, inputs_shape):
        pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
        self.pos_embedding = self.add_weight(
            'pos_embedding', 
            pos_emb_shape, 
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02)
        )

    def call(self, inputs, inputs_positions=None):
        # inputs.shape is (batch_size, seq_len, emb_dim).
        pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

        return inputs + pos_embedding
    
    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    


def mlp_block(inputs, mlp_dim, dropout_rate, activation=tf.nn.gelu):
    x = tf.keras.layers.Dense(units=mlp_dim, activation=activation)(inputs)
    if dropout_rate>0:
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)
    x = tf.keras.layers.Dense(units=inputs.shape[-1], activation=activation)(x)
    if dropout_rate>0:
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    return x



def Encoder1Dblock(inputs, num_heads, mlp_dim, dropout_rate, attention_dropout_rate):
    x = tf.keras.layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=inputs.shape[-1], 
        dropout=attention_dropout_rate
    )(x, x) # self attention multi-head
    x = tf.keras.layers.Add()([x, inputs]) # 1st residual part 

    y = tf.keras.layers.LayerNormalization(dtype=x.dtype)(x)
    y = mlp_block(y, mlp_dim, dropout_rate)
    y_1 = tf.keras.layers.Add()([y, x]) #2nd residual part 
    return y_1


def Encoder(inputs, num_layers, mlp_dim, num_heads, dropout_rate, attention_dropout_rate):
    x = AddPositionEmbs(name='posembed_input')(inputs)
    
    if dropout_rate>0:
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    for _ in range(num_layers):
        x = Encoder1Dblock(x, num_heads, mlp_dim, dropout_rate, attention_dropout_rate)

    encoded = tf.keras.layers.LayerNormalization(name='encoder_norm')(x)
    return encoded


def ConvolutionModule(inputs, conv_dropout_rate, activation='relu'):
    x = tf.keras.layers.LayerNormalization(dtype=inputs.dtype)(inputs)

    _, _, hidden_size = inputs.shape

    x = tf.keras.layers.Conv1D(hidden_size*2 , 3, padding="same", activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv1D(hidden_size, 3, padding="same", activation=activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    if conv_dropout_rate>0:
        x = tf.keras.layers.Dropout(rate=conv_dropout_rate)(x)
    
    return x

def Conv_TransformerEncoder1Dblock(inputs, num_heads, mlp_dim, dropout_rate, attention_dropout_rate, conv_dropout_rate, mlp_activation=tf.nn.gelu, conv_activation='relu'):
    # MHSA
    x = tf.keras.layers.LayerNormalization(dtype=inputs.dtype)(inputs)
    x = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=inputs.shape[-1], 
        dropout=attention_dropout_rate
    )(x, x) # self attention multi-head
    x = tf.keras.layers.Add()([x, inputs]) # 1st residual part 
    
    # convolution module
    x1 = ConvolutionModule(x, conv_dropout_rate, conv_activation)
    x1 = tf.keras.layers.Add()([x1, x]) # 2nd residual part 
    
    # MLP
    y = tf.keras.layers.LayerNormalization(dtype=x1.dtype)(x1)
    y = mlp_block(y, mlp_dim, dropout_rate, mlp_activation)
    y_1 = tf.keras.layers.Add()([y, x1]) # 3rd residual part 
    
    return y_1


def Conv_TransformerEncoder(inputs, num_layers, mlp_dim, num_heads, dropout_rate, attention_dropout_rate, conv_dropout_rate, mlp_activation=tf.nn.gelu, conv_activation='relu'):
    x = AddPositionEmbs(name='posembed_input')(inputs)
    
    if dropout_rate>0:
        x = tf.keras.layers.Dropout(rate=dropout_rate)(x)

    for _ in range(num_layers):
        x = Conv_TransformerEncoder1Dblock(x, num_heads, mlp_dim, dropout_rate, attention_dropout_rate, conv_dropout_rate, mlp_activation, conv_activation)

    encoded = tf.keras.layers.LayerNormalization(name='encoder_norm')(x)
    return encoded