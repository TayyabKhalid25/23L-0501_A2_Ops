import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

def build_baseline_cnn(input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Builds a robust Convolutional Neural Network from scratch.
    Architecture:
      - Rescaling: Normalizes raw pixel values [0, 255] to [0.0, 1.0]
      - Block 1: Conv2D(32) -> BatchNormalization -> ReLU -> MaxPooling2D
      - Block 2: Conv2D(64) -> BatchNormalization -> ReLU -> MaxPooling2D
      - Block 3: Conv2D(128) -> BatchNormalization -> ReLU -> MaxPooling2D
      - Block 4: Conv2D(256) -> BatchNormalization -> ReLU -> MaxPooling2D -> Dropout(0.3)
      - Head: GlobalAveragePooling2D -> Dense(128) -> Dropout(0.4) -> Dense(1, sigmoid)
    """
    inputs = layers.Input(shape=input_shape, name="input_tensor")
    
    # 1. Normalization
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)
    
    # Block 1 (Low-level feature extraction: edges, gradients)
    x = layers.Conv2D(32, (3, 3), padding='same', name="conv_1")(x)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.ReLU(name="relu_1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_1")(x)
    
    # Block 2 (Intermediate features: textures, simple shapes)
    x = layers.Conv2D(64, (3, 3), padding='same', name="conv_2")(x)
    x = layers.BatchNormalization(name="bn_2")(x)
    x = layers.ReLU(name="relu_2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_2")(x)
    
    # Block 3 (Complex textures and object parts)
    x = layers.Conv2D(128, (3, 3), padding='same', name="conv_3")(x)
    x = layers.BatchNormalization(name="bn_3")(x)
    x = layers.ReLU(name="relu_3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_3")(x)
    
    # Block 4 (High-level semantic representations)
    x = layers.Conv2D(256, (3, 3), padding='same', name="conv_4")(x)
    x = layers.BatchNormalization(name="bn_4")(x)
    x = layers.ReLU(name="relu_4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool_4")(x)
    x = layers.Dropout(0.3, name="dropout_block4")(x)
    
    # Classification Head with GlobalAveragePooling to avoid parameter explosion
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4), name="dense_128")(x)
    x = layers.Dropout(0.4, name="dropout_head")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_classification")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_CNN")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

def build_transfer_mobilenet(input_shape=(224, 224, 3), learning_rate=1e-4, trainable_layers=0):
    """
    Builds a Transfer Learning model using MobileNetV2 pretrained on ImageNet.
    - Preprocessing: Maps [0, 255] to [-1, 1] using mobilenet_v2.preprocess_input
    - Backbone: MobileNetV2 (weights='imagenet', include_top=False)
    - Classification Head: GlobalAveragePooling2D -> Dropout(0.4) -> Dense(1, sigmoid)
    """
    inputs = layers.Input(shape=input_shape, name="input_tensor")
    
    # Model-specific preprocessing
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    
    # Pretrained backbone
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze or selectively unfreeze layers
    if trainable_layers == 0:
        base_model.trainable = False
    else:
        base_model.trainable = True
        # Freeze all layers except the last `trainable_layers`
        for layer in base_model.layers[:-trainable_layers]:
            layer.trainable = False
            
    x = base_model(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.4, name="dropout_head")(x)
    outputs = layers.Dense(1, activation='sigmoid', name="output_classification")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Transfer")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model, base_model
