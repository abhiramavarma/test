import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# Define image dimensions (should match what you used for training)
IMG_WIDTH = 224
IMG_HEIGHT = 224

# Define the number of classes (should match your diagnosis_map)
num_classes = 5 # Adjust if you have a different number of classes

# Build the model architecture exactly as you do in app.py
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Print the model summary
model.summary()