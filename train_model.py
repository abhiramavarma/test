import pandas as pd
import os

# --- USER CONFIGURATION ---
# Path to your train.csv file in Google Drive
csv_path = 'train.csv' # *CHANGE THIS PATH*

# Path to the directory containing your image folders (Mild, Moderate, etc.)
image_dir = 'gaussian_filtered_images/' # *CHANGE THIS PATH*

# Mapping from numerical diagnosis to category name
diagnosis_map = {
    0: 'No_DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferate_DR'
}
# --- END USER CONFIGURATION ---

# Load the training data
df = pd.read_csv(csv_path)

# Map numerical diagnosis to category names
df['diagnosis_category'] = df['diagnosis'].map(diagnosis_map)

# Create the full path to each image
df['image_path'] = df['id_code'].apply(lambda x: os.path.join(image_dir, df[df['id_code'] == x]['diagnosis_category'].iloc[0], f'{x}.png'))

# Display the first few rows with the new columns
print(df.head())

from sklearn.model_selection import train_test_split

# Split the DataFrame
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# Define image dimensions
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32 # You can adjust this based on your GPU memory

# Create ImageDataGenerators for training and validation
# We'll use data augmentation for the training set to improve generalization
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values to [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescale for validation set (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# Flow images from DataFrame
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path', # Column with image file paths
    y_col='diagnosis_category', # Column with class labels
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Use 'categorical' for one-hot encoded labels
    seed=42
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='diagnosis_category',
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=42
)

# Get the class indices (mapping from category name to index)
class_indices = train_generator.class_indices
print("Class Indices:", class_indices)

# After calling train_datagen.flow_from_dataframe(...)

print("\n--- Generator Information ---")
print(f"Number of samples in train_df: {len(train_df)}")
print(f"Number of samples found by train_generator: {train_generator.samples}")

# Access num_classes via the length of class_indices
if hasattr(train_generator, 'class_indices'):
    print(f"Number of classes detected by train_generator: {len(train_generator.class_indices)}")
    print("Class indices detected by train_generator:", train_generator.class_indices)
else:
    print("train_generator does not have a 'class_indices' attribute.")

print("--- End Generator Information ---")

# Also, double-check the unique values in your target column
print("\nUnique values in train_df['diagnosis_category']:")
print(train_df['diagnosis_category'].value_counts())

from sklearn.utils.class_weight import compute_class_weight
import numpy as np # Make sure numpy is imported

# Get the class labels from the training DataFrame
train_labels = train_df['diagnosis_category']

# Get the unique class names in sorted order
sorted_class_names = sorted(train_generator.class_indices.keys(), key=lambda x: train_generator.class_indices[x])

# Calculate class weights
class_weights_array = compute_class_weight(
    'balanced',
    classes=np.array(sorted_class_names), # <--- CHANGE THIS LINE
    y=train_labels
)

# Create a dictionary mapping class index to class weight
class_weight_dict = dict(enumerate(class_weights_array))

print("Calculated Class Weights:")
print(class_weight_dict)
index_to_name = {v: k for k, v in train_generator.class_indices.items()}
print("Class Index to Name Mapping:")
print(index_to_name)

from tensorflow.keras.applications import DenseNet121 # type: ignore
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

# Load DenseNet121 pre-trained on ImageNet, without the top classification layer
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Add custom classification layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x) # Reduce spatial dimensions
x = Dense(512, activation='relu')(x) # Add a dense layer
x = Dropout(0.5)(x) # Add dropout for regularization
predictions = Dense(len(diagnosis_map), activation='softmax')(x) # Output layer with softmax for classification

# Define the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the weights of the pre-trained layers initially (optional but common)
# This helps in training the new layers before fine-tuning the whole model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy', # Use categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

model.summary()

EPOCHS = 1 # Or more epochs

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    class_weight=class_weight_dict # Add this line
)

# ... rest of the code (saving weights)

# Optional: Save the trained model weights
model.save_weights('diabetic_retinopathy_classification_weights.weights.h5')