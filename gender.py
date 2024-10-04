import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

print("Starting gender classification model training...")

# Set random seed for reproducibility
np.random.seed(15)
tf.random.set_seed(15)
print("Random seeds set.")

# Set up paths
data_dir = 'Images1/'  # Update this path if necessary
print(f"Data directory set up: {data_dir}")

# Set up image parameters
img_height, img_width = 224, 224
batch_size = 15
print(f"Image parameters set: {img_height}x{img_width}, batch size: {batch_size}")

# Prepare data generators for training and validation sets
print("Preparing data generators...")
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2  # Split data into training and validation
)

# Create data generators
print("Loading training data from directory...")
train_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset='training',  # Set as training data
    classes=['Male', 'Female'],  # Ensure correct mapping of labels
)

print("Loading validation data from directory...")
val_generator = train_datagen.flow_from_directory(
    directory=data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="binary",
    subset='validation',  # Set as validation data
    classes=['Male', 'Female'],  # Ensure correct mapping of labels
)

# Create the model
print("Creating the model...")
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False  # Freeze the base model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
print("Model creation complete.")

# Compile the model
print("Compiling the model...")
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
print("Model compilation complete.")

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
print("Callbacks set up.")

# Train the model
print("Starting model training...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)
print("Model training complete.")

# Save the model
print("Saving the model...")
model.save('gender_classification_model.h5')
print("Model saved as 'gender_classification_model.h5'")

# Optional: Fine-tuning
print("Starting fine-tuning process...")
base_model.trainable = True

print("Recompiling the model for fine-tuning...")
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Fine-tune the model
print("Fine-tuning the model...")
history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)
print("Fine-tuning complete.")

# Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save('gender_classification_model_fine_tuned.h5')
print("Fine-tuned model saved as 'gender_classification_model_fine_tuned.h5'")

print("All processes completed successfully.")
