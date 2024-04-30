import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
import pandas as pd
from sklearn.utils import shuffle
import os

# Load the Ham10000 dataset
ham_df = pd.read_csv(r'D:/Visual Studio Code/Projects/Skin_Cancer_Detection/Files/HAM/HAM10000_metadata.csv')

# Preprocess the data
ham_df = shuffle(ham_df)  # Shuffle the data
ham_df['image_path'] = r'D:/Visual Studio Code/Projects/Skin_Cancer_Detection/Files/Net/' + ham_df['image_id'] + '.jpg'


# Check if image files exist
missing_images = ham_df[~ham_df['image_path'].apply(os.path.exists)]
if not missing_images.empty:
    print("Missing images:")
    print(missing_images)
    ham_df = ham_df.drop(missing_images.index)  # Drop rows with missing images
    ham_df.reset_index(drop=True, inplace=True)  # Reset index after dropping rows
else:
    print("All image files exist.")

# Check sample distribution
class_distribution = ham_df['dx'].value_counts()
print("Class distribution:")
print(class_distribution)


# Define image dimensions and batch size
img_height, img_width = 224, 224
batch_size = 32

# Filter out invalid image filenames
valid_image_paths = []
for img_path in ham_df['image_path']:
    if os.path.exists(img_path):
        valid_image_paths.append(img_path)

# Update DataFrame with valid image paths
ham_df = ham_df[ham_df['image_path'].isin(valid_image_paths)]


# Define data generator with data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_dataframe(
    dataframe=ham_df,
    x_col="image_path",
    y_col="dx",
    subset="training",
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(img_height, img_width)
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=ham_df,
    x_col="image_path",
    y_col="dx",
    subset="validation",
    batch_size=batch_size,
    seed=42,
    shuffle=True,
    class_mode="categorical",
    target_size=(img_height, img_width)
)


# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7 classes in Ham10000 dataset
])


# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Check the number of samples in training and validation sets
print("Number of training samples:", train_generator.samples)
print("Number of validation samples:", validation_generator.samples)

# Calculate steps per epoch
# Set minimum value for steps_per_epoch
steps_per_epoch_train = max(train_generator.samples // batch_size, 1)
steps_per_epoch_val = max(validation_generator.samples // batch_size, 1)

print("Steps per epoch (train):", steps_per_epoch_train)
print("Steps per epoch (validation):", steps_per_epoch_val)


# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_val
)


# Evaluate the model
loss, accuracy = model.evaluate(validation_generator, verbose=0)
print("Validation Accuracy: {:.2f}%".format(accuracy * 100))

# Define the path to save the model weights
model_path = r'D:/Visual Studio Code/Projects/Skin_Cancer_Detection/Files/Trained Models/ham10000_cnn_model.h5'

# Save model weights
model.save(model_path)
