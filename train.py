import os
import random
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# Initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

# Correct folder path to the location of the dataset
dataset_path = r'C:\Users\naban\Downloads\gender_dataset_face'

# Load image files from the dataset (ensure we are getting images from subfolders)
image_files = [f for f in glob.glob(dataset_path + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# Print the number of images found for quick verification
print(f"Total images found: {len(image_files)}")

# Check if image files are found
if len(image_files) == 0:
    raise ValueError("No images found in the dataset. Please check the path and ensure there are images in the folder.")

# Converting images to arrays and labelling the categories
for img in image_files:
    image = cv2.imread(img)
    
    # Resize image to match the required dimensions (96x96)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    # Extract label from folder name (woman or man)
    label = img.split(os.path.sep)[-2]  # e.g., woman or man from path like C:\path\woman\face_1162.jpg
    if label == "woman":
        label = 1  # Label for woman
    else:
        label = 0  # Label for man
    
    labels.append([label])  # Append label to the list

# Print the shape of the dataset (number of images, height, width, channels)
print(f"Data shape: {np.array(data).shape}")
print(f"Labels shape: {np.array(labels).shape}")

# Pre-processing
data = np.array(data, dtype="float") / 255.0  # Normalize images to [0, 1]
labels = np.array(labels)

# Check if data is empty
if data.shape[0] == 0:
    raise ValueError("The data array is empty. No images were loaded correctly.")

# Split dataset into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Check if trainX is empty
if len(trainX) == 0:
    raise ValueError("The training set is empty. Please check the data loading process.")

# One-hot encode the labels
trainY = to_categorical(trainY, num_classes=2)  # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)

# Augmenting the dataset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Define the model
def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":  # Returns a string, either 'channels_first' or 'channels_last'
        inputShape = (depth, height, width)
        chanDim = 1

    # Add layers
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# Build the model
model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

# Compile the model
opt = Adam(learning_rate=lr)  # Fixed Adam optimizer usage
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY),
              steps_per_epoch=len(trainX) // batch_size,
              epochs=epochs, verbose=1)

# Save the model to disk
model.save('gender_detection.h5')

# Plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# Save plot to disk
plt.savefig('plot.png')
