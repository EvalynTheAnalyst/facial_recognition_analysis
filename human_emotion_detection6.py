
# Data Preprocessing & Augmentation

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau 
from tensorflow.keras.optimizers import Adam,SGD
from collections import defaultdict
from tensorflow.keras.applications import MobileNetV2,ResNet50V2
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,classification_report
import os
import shutil
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns



#Dataset path
base_path = r"C:\Users\pavilion14\Downloads\project data  extraction"

#Dictionary to store counts per split and per emotion
split_counts = {
    'train': defaultdict(int),
    'test': defaultdict(int)
}

#Loop through each split
for split in split_counts.keys():
    split_path = os.path.join(base_path, split)
    if not os.path.exists(split_path):
        continue
    for emotion in os.listdir(split_path):
        emotion_path = os.path.join(split_path, emotion)
        if os.path.isdir(emotion_path):
            count = len([
                f for f in os.listdir(emotion_path)
                if os.path.isfile(os.path.join(emotion_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ])
            split_counts[split][emotion] = count

#Display results
total_all = 0
print("Image Count Breakdown Per Split and Class:\n")
for split in ['train', 'test']:
    print(f"🔹 {split.upper()} SPLIT")
    total_split = 0
    for emotion, count in split_counts[split].items():
        print(f"{emotion}: {count} images")
        total_split += count
    total_all += total_split
    print(f"Total in {split}: {total_split} images\n")

print(f"Grand Total Across All Splits: {total_all} images")
#

train_path = r'C:\Users\pavilion14\Downloads\project data extraction\train'

test_path = r'C:\Users\pavilion14\Downloads\project data extraction\test'


#2. Data Preprocessing & Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,            # slightly higher rotation
    width_shift_range=0.15,       # shift up to 15%
    height_shift_range=0.15,      # shift up to 15%
    zoom_range=0.2,               # zoom in/out up to 20%
    shear_range=10,               # slight shear
    horizontal_flip=True,
    fill_mode='nearest' ,          # avoid empty pixels after transform
    validation_split = 0.2
)

test_datagen = ImageDataGenerator(
    rescale =1./255)

val_datagen = ImageDataGenerator(
    rescale =1./255,
    )

train_data = train_datagen.flow_from_directory(
    train_path,
    target_size =(224,224),
    batch_size = 32,
    class_mode = 'sparse',
    shuffle = True,
    subset = "Training"
)

val_data = val_datagen.flow_from_directory(
    train_path,
    target_size =(224,224),
    batch_size = 32,
    class_mode = 'sparse',
    shuffle = True,
    subset = "validation"
)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size =(224,224),
    batch_size = 32,
    class_mode = 'sparse'
)

#first 6 image
image, label = next(train_data)
class_names = list(train_data.class_indices.keys())

plt.figure(figsize=(8,6))
for i in range(6):
   plt.subplot(2, 3, i+1)
   plt.imshow(image[i])
   label_index = int(label[i])  # directly use integer label
   plt.title(f"Label: {class_names[label_index]}")
   plt.axis("off")
plt.tight_layout()
plt.show()

base =ResNet50V2(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(6, activation='softmax')(x)

model = Model(inputs=base.input, outputs=output)

# Unfreeze last 20 layers for fine-tuning
for layer in base.layers[:-20]:
    layer.trainable = False
for layer in base.layers[-20:]:
    layer.trainable = True

optimizer = Adam(learning_rate=1e-4)

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', min_delta = 0,verbose = 1, patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_delta=0.0001),
    ModelCheckpoint('best_model.h5', save_best_only=True)]

history = model.fit(train_data, validation_data=val_data, epochs=50,callbacks=callbacks)

loss, acc = model.evaluate(test_data)
print(f"Test Accuracy: {acc}")

pred = model.predict(test_data)

model.summary()

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

cm = confusion_matrix(test_data.classes, np.argmax(pred,axis=1))
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_data.class_indices)
display.plot(cmap=plt.cm.Blues)
plt.show()

print(classification_report(test_data.classes,np.argmax(pred,axis=1)))

model.save('model.h5')

