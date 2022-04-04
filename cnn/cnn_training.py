#!/usr/bin/env python
# coding: utf-8

# # U-Net Convolutional Neural Network (CNN)
# ## Segmentation of images of colloidal particles
# 
# ->Train using updated generation script
# Ross Carmichael  // adapted from TensorFlow example
# 12/10/21

import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing 

from tensorflow_examples.models.pix2pix import pix2pix

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

from glob import glob
import pickle
import os
import numpy as np
import cv2

BATCH_SIZE = 64
BUFFER_SIZE = 1000

def load_data(path):
    images = sorted(glob(os.path.join(path, "particles/*.png")))
    masks = sorted(glob(os.path.join(path, "masks/*.png")))

    return images, masks

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (128, 128))
    image = image / 255.0
    image = image.astype(np.float32)

    return image

def read_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (128, 128))
    
    # Class labels {0, 1, 2}
    mask = np.round(mask * (2/255), 0)
    mask = np.expand_dims(mask, axis=-1)
    mask = mask.astype(np.float32)
    
    return mask

def preprocess(x, y):
    def f(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)

        return x, y

    images, masks = tf.numpy_function(f, [x, y], [tf.float32, tf.float32])
    images.set_shape([128, 128, 3])
    masks.set_shape([128, 128, 1])

    return images, masks

def tf_dataset(x, y, batch=BATCH_SIZE, buffer_size=BUFFER_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size, seed=2)
    dataset = dataset.map(preprocess)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    
    return dataset

def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1):
    assert (train_split + test_split + val_split) == 1
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    test_size = int(test_split * ds_size)

    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size).take(test_size)
    
    return train_ds, val_ds, test_ds

path = "../preprocessing/data/augmented/"

# File paths
images, masks = load_data(path)
print(f"Images: {len(images)} - Masks: {len(masks)}")

x = read_image(images[0])
y = read_mask(masks[0])

dataset = tf_dataset(images, masks)

assert len(images) == len(masks), "Number of images and segmentation masks is not equal."
ds_size = len(images)

_train_ds, _val_ds, _test_ds = get_dataset_partitions_tf(dataset, ds_size)

# Dataset containing subsets for train, val and test
dataset = {
           "train" :      _train_ds, 
           "validation" : _val_ds,
           "test" :       _test_ds
          }

train_batches = dataset["train"].batch(BATCH_SIZE)
test_batches = dataset["test"].batch(BATCH_SIZE)
validation_batches = dataset["validation"].batch(BATCH_SIZE)


SIZE_TRAIN = len(dataset["train"])
SIZE_VALIDATION = len(dataset["validation"])
SIZE_TEST = len(dataset["test"])
STEPS_PER_EPOCH = SIZE_TRAIN // BATCH_SIZE

print(f"Batch size:       {BATCH_SIZE}")
print(f"Buffer size:      {BUFFER_SIZE}")
print(f"Training batches: {len(train_batches)}")
print(f"Test batches:     {len(test_batches)}")
print(f"Steps per epoch:  {STEPS_PER_EPOCH}\n")

print(f"Train size:       {SIZE_TRAIN}")
print(f"Validation size:  {SIZE_VALIDATION}")
print(f"Test size:        {SIZE_TEST}")

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x)


OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

print(model.summary())

EPOCHS = 100
VAL_SUBSPLITS = 2
VALIDATION_STEPS = SIZE_TEST//BATCH_SIZE//VAL_SUBSPLITS

checkpoint_path = "checkpoint/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.save_weights(checkpoint_path.format(epoch=0))

model_history = model.fit(train_batches, epochs=EPOCHS,
                          #steps_per_epoch=STEPS_PER_EPOCH,
                          #validation_steps=VALIDATION_STEPS,
                          validation_data=validation_batches, # should be validation data
                          callbacks=[cp_callback])

model.save("saved_model/model_4_100_epoch.h5")
pickle.dump(model_history, open("saved_model/model_4_100_epoch_history.pickle", "wb"))

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.savefig("training_and_validation_loss_4_100_epochs.eps")
plt.legend()
plt.close()

acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy']

plt.figure()
plt.plot(model_history.epoch, acc, 'r', label='Training accuracy')
plt.plot(model_history.epoch, val_acc, 'bo', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy Value')
plt.ylim([0, 1])
plt.legend()
plt.savefig("training_and_validation_accuracy_4_100_epochs.eps")

print(model_history.history["accuracy"][-1])
