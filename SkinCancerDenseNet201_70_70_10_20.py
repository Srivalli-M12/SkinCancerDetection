import tensorflow as tf
tf.test.gpu_device_name()
from google.colab import drive
drive.mount('/content/drive')
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
!pip install np_utils
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
import seaborn as sns
import pprint as pp

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import np_utils

import itertools

import cv2
from PIL import Image
np.random.seed(42)

image_path1 = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join('drive/My Drive/SkinCancerDataset/', '*', '*.jpg'))}
image_path2 = {os.path.splitext(os.path.basename(x))[0]: x for x in glob(os.path.join('drive/My Drive/SkinCancerDataset/HAM10000_images_part_1/', '*', '*.jpg'))}
image_path1.update(image_path2)

print(image_path1)
lesion_type_dict = {
    'cancer':'Cancer',
    'non_cancer':"Not cancer"
}
skin_data = pd.read_csv(os.path.join('drive/My Drive/SkinCancerDataset/HAM10000_metadata_.csv'))

skin_data['path'] = skin_data['image_id'].map(image_path1.get)
skin_data['cell_type'] = skin_data['dx'].map(lesion_type_dict.get)
skin_data['cell_type_idx'] = pd.Categorical(skin_data['cell_type']).codes

skin_data.head()

def balanced_dataset(df):
    balanced = pd.DataFrame()
    for x in df['cell_type_idx'].unique():
        sample = resample(df[df['cell_type_idx'] == x],
                        replace=True,     # sample with replacement
                        n_samples=5000,   # to match majority class
                        random_state=123) # reproducible results
        # Combine majority class with upsampled minority class
        balanced = pd.concat([balanced, sample])

    balanced['cell_type'].value_counts()

    return balanced

def load_img_data(df, balanced=False):
    np.random.seed(42)
    images = []
    if balanced:
        df = balanced_dataset(df)
    image_paths = list(df['path'])

    for i in tqdm(range(len(image_paths))):
        image = cv2.imread(image_paths[i])
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.
        images.append(image)

    images = np.stack(images, axis=0)
    print(images.shape)

    label=df['cell_type_idx'].values
    trainx, testx, trainy, testy = train_test_split(images, label, test_size=0.30,random_state=42)
    testx, valx, testy, valy = train_test_split(testx, testy, test_size=0.667,random_state=42)
    print(len(trainx), len(valx), len(testx))
    return (trainx, trainy, valx, valy, testx, testy)

image_data = load_img_data(skin_data, balanced=True)

from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt

class BuildModel(object):

    def __init__(self, data, deep_model, layers_, batch_size, epoch):
        self.shape = data[0][0].shape
        self.base = deep_model
        self.net = self.download_network()
        self.net.trainable = False

        self.layers_ = layers_
        self.classes = 2
        self.trainx = data[0]
        self.trainy = data[1]
        self.valx = data[2]
        self.valy = data[3]
        self.testx = data[4]
        self.testy = data[5]

        self.epoch = epoch
        self.batch_size = batch_size

        self.model = self.build()
        self.predictions = None
        self.score = None

        self.best_weight = None

    def download_network(self):

        net = None

        if self.base == 'DenseNet201':
            net = tf.keras.applications.DenseNet201(weights='imagenet', include_top=False, input_shape=(128,128,3))
        elif self.base == 'DenseNet169':
            net = tf.keras.applications.DenseNet169(weights='imagenet', include_top=False, input_shape=self.shape)
        elif self.base == 'DenseNet121':
            net = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=self.shape)
        elif self.base == 'ResNet152':
            net = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=self.shape)
        return net
    def run(self):

        self.fine_tune()

    def build(self):

        model = tf.keras.models.Sequential()
        model.add(self.net)
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.15))

        for x in self.layers_:
            model.add(tf.keras.layers.Dense(x, activation='relu'))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Dropout(0.36))

        model.add(tf.keras.layers.Dense(self.classes, activation='softmax'))
        print (model.summary())

        return model

    def load_weights(self, name):

        self.model.load_weights(name)

    def get_class_weight(self):

        weight_0 = (1 / (self.trainy == 0).sum())*(len(self.trainy))/2.0
        weight_1 = (1 / (self.trainy == 1).sum())*(len(self.trainy))/2.0

        classes_weight = {
            0: weight_0,
            1: weight_1,
        }


        return classes_weight

    def fine_tune(self):

        trainingSamples = self.trainx.shape[0]
        validationSamples = self.valx.shape[0]

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

        best_weights = self.base + "_weights.hdf5"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(best_weights, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

        history = self.model.fit(
            self.trainx, self.trainy,
            steps_per_epoch=trainingSamples // self.batch_size,
            epochs=self.epoch,
            validation_data=(self.valx, self.valy),
            validation_steps=validationSamples // self.batch_size,
            #class_weight=self.get_class_weight(),
            callbacks=[checkpoint])

        best_acc = max(history.history["val_accuracy"])

        self.load_weights(best_weights)

        self.predict()

        self.plotting_loss(history, self.epoch )


    def predict(self):
        print("\n")
        self.score = self.model.evaluate(self.testx, self.testy, verbose=0)
        print('Test score:', self.score)
        self.predictions = self.model.predict(self.testx, batch_size=self.batch_size)
        print("\n")

    def plotting_loss(self, history, epochs):

        plt.figure(figsize=(6,6))
        plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
        plt.plot(np.arange(0, epochs), history.history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, epochs), history.history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()
deep_model = 'DenseNet201'
layers_ = [128,128]
batch_size = 128
epoch = 70

model = BuildModel(image_data, deep_model, layers_, batch_size, epoch)
model.run()

predictions = model.predictions
y_pred =[]
for x in predictions:
  y_pred.append(np.argmax(x))
print(confusion_matrix(model.testy, y_pred))
cm = confusion_matrix(model.testy, y_pred)
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]
cm = cm_norm * 100
sns.heatmap(cm, annot=True)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion matrix")
plt.show()

print(classification_report(model.testy, y_pred))

model.model.save('drive/MyDrive/Final/skin_densemodel_201_epoch70_70_10_20.h5')

import numpy as np
import cv2

def test_image(model, image_path):

  image = cv2.imread(image_path)
  plt.imshow(image)
  image = cv2.resize(image, (128, 128))
  image = image.astype(np.float32) / 255.
  image = np.stack(image, axis=0)
  image = np.array([image])
  prediction = model.predict(image)
  predicted_class = np.argmax(prediction)
  return predicted_class

model1 = tf.keras.models.load_model('drive/MyDrive/Final/skin_densemodel_201_epoch70_70_10_20.h5')

image_path = 'drive/My Drive/SkinCancerDataset/HAM10000_images_part_1/ISIC_0028343.jpg'
p_class = test_image(model1, image_path)

if p_class==0:
  label='Cancer'
else:
  label='Not Cancer'
print("Predicted class label:", label)


image_path = 'drive/My Drive/SkinCancerDataset/HAM10000_images_part_1/HAM10000_images_part_2/ISIC_0034026.jpg'
p_class = test_image(model1, image_path)

if p_class==0:
  label='Cancer'
else:
  label='Not Cancer'
print("Predicted class label:", label)

image_path = 'drive/My Drive/SkinCancerDataset/HAM10000_images_part_1/ISIC_0024316.jpg'
p_class = test_image(model1, image_path)

if p_class==0:
  label='Cancer'
else:
  label='Not Cancer'
print("Predicted class label:", label)
