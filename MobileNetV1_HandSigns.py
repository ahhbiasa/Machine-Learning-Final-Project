
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.optimizers import RMSprop,Adam ,Nadam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Model
import pandas as pd
import itertools
import shutil
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from PIL import Image

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv")
test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv")

train.head()
print(sorted(train.label.unique()))

train_sample = train.loc[:499,:]
train_sample.shape

train_sample.head

valid_sample = train.iloc[-200:,:]
valid_sample.shape
test_sample = test.loc[:199,:]
test_sample.shape

train_sample.to_csv("train_sample.csv",index=False)
test_sample.to_csv("test_sample.csv",index=False)
valid_sample.to_csv("valid_sample.csv",index=False)

!python ../input/how-to-convert-csv-to-images/make_imgs.py --label label ./test_sample.csv ./mnist-imgs/sample/test/ 
!python ../input/how-to-convert-csv-to-images/make_imgs.py --label label ./train_sample.csv ./mnist-imgs/sample/train/ 
!python ../input/how-to-convert-csv-to-images/make_imgs.py --label label ./valid_sample.csv mnist-imgs/sample/valid/

train_path = "./mnist-imgs/sample/train"
test_path = "./mnist-imgs/sample/test"
valid_path = "./mnist-imgs/sample/valid"


train_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)

valid_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)

test_data = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

mobil = tf.keras.applications.mobilenet.MobileNet()

mobil.summary()



take = mobil.layers[-6].output
drop = tf.keras.layers.Dropout(0.2)(take)
output = Dense(units=24,activation="softmax")(drop)

model = Model(inputs=mobil.input, outputs= output)

print(len(model.trainable_variables))
print(len(model.non_trainable_variables))

for layer in model.layers[:-23]:
    layer.trainable =False

print(len(model.trainable_variables))
print(len(model.non_trainable_variables))

model.summary()

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4,
    decay_steps=10000,
    decay_rate=0.9)


optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

for layer in model.layers[:-23]:
    layer.trainable =True
    
model.compile(optimizer=keras.optimizers.SGD(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])


from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# Train the model with early stopping
history = model.fit(
    x=train_data,
    validation_data=valid_data,
    epochs=100,  # Set a large number of epochs
    verbose=2,
    callbacks=[early_stopping]
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

test_label = test_data.classes
prediction = model.predict(x=test_data, verbose=2)
cm=confusion_matrix(y_true=test_label, y_pred=prediction.argmax(axis=1))

test_data.class_indices
cm_label=[str(x) for x in range(24 + 1)]
cm_label

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,width=10,height=10):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(width,height))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


plot_confusion_matrix(cm=cm,classes=cm_label, title ="CONFUSION MATRIX")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


classes=cm_label
y_true=test_label
y_pred=prediction.argmax(axis=1)

confusion = confusion_matrix(y_true, y_pred)
#print('Confusion Matrix\n')
#importing accuracy_score, precision_score, recall_score, f1_score

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))
print('Micro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='micro')))
print('Micro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='micro')))
print('Micro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='micro')))

print('Macro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='macro')))
print('Macro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='macro')))
print('Macro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='macro')))

print('Weighted Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='weighted')))
print('Weighted Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='weighted')))
print('Weighted F1-score: {:.2f}'.format(f1_score(y_true, y_pred, average='weighted')))

from sklearn.metrics import classification_report
from PIL import Image

classes = ["Class " + str(i) for i in range(25) if i != 9]
print(classification_report(y_true, y_pred, target_names=classes))
test_indices = [0, 12, 25, 30]

plt.figure(figsize=(15, 7))

for i, idx in enumerate(test_indices):
    plt.subplot(1, 5, i + 1)
    img_path = test_data.filepaths[idx]
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"True: {chr(y_true[idx] + 65)}\nPred: {chr(y_pred[idx] + 65)}")
    plt.axis('off')

plt.show()