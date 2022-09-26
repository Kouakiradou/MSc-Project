import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage
import tqdm
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt

import saliency.core as saliency

from models import model_edl

print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Download and process the original dataset for primary network
# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

x_train = x_train.astype('float32')
x_train = x_train / 255.0

x_test = x_test.astype('float32')
x_test = x_test / 255.0

y_train = tf.keras.utils.to_categorical(y_train,10)
y_test = tf.keras.utils.to_categorical(y_test,10)


# Callback function for controling the epoch num
ep = 1

class GetEpochs(Callback):
    def on_epoch_end(self, epoch, logs={}):
        global ep
        ep += 1

# # Function to calculate the accuracy for EDL networks
def edl_accuracy(yTrue, yPred):
    pred = K.argmax(yPred, axis=1)
    truth = K.argmax(yTrue, axis=1)
    match = K.reshape(K.cast(K.equal(pred, truth), "float32"),(-1,1))
    return K.mean(match)

# # Define the loss function
lgamma = tf.math.lgamma
digamma = tf.math.digamma

def KL(alpha, num_classes=10):
  one = K.constant(np.ones((1,num_classes)),dtype=tf.float32)
  S = K.sum(alpha,axis=1,keepdims=True)  

  kl = lgamma(S) - K.sum(lgamma(alpha),axis=1,keepdims=True) +\
      K.sum(lgamma(one),axis=1,keepdims=True) - lgamma(K.sum(one,axis=1,keepdims=True)) +\
      K.sum((alpha - one)*(digamma(alpha)-digamma(S)),axis=1,keepdims=True)
          
  return kl


def loss_func(y_true, output):
    y_evidence = K.relu(output)
    alpha = y_evidence+1
    S = K.sum(alpha,axis=1,keepdims=True)
    p = alpha / S  

    err = K.sum(K.pow((y_true-p),2),axis=1,keepdims=True)
    var = K.sum(alpha*(S-alpha)/(S*S*(S+1)),axis=1,keepdims=True)
    
    l =  K.sum(err + var,axis=1,keepdims=True)
    l = K.sum(l)
    
    
    kl =  K.minimum(1.0, ep/10) * K.sum(KL((1-y_true)*(alpha)+y_true))
    return l + kl


# Compile and fit the model
model_edl.compile(loss=loss_func, optimizer="adam", metrics=edl_accuracy)

batch_size = 1024
history = model_edl.fit(x_train, 
              y_train,
          batch_size=batch_size,
          epochs=50,
          validation_split=0.1,
          callbacks=[GetEpochs(),])


# loss_train = history.history['train_loss']
# loss_val = history.history['val_loss']
print(history.history.keys())
accuracy = history.history['edl_accuracy']
val_accuracy = history.history['val_edl_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(0,50)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.show()
plt.savefig('loss_epoch.png')

plt.clf()

plt.plot(epochs, accuracy, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('acc_epoch.png')