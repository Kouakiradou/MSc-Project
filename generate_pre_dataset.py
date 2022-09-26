import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from scipy import ndimage
import tqdm
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback

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


model_edl = tf.keras.models.load_model('saved_model/model_edl', custom_objects={'edl_accuracy':edl_accuracy, 'loss_func':loss_func})


# Function to calculate the probability and uncertainty for predicted class
def calc_prob_uncertinty(p):
  
  evidence = np.maximum(p[0], 0)

  alpha = evidence +1

  u = 10/ alpha.sum()
  prob = alpha[np.argmax(alpha)] / alpha.sum()
  return prob, u

# Define the call model function for saliency mapping
class_idx_str = 'class_idx_str'

def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx =  call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output_layer = model_edl(images)
            output_layer = output_layer[:,target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
          print("wrong keys")


# Set the uncertainty threshold
uncertainty_threshold_lower = 0.4
uncertainty_threshold_upper = 0.6

# Generate the saliency dataset from primary network and Cifar10
gradient_saliency = saliency.GradientSaliency()
x_whole_set = np.concatenate((x_train, x_test))
y_whole_set = np.concatenate((y_train, y_test))
print(len(x_whole_set))
print(len(y_whole_set))
n = len(x_whole_set)
saliency_im_true = []
saliency_im_false = []

mis_predicted = 0
cor_predicted = 0

for i in tqdm.tqdm(range(n)):
   image, lable = x_whole_set[i], y_whole_set[i]
   prediction = model_edl(np.array([image]), training=False)
   predicted_label = np.argmax(prediction[0])

   
   prob, u = calc_prob_uncertinty(prediction)

   call_model_args = {class_idx_str: predicted_label}
   smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(image, call_model_function, call_model_args, stdev_spread=0.15, nsamples=50)
  #  smoothgrad_mask_3d = gradient_saliency.GetMask(image, call_model_function, call_model_args)
   smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)

   if np.argmax(lable) != predicted_label:
     saliency_im_false.append(smoothgrad_mask_grayscale)
     mis_predicted = mis_predicted + 1
   else:
     saliency_im_true.append(smoothgrad_mask_grayscale)
     cor_predicted = cor_predicted + 1

print(cor_predicted)
print(mis_predicted)


dataset_size = cor_predicted + mis_predicted


# Make two class balance
np.random.shuffle(saliency_im_true)
saliency_im_true = saliency_im_true[:mis_predicted]

# Turn the saliency maps into tensorflow dataset

x_secondary = np.array(saliency_im_true+saliency_im_false)
y_secondary = np.array([0]*mis_predicted+[1]*mis_predicted)
y_secondary = tf.keras.utils.to_categorical(y_secondary,2)
dataset_secondary = tf.data.Dataset.from_tensor_slices((x_secondary, y_secondary))
dataset_secondary = dataset_secondary.shuffle(dataset_size, reshuffle_each_iteration=True) 

print(len(x_secondary))

print(dataset_secondary.element_spec)

tf.data.experimental.save(dataset_secondary, 'saved_data/saved_pre_data')