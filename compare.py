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

from models import model_regular

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



# Compile and fit the model
model_regular.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

batch_size = 1024
model_regular.fit(x_train, 
              y_train,
          batch_size=batch_size,
          epochs=50,
          validation_split=0.1,)


# # Save the model
# # os.mkdir('saved_model')
model_regular.save('saved_model/model_regular2')


# If it is not the first time training this model, we can load the parameter from pickle
# model_edl = tf.keras.models.load_model('saved_model/model_edl', custom_objects={'edl_accuracy':edl_accuracy, 'loss_func':loss_func})

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
            output_layer = model_regular(images)
            output_layer = output_layer[:,target_class_idx]
            gradients = np.array(tape.gradient(output_layer, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
          print("wrong keys")




# Set the uncertainty threshold
prob_threshold_lower = 0.7
# prob_threshold_upper = 0.6§

# Generate the saliency dataset from primary network and Cifar10
gradient_saliency = saliency.GradientSaliency()
x_whole_set = np.concatenate((x_train, x_test))
y_whole_set = np.concatenate((y_train, y_test))
print(len(x_whole_set))
print(len(y_whole_set))
n = len(x_whole_set)
saliency_im_certain = []
saliency_im_uncertain = []

mis_predicted = 0

for i in tqdm.tqdm(range(n)):
   image, lable = x_whole_set[i], y_whole_set[i]
   prediction = model_regular(np.array([image]), training=False)
   predicted_label = np.argmax(prediction[0])

   if np.argmax(lable) != predicted_label:
     mis_predicted = mis_predicted + 1
     continue
   

   call_model_args = {class_idx_str: predicted_label}
   smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(image, call_model_function, call_model_args, stdev_spread=0.15, nsamples=50)
   smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
   
   prob = np.max(prediction[0])

   if prob < prob_threshold_lower:
     saliency_im_uncertain.append(smoothgrad_mask_grayscale)
   elif prob > prob_threshold_lower:
     saliency_im_certain.append(smoothgrad_mask_grayscale)

print(mis_predicted)

certain_num = len(saliency_im_certain)
uncertain_num = len(saliency_im_uncertain)

dataset_size = certain_num + uncertain_num

print(certain_num)
print(uncertain_num)

# Make two class balance
np.random.shuffle(saliency_im_certain)
saliency_im_certain = saliency_im_certain[:uncertain_num]

# Turn the saliency maps into tensorflow dataset

x_secondary = np.array(saliency_im_certain+saliency_im_uncertain)
y_secondary = np.array([0]*uncertain_num+[1]*uncertain_num)
y_secondary = tf.keras.utils.to_categorical(y_secondary,2)
dataset_secondary = tf.data.Dataset.from_tensor_slices((x_secondary, y_secondary))
dataset_secondary = dataset_secondary.shuffle(dataset_size, reshuffle_each_iteration=True) 

print(dataset_secondary.element_spec)

tf.data.experimental.save(dataset_secondary, 'saved_data/saved_data_regular')


