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

model_edl = tf.keras.models.load_model('saved_model/model_edl')

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
uncertainty_threshold = 0.4

# Generate the saliency dataset from primary network and Cifar10
gradient_saliency = saliency.GradientSaliency()
x_whole_set = np.concatenate((x_train, x_test))
y_whole_set = np.concatenate((y_train, y_test))
print(len(x_whole_set))
print(len(y_whole_set))
n = len(x_whole_set)
saliency_im_certain = []
saliency_im_uncertain = []

for i in tqdm.tqdm(range(n)):
   image, lable = x_whole_set[i], y_whole_set[i]
   prediction = model_edl(np.array([image]), training=False)
   predicted_label = np.argmax(prediction[0])
   
   prob, u = calc_prob_uncertinty(prediction)

   call_model_args = {class_idx_str: predicted_label}
   smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(image, call_model_function, call_model_args, stdev_spread=0.15, nsamples=50)

   if u > uncertainty_threshold:
     saliency_im_uncertain.append(smoothgrad_mask_3d)
   else:
     saliency_im_certain.append(smoothgrad_mask_3d)

# split the dataset by the prediction correctness instead

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

tf.data.experimental.save(dataset_secondary, 'saved_data/saved_data')