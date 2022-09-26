import numpy as np
import tqdm
import tensorflow as tf
import saliency.core as saliency
from tensorflow.keras import backend as K
from PIL import Image
import cv2

def edl_accuracy(yTrue, yPred):
    pred = K.argmax(yPred, axis=1)
    truth = K.argmax(yTrue, axis=1)
    match = K.reshape(K.cast(K.equal(pred, truth), "float32"),(-1,1))
    return K.mean(match)

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

model_regular = tf.keras.models.load_model('saved_model/model_regular')
model_edl = tf.keras.models.load_model('saved_model/model_edl', custom_objects={'edl_accuracy':edl_accuracy, 'loss_func':loss_func})


class_idx_str = 'class_idx_str'


def call_model_function_edl(images, call_model_args=None, expected_keys=None):
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

def call_model_function_regular(images, call_model_args=None, expected_keys=None):
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

# Generate the saliency dataset from primary network and Cifar10
gradient_saliency = saliency.GradientSaliency()
x_whole_set = np.concatenate((x_train, x_test))
y_whole_set = np.concatenate((y_train, y_test))
n = len(x_whole_set)



# idx = 1

# image = x_train[idx]
# label = y_train[idx]

# prediction = model_regular(np.array([image]), training=False)
# predicted_label = np.argmax(prediction[0])
# call_model_args = {class_idx_str: predicted_label}
# smoothgrad_mask_3d_regular = gradient_saliency.GetSmoothedMask(image, call_model_function_regular, call_model_args, stdev_spread=0.15, nsamples=50)
# smoothgrad_mask_grayscale_regular = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d_regular)

# prediction = model_edl(np.array([image]), training=False)
# predicted_label = np.argmax(prediction[0])
# call_model_args = {class_idx_str: predicted_label}
# smoothgrad_mask_3d_edl = gradient_saliency.GetSmoothedMask(image, call_model_function_edl, call_model_args, stdev_spread=0.15, nsamples=50)
# smoothgrad_mask_grayscale_edl = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d_edl)

# # image = Image.fromarray((image * 255).astype(np.uint8))
# # smoothgrad_mask_grayscale_edl = Image.fromarray((smoothgrad_mask_grayscale_edl * 255).astype(np.uint8))
# smoothgrad_mask_grayscale_edl = cv2.cvtColor(smoothgrad_mask_grayscale_edl,cv2.COLOR_GRAY2BGR)
# origin_with_saliency_edl = np.concatenate((image,smoothgrad_mask_grayscale_edl), axis=1)
# origin_with_saliency_edl = Image.fromarray((origin_with_saliency_edl * 255).astype(np.uint8))
# origin_with_saliency_edl.save("origin_with_saliency_edl_1.jpeg")

# smoothgrad_mask_grayscale_regular = cv2.cvtColor(smoothgrad_mask_grayscale_regular,cv2.COLOR_GRAY2BGR)
# origin_with_saliency_all = np.concatenate((image,smoothgrad_mask_grayscale_edl,smoothgrad_mask_grayscale_regular), axis=1)
# origin_with_saliency_all = Image.fromarray((origin_with_saliency_all * 255).astype(np.uint8))
# origin_with_saliency_all.save("origin_with_saliency_all_1.jpeg")


final = np.array([]).reshape(0,96,3)

list = [11,12,14]
not_list = [2,3,5,7]
print(y_train[[1,4,6]])
# for idx in list:
#     image = x_train[idx]
#     label = y_train[idx]

#     prediction = model_regular(np.array([image]), training=False)
#     predicted_label = np.argmax(prediction[0])
#     call_model_args = {class_idx_str: predicted_label}
#     smoothgrad_mask_3d_regular = gradient_saliency.GetSmoothedMask(image, call_model_function_regular, call_model_args, stdev_spread=0.15, nsamples=50)
#     smoothgrad_mask_grayscale_regular = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d_regular)

#     prediction = model_edl(np.array([image]), training=False)
#     predicted_label = np.argmax(prediction[0])
#     call_model_args = {class_idx_str: predicted_label}
#     smoothgrad_mask_3d_edl = gradient_saliency.GetSmoothedMask(image, call_model_function_edl, call_model_args, stdev_spread=0.15, nsamples=50)
#     smoothgrad_mask_grayscale_edl = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d_edl)

#     # image = Image.fromarray((image * 255).astype(np.uint8))
#     # smoothgrad_mask_grayscale_edl = Image.fromarray((smoothgrad_mask_grayscale_edl * 255).astype(np.uint8))
#     smoothgrad_mask_grayscale_edl = cv2.cvtColor(smoothgrad_mask_grayscale_edl,cv2.COLOR_GRAY2BGR)
#     # origin_with_saliency_edl = np.concatenate((image,smoothgrad_mask_grayscale_edl), axis=1)
#     # origin_with_saliency_edl = Image.fromarray((origin_with_saliency_edl * 255).astype(np.uint8))
#     # origin_with_saliency_edl.save("origin_with_saliency_edl_1.jpeg")

#     smoothgrad_mask_grayscale_regular = cv2.cvtColor(smoothgrad_mask_grayscale_regular,cv2.COLOR_GRAY2BGR)
#     origin_with_saliency_all = np.concatenate((image,smoothgrad_mask_grayscale_edl,smoothgrad_mask_grayscale_regular), axis=1)
#     # origin_with_saliency_all.save("origin_with_saliency_all_1.jpeg")

#     final = np.concatenate((final,origin_with_saliency_all),axis=0)

# final = Image.fromarray((final * 255).astype(np.uint8))
# final.save("final.jpeg")