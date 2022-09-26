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

model_edl = tf.keras.models.load_model('saved_model/model_edl', custom_objects={'edl_accuracy':edl_accuracy, 'loss_func':loss_func})


class_idx_str = 'class_idx_str'

def call_model_function(images, call_model_args=None, expected_keys=None):
    target_class_idx =  call_model_args[class_idx_str]
    images = tf.convert_to_tensor(images)
    with tf.GradientTape() as tape:
        if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
            tape.watch(images)
            output_layer = model_edl(images)

            # evidence = tf.math.maximum(output_layer, 0)
            # alpha = evidence +1

            # print(output_layer)
            alpha = output_layer + 1
            # print(alpha)
            u = 10 / tf.reduce_sum(alpha)
            # u = tf.reshape(u, (1,)) # Shoud I?

            gradients = np.array(tape.gradient(100*u, images))
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
          print("wrong keys")

def call_model_function2(images, call_model_args=None, expected_keys=None):
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

# Generate the saliency dataset from primary network and Cifar10
gradient_saliency = saliency.GradientSaliency()
x_whole_set = np.concatenate((x_train, x_test))
y_whole_set = np.concatenate((y_train, y_test))
n = len(x_whole_set)

saliency_maps = []
saliency_maps_evi = []

for i in tqdm.tqdm(range(10)):
   image, lable = x_whole_set[i], y_whole_set[i]
   prediction = model_edl(np.array([image]), training=False)
#    print(prediction)
   predicted_label = np.argmax(prediction[0])
   

   call_model_args = {class_idx_str: predicted_label}
   smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(image, call_model_function, call_model_args, stdev_spread=0.15, nsamples=50)
   smoothgrad_mask_grayscale = saliency.VisualizeImageDiverging(smoothgrad_mask_3d)
#    saliency_maps.append(smoothgrad_mask_3d)
   saliency_maps.append(smoothgrad_mask_grayscale)
   
   smoothgrad_mask_3d2 = gradient_saliency.GetSmoothedMask(image, call_model_function2, call_model_args, stdev_spread=0.15, nsamples=50)
   smoothgrad_mask_grayscale2 = saliency.VisualizeImageDiverging(smoothgrad_mask_3d2)
#    saliency_maps.append(smoothgrad_mask_3d)
   saliency_maps_evi.append(smoothgrad_mask_grayscale2)

lis = [6,7,8]
final = np.array([]).reshape(0,64)
for idx in lis:
    A = saliency_maps[idx]
    im1 = Image.fromarray((A * 255).astype(np.uint8))
    # im1.save("1.jpeg")
    # A = cv2.cvtColor(A,cv2.COLOR_GRAY2BGR)

    B = x_whole_set[idx]
    im2 = Image.fromarray((B * 255).astype(np.uint8))
    # im2.save("2.jpeg")


    C = saliency_maps_evi[idx]
    im3 = Image.fromarray((C * 255).astype(np.uint8))
    # im3.save("3.jpeg")
    # C = cv2.cvtColor(C,cv2.COLOR_GRAY2BGR)


    D = np.concatenate((C,A), axis=1)
    final = np.concatenate((final,D),axis=0)
    
im4 = Image.fromarray((final * 255).astype(np.uint8))
im4.save("4.jpeg")

