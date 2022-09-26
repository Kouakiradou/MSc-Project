import tensorflow as tf
import numpy as np
from models import model_secondary
from PIL import Image
import cv2
print(tf.__version__)

dataset_secondary = tf.data.experimental.load('saved_data/saved_data',element_spec=(tf.TensorSpec(shape=(32, 32), dtype=tf.float32, name=None), tf.TensorSpec(shape=(2,), dtype=tf.float32, name=None)))
dataset_size = dataset_secondary.cardinality().numpy()

x = []
y = []

for images, labels in dataset_secondary:
  x.append(images.numpy())
  # print(x[-1].shape)
  y.append(labels.numpy())

x = np.array(x)
x = np.reshape(x,(x.shape[0], x.shape[1],x.shape[2],1))
y = np.array(y)

id_list = [1,3,4,7,10,12,13,17,20,22,24,27,32,42,44,46]

idx = 0
final = np.array([]).reshape(0,128,1)
for i in range(4):
    temp = np.array([]).reshape(32,0,1)
    for j in range(4):
        temp = np.concatenate((temp,x[id_list[idx]]), axis=1)
        idx = idx + 1
    final = np.concatenate((final,temp), axis=0)
final = final.reshape(128,128)
final = Image.fromarray((final * 255).astype(np.uint8))
# final = cv2.cvtColor(final,cv2.COLOR_GRAY2BGR)
# im4 = Image.fromarray((final * 255).astype(np.uint8))
final.save("data.jpeg")
    
        