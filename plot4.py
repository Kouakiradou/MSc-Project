import tensorflow as tf
import numpy as np
from models import model_secondary
import matplotlib.pyplot as plt

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
# print(x.shape)

# x_train = np.array(x_train)

train_size = int(0.9 * dataset_size)
x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:]
y_test = y[train_size:]
# dataset_secondary_train = dataset_secondary.take(train_size)
# dataset_secondary_test = dataset_secondary.skip(train_size)

model_secondary.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model_secondary.fit(x_train,
                    y_train,
                    batch_size=256,
                    epochs=50,
                    validation_split=0.1)
print(history.history.keys())
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
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
plt.savefig('secondary_loss_epoch.png')

plt.clf()

plt.plot(epochs, accuracy, 'g', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
# plt.show()
plt.savefig('secondary_acc_epoch.png')