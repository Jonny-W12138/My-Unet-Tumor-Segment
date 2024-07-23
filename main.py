# To ensure GPU is enabled on Colab

# %tensorflow_version 1.x
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))
from model import *
from augmentation import *
from metrics import *
from plots import *
from utils import *
from keras.optimizers import Adam
opt = Adam(learning_rate =1E-6, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
m.compile(loss=dice_coef_loss, optimizer=opt, metrics=['accuracy', iou, F1, recall, precision]) # Keeping track of these metrics