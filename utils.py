import tensorflow as tf 

import io
import os
import re
import shutil
import string
import tensorflow as tf
import random

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM
import tensorflow_hub as hub
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def tokenise(sentence):
  tensor = tf.constant(tokenizer(sentence)['input_ids'])
  arr = tensor.numpy()
  dim = 500 * 522 * 3
  mul = (dim // arr.shape[0]) + 1
  arr = np.repeat(arr, mul)
  arr = arr[:dim]
  arr = np.random.permutation(arr)
  tensor = tf.constant(arr)
  tensor = tf.reshape(tensor, ([500,522,3])) 
  tiled_tensor = tensor % 255

  img = tensor_to_image(tiled_tensor)
  return tiled_tensor, img

def load_img(path_to_img):
  parent_folder = os.listdir(path_to_img)
  folder = random.choice(parent_folder)
  path = path_to_img + "/" + folder
  images = os.listdir(path)
  img = random.choice(images)
  path_to_img = path + "/" + img

  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

def generate_art(questionable_style_image, content_image, style_image):
  stylized_image = hub_model(questionable_style_image, tf.constant(style_image))[0]
  alien_splatter = hub_model(tf.constant(content_image), stylized_image)[0]
  return alien_splatter

def generate_image(questionable_style_image, content_image, style_image):
  art = generate_art(questionable_style_image, content_image, style_image)
  art = tensor_to_image(art)
  im1 = art.save("art.png", "PNG")
  return im1