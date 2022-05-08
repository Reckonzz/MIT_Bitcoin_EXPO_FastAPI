import tensorflow as tf
import tensorflow_hub as hub
from transformers import AutoTokenizer

import numpy as np
import PIL.Image
import random
import time
import functools

from fastapi import FastAPI
from pydantic import BaseModel
from utils import *
import json

import ipfsApi

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")

MODEL = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

app = FastAPI()

class ModelInput(BaseModel):
    name: str
    school: str
    linkedIn: str
    github: str

@app.get("/")
async def root():
    return {"message": "test"}

@app.post('/predict/')
async def generateImage(ModelInput: ModelInput):
    input = " ".join(ModelInput.__dict__)

    tiled_tensor, img = tokenise(input)
    tensor_to_image(tiled_tensor)

    questionable_style_image = tf.cast(tf.reshape(tiled_tensor, [1] + tiled_tensor.shape), tf.float32)
    content_path = "/content/raw-img"
    style_path = "/content/musemart/dataset_updated/training_set"
    content_image = load_img(content_path)
    style_image = load_img(style_path)

    plt.subplot(1, 2, 1)
    imshow(content_image, 'Content Image')

    plt.subplot(1, 2, 2)
    imshow(style_image, 'Style Image')

    image = generate_image(questionable_style_image, content_image, style_image)

    # connect to a public IPFS gateway
    api = ipfsApi.Client(host='https://ipfs.infura.io', port=5001)

    result = api.add(image)
    display(result)
    # get the hash of the image from the result variable obtained earlier
    CID = result['Hash']
    # get the image from ipfs
    api.get(CID)

    # load the image locally
    img_PIL = PIL.Image.open(CID)
    # display the image inline
    display(img_PIL)

    ipfsGateway = f"https://ipfs.io/ipfs/{CID}"

    return {
            'name': ModelInput.name,
            'school': ModelInput.school,
            'linkedIn': ModelInput.linkedIn,
            'github': ModelInput.github,
            'image': ipfsGateway
    }

    # input_tensor = tf.constant(tokenizer(input)['input_ids'])
    # input_tensor = tf.reshape(input_tensor, (input_tensor.shape[0], 1, 1))
    # tiled_tensor = tf.tile(input_tensor, tf.constant([430 // input_tensor.shape[0], 522, 3]))

    # content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
    # style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')

    # content_image = load_img(content_path)
    # style_image = load_img(style_path)

    # mask_image = tf.cast(tf.reshape(tiled_tensor, [1] + tiled_tensor.shape), tf.float32)

    # prediction = MODEL(mask_image, style_image)[0]
    # prediction = MODEL(content_image, prediction)[0]