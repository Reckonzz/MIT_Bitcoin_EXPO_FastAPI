import tensorflow as tf

import numpy as np
import PIL.Image
import time
import functools

from fastapi import FastAPI
from pydantic import BaseModel

import tensorflow_hub as hub
MODEL = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


app = FastAPI()

class UserInput(BaseModel):
    name: string
    school: string
    linkedIn: string
    github: string

@app.get('/')
async def index():
    return {"Test": "Test"}

@app.post('/predict/')
async def generateImage(UserInput: UserInput):
    # PSEUDO CODE HERE
    # randomize from images folder
    # vectorize input from user input 
    # Create image matrix from vectorized input 
    # pass image matrix to the model 
    
    prediction = MODEL.predict(INPUT_IMAGE)
    return {"prediction": float(prediction)}