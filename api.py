import uvicorn
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from glob import glob
from tensorflow import keras
from tensorflow.keras.preprocessing import image


class Input(BaseModel):
    model_path: str = './models/mobile_net_100_20201207-090439/mobile_net_100_20201207-090439'
    img_path: str = './dataset/test/PotatoHealthy1.JPG'


# Instantiate FastAPI object
api = FastAPI()

# Allow different origins (https://fastapi.tiangolo.com/tutorial/cors/)
origins = ['http://localhost:5000',
           'https://localhost:5000']

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Expose the prediction functionality
@api.post('/predict')
def predict(params: Input):
    # Get parameters from API call
    data = params.dict()

    # Load model
    model = keras.models.load_model(data["model_path"])

    # Get target label
    target = data["img_path"].split("/")[3].split(".")[0]

    # Load and preprocess image
    img = image.load_img(data["img_path"], target_size=(224, 224))
    input_arr = image.img_to_array(img) / 255
    input_arr = np.array([input_arr])

    # Make prediction
    prediction = model.predict(input_arr)

    # Get prediction label
    prediction_label = sorted([x.split('/')[3] for x in glob("./dataset/train/*")])[np.argmax(prediction)]

    return {'img_path': data["img_path"],
            'target': target,
            'prediction': prediction_label}


if __name__ == '__main__':
    uvicorn.run(api, host='127.0.0.1', port=8000)
