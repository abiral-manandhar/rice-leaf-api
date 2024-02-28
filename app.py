from fastapi import FastAPI
app = FastAPI(title='Rice Leaf detection')
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import VGG16, preprocess_input
import uvicorn
from fastapi import FastAPI, UploadFile, File
from io import BytesIO

from PIL import Image

import numpy as np
img_height, img_width = 224,224
model = load_model('rice.h5')

def read_imagefile(file)  -> Image.Image:
    # img_path = file  
    # img = load_img(img_path, target_size=(img_height, img_width))
    # return img
    image = Image.open(BytesIO(file))
    return image
def predict(image: Image.Image):
    img_array = img_to_array(image.resize((224,224)))
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    print(predictions)
    predicted_label = np.argmax(predictions, axis=1)
    classes = ['leaf_blight', 'brown_spot', 'healthy', "leaf_blast", "leaf_scald", 'narrow brown spot']
    class_name = classes[predicted_label[0]]

    return class_name



@app.get('/index')
async def hello_world():
    return "hello world"

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    return prediction
