from typing import Annotated
from fastapi import FastAPI, Form, File, UploadFile
import numpy as np

from image_model import model

app = FastAPI()


@app.post("/image_class_prediction")
async def image_class_prediction(file: UploadFile = File(...)):
    '''
    Ф-я выполняет предсказание класса изображения файлу изображения
    '''
    contents = await file.read()
    np_img = np.fromstring(contents, np.uint8)
    
    image_class = model.get_result(np_img)
    return {"class is": image_class}
    

@app.post("/image_class_prediction_api/")
async def image_class_prediction():
    ...