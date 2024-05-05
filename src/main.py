from typing import Annotated
from fastapi import FastAPI, Form, File, UploadFile, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np

from image_model import model

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")  


@app.get("/static/styles.css")
async def styles():
    response = Response(content=open("styles.css").read(), media_type="text/css")
    response.headers["Cache-Control"] = "public, max-age=10" 
    return response

@app.get("/")
async def read_home(request: Request):
    return templates.TemplateResponse(request=request,
                                      name="index.html"
                                      )

@app.post("/")
async def post_home(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.fromstring(contents, np.uint8)
    
    result = model.get_result(np_img)
    return templates.TemplateResponse(request=request,
                                      name="index.html", context={"result": result})

@app.post("/image_class_predict")
async def post_home(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    np_img = np.fromstring(contents, np.uint8)
    
    result = model.get_result(np_img)
    return result