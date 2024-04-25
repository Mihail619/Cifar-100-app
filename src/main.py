from fastapi import FastAPI

app = FastAPI()


@app.get("/image_class_prediction")
async def image_class_prediction():
    return 
    ...

@app.get("/image_class_prediction_api/")
async def image_class_prediction():
    ...