import cv2
import io
import numpy as np
import datetime
import onnxruntime


'''Реализуется предсказание модели с помощью библиотеки onnxruntime и OpenCV'''

def transform_image(image_bytes):
    '''Функция преобразует изображение в нужный формат, размер и нормализует'''

    mean = np.array([0.50707516, 0.48654887, 0.44091784])
    std = np.array([0.26733429, 0.25643846, 0.27615047])

    # Откроем изображение
    # image = Image.open(io.BytesIO(image_bytes))
    image = cv2.imread(image_bytes)
    
    # Приведем изображение к нужному формату
    image = cv2.resize(image, (32, 32))
    
    # Транспонирование изображения на (3, 32, 32)
    image = np.transpose(image, (2, 0, 1))

    # Нормализация изображения
    img_normalized = (image/255 - mean[:, None, None]) / std[:, None, None]
    
    # приведем массив в формат np.float32
    img_normalized = img_normalized.astype(np.float32)

    return img_normalized



def get_image_class_prediction(image_bytes):
    '''
    Ф-я выполняет предсказание класса изображения файлу изображения
    '''
    # создаем сессию на основе модели
    session = onnxruntime.InferenceSession("image_model/model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # преобразуем изображение  
    img = transform_image(image_bytes=image_bytes,
                          )
    
    # выполняем предсказание моделью
    output = session.run([output_name], {input_name: img})

    # получим вероятность предсказания каждого из символов
    result_probability = output[0][0]


if __name__ == '__main__':
    with open('static\images\M.png', 'rb') as f:
        result = get_image_class_prediction(f.read())
    # print(result)