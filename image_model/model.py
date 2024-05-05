import cv2
import os
import random
import numpy as np
import datetime
import onnxruntime
import json


'''Реализуется предсказание модели с помощью библиотеки onnxruntime и OpenCV'''

def get_possible_classes():
    with open('image_model\classes.json') as file:
        data = json.load(file)
    return data

def transform_image(image):
    '''Функция преобразует изображение в нужный формат, размер и нормализует'''

    mean = np.array([0.50707516, 0.48654887, 0.44091784])
    std = np.array([0.26733429, 0.25643846, 0.27615047])

    # Откроем изображение
    # image = Image.open(io.BytesIO(image_bytes))
    # image = cv2.imread(image_bytes)
    image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    
    # Приведем изображение к нужному формату
    image = cv2.resize(image, (32, 32))
    
    # Транспонирование изображения на (3, 32, 32)
    image = np.transpose(image, (2, 0, 1))

    # Нормализация изображения
    img_normalized = (image/255 - mean[:, None, None]) / std[:, None, None]
    
    # приведем массив в формат np.float32
    img_normalized = img_normalized.astype(np.float32)

    # Добавляем размерность для мини-батча
    image = np.expand_dims(img_normalized, axis=0)

    return image



def get_image_class_prediction(image):
    '''
    Ф-я выполняет предсказание класса изображения файлу изображения
    '''
    # создаем сессию на основе модели
    session = onnxruntime.InferenceSession("image_model/model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # преобразуем изображение  
    img = transform_image(image=image)
    
    # выполняем предсказание моделью
    output = session.run([output_name], {input_name: img})

    # получим вероятность предсказания каждого из символов
    result_probability = output[0][0]

    # получим индекс символа который был предсказан с наибольшей вероятностью
    predicted_index = int(result_probability.argmax())
    
    # получим словарь классов изображений и их меток
    classes_dict = get_possible_classes()

    # получим название символа который был предсказан с наибольшей вероятностью
    predicted_class_name = classes_dict[str(predicted_index) ]
    
    return predicted_class_name


def get_result(image_file, is_api=False) -> dict:

    # зафиксируем начальное время
    start_time = datetime.datetime.now()

    print(type(image_file), image_file.shape)

    # выполним предсказание моделью и получим класс изображения (название символа)
    class_name = get_image_class_prediction(image=image_file)
    # Проверка успешности загрузки изображения
    if image_file is not None:
        print("Изображение успешно загружено.")
    else:
        print("Ошибка при загрузке изображения.")
    # зафиксируем время окончания предсказания
    end_time = datetime.datetime.now()

    # вычислим сколько времени затрачено на предсказание в милисекундах
    time_diff = (end_time - start_time)
    execution_time = f'{round(time_diff.total_seconds()*1000)} ms'

    return {
        'predictions': {'class_name': class_name},
        'execution_time': execution_time
    }



if __name__ == '__main__':
    pass