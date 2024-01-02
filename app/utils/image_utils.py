# from app.utils.path_points import points

from PIL import Image, ImageDraw
import onnxruntime as rt
import numpy as np
import math


def preprocess_image(input_image: Image, resize:tuple=(192,192)):

    input_image = input_image.resize(resize)

    np_image = np.array(input_image)

    norm_img_data = (np_image - np.min(np_image)) / (np.max(np_image) - np.min(np_image))
    norm_img_data = np.array(norm_img_data, dtype=np.float32)

    return np.expand_dims(norm_img_data, axis=0)


def preprocess_image_accessories(input_image:Image, resize:int, crop_size:int):

    input_image = input_image.convert("RGB")
    input_image = input_image.resize((resize, resize))

    left   = (resize - crop_size) / 2
    top    = (resize - crop_size) / 2
    right  = (resize + crop_size) / 2
    bottom = (resize + crop_size) / 2

    input_image = input_image.crop((left, top, right, bottom))

    np_image = np.array(input_image)

    np_image = np_image.transpose(2, 0, 1)

    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])

    norm_img_data = np.zeros(np_image.shape, dtype=np.float32)
    
    for i in range(np_image.shape[0]):
        norm_img_data[i, :, :] = (np_image[i, :, :] / 255 - mean_vec[i]) / std_vec[i]

    return np.expand_dims(norm_img_data, axis=0)


def paint_face_mesh(input_image:Image, Xs, Ys, points):
    
    drawImage = input_image
    img1 = ImageDraw.Draw(input_image)
    for k in range(0, len(points), 2):
        i = points[k]
        j = points[k + 1]
        shape = [(int(Xs[i]), int(Ys[i])), (int(Xs[j]), int(Ys[j]))]
        img1.line(shape, fill="cyan", width=0)

    drawImage.show()


def get_landmarks(landmark_list:list, isX:bool=True)->list:
    if isX:
        return landmark_list[::3]
    else: 
        return landmark_list[1::3]


def get_face_mesh(input_image: Image) -> list:
    
    if False:
        fm_path = "../models/fm.bin"
    else:
        fm_path = "./app/models/fm.bin"

    session_mesh = rt.InferenceSession(fm_path)
    
    input_name = session_mesh.get_inputs()[0].name
    output_name = session_mesh.get_outputs()[0].name

    np_image_array = preprocess_image(input_image)

    result_list = session_mesh.run([output_name], {input_name: np_image_array})
    
    return result_list[0].flatten()


def crop_face(input_image: Image) -> Image: 
    
    input_width, input_height = input_image.size

    complete_landmarks = get_face_mesh(input_image)


    Xs = get_landmarks(complete_landmarks)
    Ys = get_landmarks(complete_landmarks, False)

    Xs = [((value / 192.0)*input_width) for value in Xs]
    Ys = [((value / 192.0)*input_height) for value in Ys]

    x = min(Xs)
    y = min(Ys)

    w = max(Xs)
    h = max(Ys)

    dist = (h - y) * 0.65

    x = int(Xs[5] - (dist))
    y = int(Ys[5] - (dist))
    w = int(Xs[5] + (dist))
    h = int(Ys[5] + (dist))

    if x < 0:
        x = 0

    if y < 0:
        y = 0

    if  w >= input_width - 1:
        w = input_width - x - 1

    if  h >= input_height - 1:
        h = input_height - y - 1


    output_image:Image = input_image.crop((int(x), int(y), int(w), int(h)))


    ref = distance(Xs[133], Ys[133], Xs[362], Ys[362])

    distances = []

    points = [1, 133, 1, 362]

    for k in range(0, len(points), 2):
        i = points[k]
        j = points[k + 1]

        distances.append((distance(Xs[i], Ys[i], Xs[j], Ys[j]) / ref) * 100)

    angleInRadian = math.atan2(Ys[362] - Ys[133], Xs[362] - Xs[133])
    angleInDegree = angleInRadian * 180 / math.pi
    distances.append(angleInDegree)


    return output_image, distances


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

