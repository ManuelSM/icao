from PIL import Image
import numpy as np
import onnxruntime as rt


def preprocess_face_count(input_image: Image, resize, crop_size_onnx):
    
    image = input_image.convert("RGB")
    image = image.resize((resize, crop_size_onnx))

    np_image = np.array(image)
    np_image = np_image.transpose(2, 0, 1)

    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])

    norm_img_data = np.zeros(np_image.shape, dtype=np.float32)

    for i in range(np_image.shape[0]):
        norm_img_data[i, :, :] = (np_image[i, :, :] / 255 - mean_vec[i]) / std_vec[i]

    return np.expand_dims(norm_img_data, axis=0)


def get_fd_session(np_input_image: np.array):
    
    if False:
        fd_model_path = "../models/fd.bin"
    else: 
        fd_model_path = "./app/models/fd.bin"

    fd_session = rt.InferenceSession(fd_model_path)
    input_name = fd_session.get_inputs()[0].name
    
    confidences, boxes = fd_session.run(None, {input_name: np_input_image})
    
    return confidences, boxes


def predict_faces(width, height, confidences, boxes, threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]

    picked_box_probs = []
    picked_labels = []

    for class_index in range(1, confidences.shape[1]):
        
        probs = confidences[:, class_index]
        
        mask = probs > threshold
        probs = probs[mask]

        if probs.shape[0] == 0:
            continue
        
        subset_boxes = boxes[mask, :]
        
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        
        box_probs = hard_nms(box_probs, iou_threshold=iou_threshold,top_k=top_k)

        picked_box_probs.append(box_probs)
        
        picked_labels.extend([class_index] * box_probs.shape[0])

    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    
    picked_box_probs = np.concatenate(picked_box_probs)

    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height

    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def face_detector(input_image: Image, threshold = 0.7):
    
    input_width, input_height = input_image.size

    np_image = preprocess_face_count(input_image, 640, 480)

    confidences, boxes = get_fd_session(np_image)

    boxes, labels, probs = predict_faces(input_width, input_height, confidences, boxes, threshold)

    if  probs is None or len(probs) < 1:
        return 0,0
    
    return  len(probs[probs > 0.98]) , int (probs.max() * 100)

