from PIL import Image
import onnxruntime as rt
import numpy as np

from app.utils.image_utils import preprocess_image_accessories


def softmax_stable(x):
    return(np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


def detect_mask(cropped_input_image:Image)-> float:

    if False:
        path_to_msk = "../models/msk.bin"
    else:
        path_to_msk = "./app/models/msk.bin"

    msk_session = rt.InferenceSession(path_to_msk)

    input_name = msk_session.get_inputs()[0].name
    output_name = msk_session.get_outputs()[0].name

    sess_input = msk_session.get_inputs()
    sess_output = msk_session.get_outputs()

    for idx, input_ in enumerate(range(len(sess_input))):
        input_name = sess_input[input_].name
        input_shape = sess_input[input_].shape
        input_type = sess_input[input_].type

    for idx, output in enumerate(range(len(sess_output))):
        output_name = sess_output[output].name

    np_image = preprocess_image_accessories(cropped_input_image, input_shape[2], input_shape[3])

    result_model = msk_session.run([output_name], {input_name: np_image})

    return softmax_stable(result_model[0].flatten())[0]
