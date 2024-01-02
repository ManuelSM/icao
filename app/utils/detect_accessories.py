from PIL import Image
import onnxruntime as rt
import numpy as np

from app.utils.image_utils import preprocess_image_accessories


def face_parsing(input_image: Image):
    
    if False:
        path_to_fp = "../models/fp.bin"
    else:
        path_to_fp = "./app/models/fp.bin"

    fp_session = rt.InferenceSession(path_to_fp)

    input_name = fp_session.get_inputs()[0].name
    output_name = fp_session.get_outputs()[0].name

    input_session = fp_session.get_inputs()
    ouput_session = fp_session.get_outputs()

    for idx, input_ in enumerate(range(len(input_session))):
        input_name = input_session[input_].name
        input_shape = input_session[input_].shape

    for idx, output in enumerate(range(len(ouput_session))):
        output_name = ouput_session[output].name


    np_image = preprocess_image_accessories(input_image, input_shape[2], input_shape[3])

    return fp_session.run([output_name], {input_name: np_image})


def detect_glass_hat(cropped_input_image:Image):
    
    # resize input image 
    cropped_input_image = cropped_input_image.resize((512, 512), Image.BICUBIC)
    result_model = np.array(face_parsing(cropped_input_image)[0])

    parsing = result_model.squeeze(0).argmax(0)

    glasses = np.count_nonzero(parsing == 6) - np.count_nonzero(parsing == 5) - np.count_nonzero(parsing == 4)
    hat = np.count_nonzero(parsing == 18) - np.count_nonzero(parsing == 17) - np.count_nonzero(parsing == 5) - np.count_nonzero(parsing == 4) - np.count_nonzero(parsing == 3) - np.count_nonzero(parsing == 2)


    return  (1 - (1 / glasses)) if ( glasses > 0  ) else 0, (1 - (1 / hat )) if ( hat > 0 ) else 0
