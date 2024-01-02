from fastapi import FastAPI
from pydantic import BaseModel

from PIL import Image
from io import BytesIO
import base64
import warnings
import onnxruntime as rt

from app.utils.detect_faces import face_detector
from app.utils.image_utils import crop_face
from app.utils.detect_accessories import detect_glass_hat
from app.utils.detect_mask import detect_mask

rt.set_default_logger_severity(3)

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.simplefilter("ignore")

app = FastAPI()

# Clases de Request y Response para consumir API
class ImageBase(BaseModel):
    imageData: str


class ResponseModel(BaseModel):
    prediction: bool
    score: float
    details: str
    status: int
    
# Endpoint para consumir PersonalPicPilot, evasion y accesorios
@app.post("/evasion", response_model=ResponseModel)
def evasion(input_image: ImageBase):

    image_data = input_image.imageData

    # Decodifación de la imagen de request en base64 a PIL.Image()
    try:
        image = Image.open(BytesIO(base64.b64decode(image_data)))
    except Exception as error:
        return {
            "details": f"Error: error on decode image, {error}",
            "score": 0.0,
            "prediction": True,
            "status": 1
        }
    
    # Detección de rostro de imagen de request
    try:
        num_faces, qa = face_detector(image)
    except Exception as error:
        return {
            "details": f"Error: Face Detection error, {error}",
            "score": 0.0,
            "prediction": True,
            "status": 2
        }

    if num_faces == 0:
        return {
            "prediction": True,
            "score": qa/100,
            "details": "Evasion: not front face detected",
            "status": 0
        }
        
    
    if qa < 99:
        return {
            "prediction": True,
            "score": qa/100,
            "details": "Evasion: bad pose 0 detected",
            "status": 0
        }  
        
    
    if num_faces > 1:
        return {
            "prediction": True,
            "score": 1-(1/num_faces),
            "details": "Evasion: multiple face detected",
            "status": 0
        } 
        

    # Si solo se detectó un rostro entonces cortamos imagen
    try:
        cropped_face, distances_xyz = crop_face(image)
    except Exception as error:
        return {
            "details": f"Error: crop face error, {error}",
             "score": 0.0,
            "prediction": True,
            "status": 3
        }

    dist = abs(distances_xyz[0] - distances_xyz[1])

    if distances_xyz[0] < 80 :
        return {
            "prediction": True,
            "score": 1/distances_xyz[0],
            "details": "Evasion: bad pose 1 detected",
            "status": 0
        } 
        
    if distances_xyz[0] > 200 :
        return {
            "prediction": True,
            "score": 1-(1/distances_xyz[0]),
            "details": "Evasion: bad pose 2 detected",
            "status": 0
        }
        
    if distances_xyz[1] < 80 :
        return {
            "prediction": True,
            "score": 1/distances_xyz[1],
            "details": "Evasion: bad pose 3 detected",
            "status": 0
        }
        
    if  distances_xyz[1] > 200:
        return {
            "prediction": True,
            "score": 1-(1/distances_xyz[1]),
            "details": "Evasion: bad pose 4 detected",
            "status": 0
        }
        
    if dist > 22:
        return {
            "prediction": True,
            "score": 1-(1/dist),
            "details": "Evasion: bad pose 5 detected",
            "status": 0
        }
        
    if abs(distances_xyz[2] ) > 12:
        return {
            "prediction": True,
            "score": 1-(1/abs(distances_xyz[2])),
            "details": "Evasion: bad pose 6 detected",
            "status": 0
        }
        
    # Deteccion de accesorios (lentes, mascarilla y sombrero)
    try:
        glasses, hat = detect_glass_hat(cropped_face)
    except Exception as error:
        return {
            "details": f"Error: Accessories GH error, {error}",
            "score": 0.0,
            "prediction": True,
            "status": 4
        }

    if glasses > 0:
        return {
            "prediction": True,
            "score": glasses,
            "details": "Evasion: glasses detected",
            "status": 0
        }
        
    if hat > 0:
        return {
            "prediction": True,
            "score": hat,
            "details": "Evasion: hide top head detected",
            "status": 0
        }
       
    # Detección de mascarillas
    try:
        score = detect_mask(cropped_face)
    except Exception as error:
        return {
            "details": f"Error: Accessories Mask error, {error}",
            "score": 0.0,
            "prediction": True,
            "status": 4
        }

    if score > 0.72:
        return {
            "prediction": True,
            "score": round(float(score), 3),
            "details": "Evasion: hide mouth detected",
            "status": 0
        }
    
    # Si todo está bien regresamos OK con score -1.0 flotante    
    return {
            "prediction": False,
            "score": -1.0,
            "details": "Evasion: OK",
            "status": 0
        }
