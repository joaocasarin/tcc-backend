from flask import Flask, request
from base64 import b64decode
from deepface import DeepFace
from time import time

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return 'Hello World!'

@app.route('/', methods=['POST'])
def index():
    img1 = request.json['img1']
    img2 = request.json['img2']
    
    try:
        img1 = img1.split(',')[1]
    except IndexError:
        pass
    
    try:
        img2 = img2.split(',')[1]
    except IndexError:
        pass
    
    with open('img1.png', 'wb') as f:
        f.write(b64decode(img1))
        
    with open('img2.png', 'wb') as f:
        f.write(b64decode(img2))
    
    try:
        inicio = time()
        result = DeepFace.verify(img1_path='img1.png', img2_path='img2.png', model_name="Facenet", detector_backend="ssd")
        fim = time()
        print(fim - inicio)
        print(result['verified'])
        return { "result": result['verified'] }
    except ValueError:
        return { "result": False }

app.run(host="0.0.0.0", port=5000)