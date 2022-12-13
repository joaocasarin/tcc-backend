from PIL import Image
import cv2
from base64 import b64decode
from flask import Flask, request

app = Flask(__name__)

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

    img1 = cv2.imread('img1.png')
    img2 = cv2.imread('img2.png')

    # load xml to detect face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2= cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # detect faces in the 2 images
    faces1 = face_cascade.detectMultiScale(gray1, 1.3, 5)
    faces2 = face_cascade.detectMultiScale(gray2, 1.3, 5)
    roi_gray = []
    roi_color = []

    size = gray1.shape

    # crop out only the face of the first and second images
    for (x, y, w, _) in faces1:

        extra = int(w / 6)
        x1 = max(0, x - extra)
        y1 = max(0, y - extra)
        x2 = min(size[1], x1 + 2 * extra + w)
        y2 = min(size[0], y1 + 2 * extra + w)

        img1 = cv2.rectangle(img1, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 4)
        roi_gray.append(gray1[y1:y2, x1:x2])
        roi_color.append(img1[y1:y2, x1:x2])

    if len(faces1) == 0:
        roi_gray.append(gray1)
        roi_color.append(img1)
        
    size = gray2.shape

    for (x, y, w, _) in faces2:

        extra = int(w / 6)
        x1 = max(0, x - extra)
        y1 = max(0, y - extra)
        x2 = min(size[1], x1 + 2 * extra + w)
        y2 = min(size[0], y1 + 2 * extra + w)

        img2 = cv2.rectangle(img2, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 4)
        roi_gray.append(gray2[y1:y2, x1:x2])
        roi_color.append(img2[y1:y2, x1:x2])

    if len(faces2) == 0:
        roi_gray.append(gray2)
        roi_color.append(img2)

    # create a SIFT algorithm variable 
    sift = cv2.SIFT_create()

    # detect descriptors and keypoints to image
    _, descriptors1 = sift.detectAndCompute(roi_gray[0], None)
    _, descriptors2 = sift.detectAndCompute(roi_gray[1], None)

    # compare descriptors by brutal force
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # add similar descriptors to an array
    goodMatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            goodMatches.append([m])

    print(len(matches))
    print(len(goodMatches))
    print(len(matches) * 13/100)

    # verify number of matcher 
    if len(goodMatches) >= (len(matches) * 13 / 100):
        return { "result": True }
    else:
        return { "result": False }