import cv2
import os
UPLOAD_FOLDER = './Save_image'

for fileImg in os.listdir(UPLOAD_FOLDER):
    path_set = UPLOAD_FOLDER+'/'+fileImg
    img = cv2.imread(UPLOAD_FOLDER+'/'+fileImg)
    print(path_set)
    print(img)