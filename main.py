import helper 
import os, sys
from time import time
import cv2
# import config

import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './Save_image'

app = Flask(__name__) 
cors = CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/sendingmodel', methods = ['POST']) 
def modelApp():
    file = request.files['file']
    print("data ==> ",file)
    filename = secure_filename(file.filename)
    print("filename ==> ", filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    
    for fileImg in os.listdir(UPLOAD_FOLDER):
        if fileImg == filename:
            img = cv2.imread(UPLOAD_FOLDER+'/'+fileImg)
            result_out = main(img = img, detector ='resnet50',rank= 5, addName=fileImg)
            # print('sucess !!', result_out)

    return result_out
 

def main( img = None, detector = None, rank = None, addName=None):
    tic = time()
    print("\n Furesign is searhing your image ...")
    print(addName)
    Rank = helper.Ranking(img,  detector, rank)
    output = Rank.match()
    json_out = json.loads(output)
    # To save output, Merchant Product ID, Merchant Product Name, Product URL Web (encoded), Image URL
    print('\n Finished saving output!')
    toc = time()
    # print(addName)
    # print(f"Searching time: {toc-tic:.2f} seconds")
 
    # print('debug route resultproduct -->',json_out['Merchant Product ID'])
    data = json_out

    return data 

def string(path):
    return rf'{path}'
    

if __name__ == '__main__':
    # app.debug = True
    app.run(port=6060)






