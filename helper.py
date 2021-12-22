# print("\nStart importing libraries ...\n")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pickle
import pandas as pd
import random
from skimage import io
import subprocess
import scipy
import scipy.spatial
from tqdm import tqdm
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras import layers 
import json
# subprocess.call("pip install -U opencv-python".split())
# subprocess.call("pip install -U opencv-contrib-python==3.4.2.17".split())

""" ===== Variable description =====
@Modified date: 2021-09-30

db_path (str)       :  path of json file containing all extracted features
target_path (str)   :  path of a searching image 
detector_name (str) : 'sift', 'surf', 'orb', 'resnet50', 'akaze', 'brisk' 
topn (int)          :  top n ranking
"""

def detector_MOD(detector_name):
    chunks = detector_name.split('-')
    if chunks[0] == 'resnet50':
        convbase = ResNet50(weights='imagenet')
        detector = Model(inputs=convbase.input, outputs=convbase.get_layer('avg_pool').output)    
        return detector, None     
    else:
        if chunks[0] == 'sift':
            detector = cv2.xfeatures2d.SIFT_create()
            norm = cv2.NORM_L2
        elif chunks[0] == 'surf':
            detector = cv2.xfeatures2d.SURF_create(800)
            norm = cv2.NORM_L2
        elif chunks[0] == 'orb':
            detector = cv2.ORB_create(400)
            norm = cv2.NORM_HAMMING
        elif chunks[0] == 'akaze':
            detector = cv2.AKAZE_create()
            norm = cv2.NORM_HAMMING
        elif chunks[0] == 'brisk':
            detector = cv2.BRISK_create()
            norm = cv2.NORM_HAMMING
        else:
            return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                                    table_number = 6, # 12
                                    key_size = 12,     # 20
                                    multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm,crossCheck=True)
    return detector, matcher 

class Ranking:
    def __init__(self, target_path, detector_name, topn):
        self.detector_name = detector_name
        self.target_path = target_path
        self.database = pd.read_json('./ResNet50_10K.json')
        self.topn = topn
        # self.vector_size = 64
    
    # Feature extractor
    def extract_features(self, vector_size):
        try:
            image = cv2.imread(self.target_path)  
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            image = self.target_path

        try:
            if self.detector_name == 'resnet50':
                alg = detector_MOD(self.detector_name)[0]
                image = cv2.resize(image, (224,224))
                image = np.array(image, dtype='float')/255.0
                image = np.expand_dims(image, axis=0)
                dsc = alg.predict(image)[0]   
            else:
                # Using KAZE, cause SIFT, ORB and other was moved to additional module
                # which is adding addtional pain during install
                #alg = cv2.KAZE_create()  
                alg = detector_MOD(self.detector_name)[0]
                #alg = init_feature(detector)[0]
                # Dinding image keypoints
                kps = alg.detect(image)
                # Getting first 32 of them. 
                # Number of keypoints is varies depend on image size and color pallet
                # Sorting them based on keypoint response value(bigger is better)
                kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
                # computing descriptors vector
                kps, dsc = alg.compute(image, kps)
                # Flatten all of them in one big vector - our feature vector
                dsc = dsc.flatten()
                # Making descriptor of same size
                # Descriptor vector size is 64
                needed_size = (vector_size * 64)
                if dsc.size < needed_size:
                # if we have less the 32 descriptors then just adding zeros at the
                # end of our feature vector
                    dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        except cv2.error as e:
            print ('Error: ', e)
            return None
        return dsc

    def list2array(self, list_):
        return np.array(list_)

    def getMatrix(self):
        # convert dataframe to matrix 
        copied_db = self.database.copy()
        copied_db[self.detector_name] = copied_db[self.detector_name].apply(lambda x:self.list2array(x))
        matrix = np.array((copied_db[self.detector_name][:].tolist()), dtype="object")
        return matrix
    
    def cos_cdist_MOD(self, matrix, vector):
        # getting cosine distance between search image (vector) and images database (matrix)
        vector = vector.reshape(1, -1)
        dist = scipy.spatial.distance.cdist(matrix, vector, 'cosine').reshape(-1)
        return dist
    
    def match(self):
        feature_vec = self.extract_features(vector_size = 64)    
        db_matrix = self.getMatrix()
        distances = self.cos_cdist_MOD(db_matrix, feature_vec)
        # getting top 5 records
        nearest_ids = np.argsort(distances)[:self.topn].tolist()
        nearest_img = self.database.iloc[nearest_ids][:]
        sending_json = nearest_img.to_json()
        # print('nearest_img-->',sending_json)
        return sending_json #, img_distances[nearest_ids].tolist()

    # def save_output(self, match_df, output_path):
    #     # create output format
    #     match_df = match_df[['Merchant Product ID', 'Merchant Product Name',
    #         'Product URL Web (encoded)','Image URL','Discounted Price',
    #         'Maintype','Subtype']]
    #     match_df.reset_index(drop = True, inplace = True)
    #     return match_df.to_json(output_path, force_ascii=False)