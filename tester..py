import os, sys
import cv2

path = './images_test'
files_test = os.listdir(path)    
img = cv2.imread(path + '/' + files_test[0])
cv2.imshow('0',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(files_test)