import cv2
import os
import numpy as np
import faceRecognition as fr

test_img=cv2.imread('TestImages/frame37.jpg')#test_img path
faces_detected,gray_img=fr.faceDetection(test_img)
print("faces_detected:",faces_detected)

faces,faceID=fr.labels_for_training_data('trainingImages')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.write('trainingData.yml')

cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows





