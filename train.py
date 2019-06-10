#! /home/me/software/anaconda3/envs/pl_face_rec1/lib/python2.7
#-*-coding:utf-8-*-
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
import face_recognition as fr
import sys

from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn.neighbors import KNeighborsClassifier
import sys

# def train(train_dir, model_save_path='trained_knn_model.clf', n_neighbors=3, knn_algo='ball_tree'):
train_dir = './train'
model_save_path='./trained_knn_model.clf'
n_neighbors=3
knn_algo='ball_tree'

x = []
y = []

for class_dir in os.listdir(train_dir):
    if not os.path.isdir(os.path.join(train_dir, class_dir)):
        continue

    print('start training files ' + class_dir)

    for image_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
        image = fr.load_image_file(image_path)
        boxes = fr.face_locations(image)

        x.append(fr.face_encodings(image, known_face_locations=boxes)[0])
	#print(x)
        y.append(class_dir)
        print('开始 start training photos')

if n_neighbors is None:
    n_neighbors = 3

knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
knn_clf.fit(x, y)


if model_save_path is not None:
    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)
# return knn_clf
