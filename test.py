#! /home/me/software/anaconda3/envs/pl_face_rec1/lib/python2.7
#-*-coding:utf-8-*-
import math
from sklearn import neighbors
import os
import os.path
import pickle
import cv2
from PIL import Image, ImageDraw
import face_recognition
import face_recognition as fr
import sys

from face_recognition.face_recognition_cli import image_files_in_folder
from sklearn.neighbors import KNeighborsClassifier
import sys



# prediction=predict(full_file_path,model_path='trained_knn_model.clf')
def predict(x_img_path, knn_clf=None, model_path=None, distance_threshold=0.4):
    if knn_clf is None and model_path is None:
        raise Exception('knn_clf or model_path')

    # print('debugpoint0')
    # sys.exit(0)
    if knn_clf is None:
        print(model_path)  # trained_knn_model.clf
        with open(model_path, 'rb') as f:  # https://www.cnblogs.com/tianyiliang/p/8192703.html

            knn_clf = pickle.load(f)
            # print(f)
            # print('debugpoint1', '*' * 20)
            # sys.exit(1)

    # print(x_img_path)
    # sys.exit(2)
    x_img = fr.load_image_file(x_img_path)  # https://blog.csdn.net/MG_ApinG/article/details/82252954
    # x_img=fr.load_image_file(x_img_path)
    # print('debugpoint2', '*' * 20)
    # sys.exit(2)

    x_face_location = fr.face_locations(x_img)
    # print(x_face_location)
    # print('debugpoint3', '*' * 20)
    # sys.exit(3)


    encodings = fr.face_encodings(x_img)  # http://www.360doc.com/content/18/0403/18/48868863_742603302.shtml
    # print(encodings)
    # print(len(encodings[0]))
    # print('debugpoint4', '*' * 20)
    # sys.exit(4)
    x_face_locations = fr.face_locations(x_img)
    # print('debugpoint5', '*' * 20)
    # print(x_face_locations)
    # sys.exit(5)

    closest_distace = knn_clf.kneighbors(encodings, n_neighbors=3)
    # print('debugpoint6', '*' * 20)
    print(closest_distace)  # (array([[0.34381298, 0.35287966, 0.35839984]]), array([[3, 2, 7]], dtype=int64))
    # sys.exit(6)

    are_matches = [closest_distace[0][i][0] <= distance_threshold for i in
                   range(len(x_face_locations))]
    # print('debugpoint7', '*' * 20)
    # print(are_matches)
    # sys.exit(7)

    print(knn_clf.predict(encodings))
    # print(list(x_face_locations))
    # print(list(are_matches))
    print(list(zip(knn_clf.predict(encodings), x_face_locations, are_matches)))

    return [(pred, loc) if rec else ('unknown', loc) for pred, loc, rec in
            zip(knn_clf.predict(encodings), x_face_locations,
                are_matches)]


def show_names_on_image(img_path, predictions):
    pil_image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 255))

        name = name.encode('utf-8')
        #name = name.decode('ISO-8859-1')
        name = name.decode('utf-8')
        # print('To print name is ', type(name))

        # sys.exit(1)
        text_width, text_height = draw.textsize(name)

        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
        draw.text((left, bottom - text_height - 10), name, (255, 0, 255))

        li_names.append(name)

    del draw

    pil_image.show()



li_names = []

def count(train_dir):
    # '''
    # counts the total number of the set
    # :param train_dir:
    # :return:
    # '''
    path = train_dir
    count = 0
    for fn in os.listdir(path):
        count = count + 1
    return count


def list_all(train_dir):
    # '''
    # determine the list of all names
    # :param train_dir:
    # :return:
    # '''
    path = train_dir
    result = []
    for fn in os.listdir(path):
        result.append(fn)
    return result


def stat_output():
    s_list = set(li_names)
    s_list_all = set(list_all('train'))
    if 'unknown' in s_list:
        s_list.remove('unknown')
    print('check', s_list)
    tot_num = count('train')
    s_absent = set(s_list_all - s_list)
    print('\n')
    print('***********************\n')
    print('all name:', s_list_all)
    print('sign in:', s_list)
    print('shuold arrive:', tot_num)
    print('have arrived:', len(s_list))
    print('attendence rate:{:.2f}'.format(float(len(s_list)) / float(tot_num)))
    print('not arrive:', s_absent)


if __name__ == '__main__':
    # sys.exit()

    # classifier=train('examples/train',model_save_path='trained_knn_model.clf',n_neighbors=3)
    # print('train finish')

    for image_file in os.listdir('./test'):
        # print('open success')
        for picture_flie in os.listdir('./test/{}'.format(image_file)):
            full_file_path = os.path.join('./test/{}'.format(image_file),picture_flie)
            print('full_file_path:', full_file_path)
            print(picture_flie)
            # sys.exit(0)
            print('looking for faces in {}'.format(image_file))
            # sys.exit(0)

            prediction = predict(full_file_path, model_path='trained_knn_model.clf')
            print(prediction)
            #print('accomplish a {} face {} detect'.format(image_file, picture_flie), '+' * 50)
            # sys.exit(1)
	    #cv2.imwrite('./result_{}'.format(picture_flie),full_file_path)


            #for name, (top, right, bottom, left) in prediction:
                #print('find face:{};face location:({},{},{},{})'.format(name, top, right, bottom, left))
            # sys.exit()

            show_names_on_image(os.path.join('./test/{}'.format(image_file), picture_flie), prediction)
    # sys.exit()

    stat_output()
