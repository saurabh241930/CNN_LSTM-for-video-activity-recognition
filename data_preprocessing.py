from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
import cv2
import math
import os
import shutil
from numpy import argmax
from keras.utils import to_categorical
import subprocess
import sys
import warnings
# warnings.warn("Warning...........Message")


base_model = InceptionV3(weights='imagenet',include_top=True)
extactor_model = Model(inputs=base_model.input,outputs=base_model.get_layer('avg_pool').output)


def getLength(filename):
    result = subprocess.Popen(["ffprobe", filename],
    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    info = [x for x in result.stdout.readlines() if b"Duration" in x]
    return info[0]


def pre_process(video_dir):
    classes = []
    X = []
    y = []
    all_videos_path = video_dir


    if not os.path.exists('images'):
        os.makedirs('images')

    for fld in os.listdir(all_videos_path):
        classes.append(fld)

    for fld in os.listdir(all_videos_path):
        for file in os.listdir(all_videos_path+fld):
            video_path = str(all_videos_path+fld+'/'+file)
            video_length = float(str(getLength(video_path)).split(",")[0].split(":")[-1])

            if video_length < 1:
                print("[WARN] Skipping video because of wrong length")
                pass
            else:
                imagesFolder = "images"
                cap = cv2.VideoCapture(video_path)
                frameRate = cap.get(5)
                count = 0

                if video_length < 1.5:
                    controller = 4
                elif video_length < 2.5:
                    controller = 6
                elif video_length < 3.5:
                    controller = 8
                elif video_length < 5.5:
                    controller = 10
                elif video_length < 8.5:
                    controller = 12
                else:
                    controller = 12

                folder = '/tf/CNN_LSTM/images'
                for the_file in os.listdir(folder):
                    file_path = os.path.join(folder, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        print(e)


                while(cap.isOpened()):
                    frameId = cap.get(1) 
                    ret, frame = cap.read()
                    if (ret != True):
                        break
                    if (frameId%controller==0):
                        count = count + 1
                        filename = imagesFolder + "/image_" +  str(int(count)) + ".jpg"
                        cv2.imwrite(filename, frame)

                    if frameId == controller*6:
                        break
                cap.release()
                print ("[INFO] Video Converted")

                files = []
                for f in os.listdir('images'):
                    if f.endswith(".jpg"):
                        files.append("images/"+str(f))

                sequence = []

                for f in files:
                    img = image.load_img(f, target_size=(299, 299))
                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis=0)
                    x = preprocess_input(x)
                    features = extactor_model.predict(x)
                    sequence.append(features)

                label_encoded = classes.index(str(fld))
                label_hot = to_categorical(label_encoded, len(classes))


                X.append(sequence)
                y.append(label_hot)
                print("[INFO] Features Extracted for {} Frames at controller {} for class {}".format(len(sequence),controller,fld))

    np.save("X_data",X)            
    np.save("y_data",y)



    
pre_process(sys.argv[1])