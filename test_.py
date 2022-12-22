import flwr as fl
from numpy import float32
import tensorflow as tf
import glob
import numpy as np
import cv2
import os
import copy


class face_Detection():
    def __init__(self,size_image):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.size_image=size_image
        # self.smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
        # self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')


    def Detect(self,base_image):
        grey = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(grey, 1.3, 5)

        faces_list=[] 
        for (x,y,w,h) in faces:
            croped_image=base_image[x:(x+w),y:(y+h)]
            img=cv2.resize(croped_image, self.size_image)
            faces_list.append(copy.deepcopy(img))
        
        return faces_list

        # smiles = smile_cascade.detectMultiScale(grey, 1.3, 20)
        # for (x,y,w,h) in smiles:
        #     cv2.rectangle(test_image,(x,y),(x+w,y+h),(0,255,0),2)


        # eyes = eye_cascade.detectMultiScale(grey, 1.3, 1)
        # for (x,y,w,h) in eyes:
        #     cv2.rectangle(test_image,(x,y),(x+w,y+h),(255,255,255),2)

def load_datas(path,input_size):
    # All files ending with .txt
    class_dirs=glob.glob(path+"/*/")
    detect_model=face_Detection(input_size)
    X_train=[]
    Y_train=[]
    for d in range(len(class_dirs)):
        dir_name=class_dirs[d]+'/*.jpg'
        files_d=glob.glob(dir_name)
        for f in files_d:
            # read images
            img = cv2.imread(f)
            faces_list=detect_model.Detect(img)
            for face in faces_list:
                # create onehot vector of label
                onehot_Ytrue=np.zeros((len(class_dirs),),dtype=np.float32)
                onehot_Ytrue[d]=1
                # append to list
                X_train.append(copy.deepcopy(face))
                Y_train.append(copy.deepcopy(d))

    return np.array(X_train,dtype=np.float32),np.array(Y_train,dtype=np.float32),class_dirs


def create_model(class_num,input_size):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(input_shape=input_size, filters=32, kernel_size=3, 
                        strides=2, activation='relu', name='Conv1'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(1, 1),
                     padding='valid',name='pool1'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, name='Dense1'),
    tf.keras.layers.Dense(32, name='Dense2'),
    tf.keras.layers.Dense(class_num, name='DenseOutput')
    ])
    # model.summary()

    # ,
    model.compile(optimizer='adam', 
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model



input_size01=(28,28)
input_size02=(28,28,3)
current_dir=os.getcwd()
data_path=os.path.join(current_dir,'data')
x_train,y_train,labels=load_datas(data_path,input_size01)
model=create_model(class_num=2,input_size=input_size02)
model.fit(x_train, y_train, epochs=10, batch_size=4)  # Remove `steps_per_epoch=3` to train on the full dataset
