###############################################################################
#
# genreNet.py
#
###############################################################################

import cv2
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

workingdir = "/data/Documents/Princeton/Senior_Spring2/COS424/project/project_code/"
genres = ["blues",
              "classical",
              "country",
              "disco",
              "hiphop",
              "jazz",
              "metal",
              "pop",
              "reggae",
              "rock"]

def getTrainingData():
    x = []
    y = []
    for i in range(len(genres)):
        for j in range(10,100):
            im = np.array([256,645,3])
            im = cv2.imread(workingdir + "spectrograms_xsmall/" +
                            genres[i] + "/" + 
                            genres[i] + ".000" + str(j) + ".spec.png")
            x.append(im[:,0:645,0])
            label = np.zeros(10)
            label[i] = 1
            y.append(label)
    x2 = np.array(x)
    y2 = np.array(y)
    return x2, y2

def getTestingData():
    x = []
    y = []
    for i in range(len(genres)):
        for j in range(10):
            im = np.array([256,645,3])
            im = cv2.imread(workingdir + "spectrograms_xsmall/" +
                            genres[i] + "/" + 
                            genres[i] + ".0000" + str(j) + ".spec.png")
            x.append(im[:,0:645,0])
            label = np.zeros(10)
            label[i] = 1
            y.append(label)
    x2 = np.array(x)
    y2 = np.array(y)
    return x2, y2

def makeConvNet():
    network = input_data(shape=[None, 256, 645, 1], name='input')
    network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, 
                         optimizer='adam', 
                         learning_rate=0.01,
                         loss='categorical_crossentropy', 
                         name='target')
    return network

if __name__ == "__main__":
    X, Y = getTrainingData()
    testX, testY = getTestingData()
    X = X.reshape([-1,256,645,1])
    testX = testX.reshape([-1,256,645,1])
    network = makeConvNet()

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit({'input': X}, 
              {'target': Y}, 
              n_epoch=20,
              validation_set=({'input': testX}, {'target': testY}),
              batch_size=25,
              shuffle=True,
              snapshot_epoch=True,
              snapshot_step=100,  
              show_metric=True, 
              run_id='convnet_genre')
