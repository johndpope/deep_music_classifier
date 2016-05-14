###############################################################################
#
# genreNet.py
#
###############################################################################

import os
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
    p = np.random.permutation(len(x2))
    return x2[p], y2[p]

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
    p = np.random.permutation(len(x2))
    return x2[p], y2[p]

def makeConvNetSGD():
    network = input_data(shape=[None, 256, 645, 1], name='input')
    network = conv_2d(network, 32, 20, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 20, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    sgd = tflearn.optimizers.SGD(learning_rate=0.01, 
                                 lr_decay=0.96, 
                                 decay_step=100)
    network = regression(network, 
                         optimizer=sgd, 
                         loss='categorical_crossentropy', 
                         name='target')
    return network

def makeConvNetSGD2():
    network = input_data(shape=[None, 256, 645, 1], name='input')
    network = conv_2d(network, 32, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 2)
    network = local_response_normalization(network)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.5)
    network = fully_connected(network, 10, activation='softmax')
    sgd = tflearn.optimizers.SGD(learning_rate=0.01)
    network = regression(network, 
                         optimizer=sgd, 
                         loss='categorical_crossentropy', 
                         name='target')
    return network

def makeConvNet3Layer(opt='adam', lr=0.01, los='categorical_crossentropy'):
    network = input_data(shape=[None, 256, 645, 1], name='input')
    network = conv_2d(network, 128, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 3)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 3)
    network = local_response_normalization(network)
    network = fully_connected(network, 1024, activation='tanh')
    network = dropout(network, 0.7)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, 
                         optimizer=opt, 
                         learning_rate=lr,
                         loss=los, 
                         name='target')
    return network

def makeConvNet3Layer2(opt='adam', lr=0.01, los='categorical_crossentropy'):
    network = input_data(shape=[None, 256, 645, 1], name='input')
    network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 4)
    network = local_response_normalization(network)
    network = conv_2d(network, 32, 5, activation='relu', regularizer="L2")
    network = max_pool_2d(network, 3)
    network = local_response_normalization(network)
    network = fully_connected(network, 512, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 10, activation='softmax')
    network = regression(network, 
                         optimizer=opt, 
                         learning_rate=lr,
                         loss=los, 
                         name='target')
    return network

def makeConvNet4Layer(opt='adam', lr=0.01, los='categorical_crossentropy'):
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
                         optimizer=opt,
                         learning_rate=lr,
                         loss=los, 
                         name='target')
    return network



if __name__ == "__main__":
    X, Y = getTrainingData()
    testX, testY = getTestingData()
    X = X.reshape([-1,256,645,1])
    testX = testX.reshape([-1,256,645,1])

    opt = 'sgd'
    lr = 0.01
    los = 'categorical_crossentropy'
    epochs = 25
    batch = 15 

    savestr = '3layer_'+opt+str(lr)+los+str(epochs)+'e'+str(batch)+'b'
    #network = makeConvNetSGD2()
    #network = makeConvNet4Layer(opt, lr, los)
    network = makeConvNet3Layer(opt, lr, los)

    os.mkdir('logs/'+savestr)
    model = tflearn.DNN(network, 
                        tensorboard_verbose=1, 
                        tensorboard_dir='logs/'+savestr+'/')
    model.fit({'input': X}, 
              {'target': Y}, 
              n_epoch=epochs,
              validation_set=({'input': testX}, {'target': testY}),
              batch_size=batch,
              snapshot_epoch=True,
              show_metric=True, 
              run_id=savestr)

    model.save('trained_models/'+savestr+'.tfl')
