import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def app():
    imagesPath = os.listdir('dataset/images')

    X_train, Y_train = getImagesAndLabels()
    plotImgsXEmotion(Y_train)
    #separando os dados aleatoriamento em treino e teste
    #15% vai para teste para verificar se não ouve overfitting
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size = 0.15, random_state=42)
    #imprimindo as informações do formato dos dados
    numeroClasses = len(Y_train[0])
    print("type ",type(X_train))
    print("registros: ", len(X_train))
    print("shape: ", X_train.shape)
    print("dimensões: ", X_train.ndim)
    print("Classes: ", numeroClasses)
    #dataset: 223 registros
    #imagens 256 x 256
    #array numpy

    #convertendo o formato (213,256,256) dos dados para encaixar no padrao keras, com 4 dimensões
    X_train = X_train.reshape(len(X_train),256,256,1)
    X_test = X_test.reshape(len(X_test),256,256,1)
    
    #carregar o modelo salvo
    #model = keras.models.load_model('my_model.h5')

    #criar o modelo novamento
    model = buildModel() 
    
    checkpoint_filepath = 'checkpoints/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=False,
    verbose=1)   
    
    model.load_weights(checkpoint_filepath)
    
    #treinar a rede
    model.fit(X_train, Y_train, verbose=1, batch_size=30, epochs=1, callbacks=[model_checkpoint_callback])
    #model.save('my_model.h5')
    
    #testar a precisão da rede com dados de teste
    test(X_test, Y_test, model)


def test(X, Y, model):
    Y_predict = model.predict_classes(X)
    registers = len(Y_predict)
    rights = 0
    print(Y_predict)
    exit
    for i in range(len(Y_predict)):
        if compareHotEncodedClasses(Y[i], Y_predict[i]):
            rights += 1
    accuracy = (rights/registers) * 100
    print("Precisão em teste de: {}%".format(accuracy))

def compareHotEncodedClasses(y_correto, y_previsto):
    y_correto_code = 0
    for i in range(len(y_correto)):
        if y_correto[i] == 1:
            y_correto_code = i
            break
    
    humorCorretoDecoded = decodeHumor(y_correto_code)
    humorPrevistoDecoded = decodeHumor(y_previsto)
    print("Previsto: {} : {} ".format(humorPrevistoDecoded, humorCorretoDecoded))
    if (y_correto_code == y_previsto):
        return True
    else:
        return False


def buildModel():
    kernel = (5,5)
    pooling_kernel = (2,2)
    learning_rate = 0.001
    model = keras.Sequential()
    model.add(keras.Input(shape=(256, 256, 1)))
    model.add(layers.Conv2D(64, kernel, activation='relu', input_shape=(256, 256, 1), name='Conv1'))
    #camada da função de ativação
    model.add(layers.PReLU())
    #reduzir a dimensão da imagem de 256 para 128
    model.add(layers.AveragePooling2D(pool_size=(kernel), name='Pooling1'))
    model.add(layers.PReLU(name='PRelu1'))
    model.add(layers.Conv2D(128, kernel, name='Conv2'))
    model.add(layers.AveragePooling2D(pool_size=pooling_kernel,  name='Pooling2'))
    model.add(layers.Conv2D(256, kernel,  name='Conv3'))
    model.add(layers.PReLU(name='PRelu2'))
    model.add(layers.AveragePooling2D(pool_size=pooling_kernel, name='Pooling3'))
    #evitando overfitting
    model.add(layers.Dropout(0.25, name='Dropout1'))
    #transformando em vetor
    model.add(layers.Flatten(name='Flatten'))
    
    model.add(layers.Dense(512, name='Densa1'))
    model.add(layers.PReLU(name='PRelu3'))
    #evitando overfitting
    model.add(layers.Dropout(0.5, name='Dropout2'))
    
    model.add(layers.Dense(7, activation='softmax',name='Densa2'))

    optimizer = keras.optimizers.Adam(lr=learning_rate, decay=1e-5)    
    losser = keras.losses.categorical_crossentropy
    #perda = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss = losser, optimizer = optimizer, metrics = ['accuracy'])
    
    return model

def getImagesAndLabels():
    file = open("dataset/data.csv", "r")
    file.readline()
    lines = file.readlines()
    paths = []
    labels = []
    for line in lines:
        lineSplit = line.split(",")
        path = "dataset/" + lineSplit[0].strip()
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image = image/255#normalização dos dados: 0 - 1
        npImage = np.array(image)
        paths.append(npImage)
        labels.append(getLabelCode(lineSplit[len(lineSplit) - 1].strip()))
    #if need hot enconding
    labels = to_categorical(labels, 7)
    labels = np.asarray(labels)
    return [np.asarray(paths), labels]

def plotImgsXEmotion(Y_train):
    qtdPerEmotion = [0, 0, 0, 0, 0, 0, 0]
    classes = ["neutral", "disgust", "fear", "happiness", "angry", "sadness", "surprise"]
    for y in Y_train:
        code = 0
        for i in range(len(y)):
            if y[i] == 1:
                code = i
                break
        qtdPerEmotion[code] += 1
        #humor = decodeHumor(code)
    print("Quantidade de fotos por classe")
    print(classes)
    print(qtdPerEmotion)
    fig,ax = plt.subplots(figsize=(8,2))
    ax.bar(qtdPerEmotion, 7, width=0.8, color='red')
    ax.set_xticks(qtdPerEmotion)
    ax.set_xticklabels(classes)
    #plt.show()

def decodeHumor(code):
    if (code == 0):
        return "neutral"
    elif (code == 1):
        return "disgust"
    elif (code == 2):
        return "fear"
    elif (code == 3):
        return "happiness"
    elif (code == 4):
        return "angry"
    elif (code == 5):
        return "sadness"
    elif (code == 6):
        return "surprise"
    else:
        return "None"


def getLabelCode(label):
    if (label == "neutral"):
        return 0
    elif (label == "disgust"):
        return 1
    elif (label == "fear"):
        return 2
    elif (label == "happiness"):
        return 3
    elif (label == "angry"):
        return 4
    elif (label == "sadness"):
        return 5
    elif (label == "surprise"):
        return 6
    else:
        print("Label not found: " + label)
        return None

app()