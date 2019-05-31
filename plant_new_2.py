import os
print(os.listdir('data/train/'))

######
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import seaborn as sn


import fnmatch
import os
import numpy as np
import pandas as pd
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import keras
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16

from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt


np.random.seed(21)

'''
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
'''


path = 'data/train/'
train_label = []
train_img = []
label2num = {'Loose Silky-bent':0, 'Charlock':1, 'Sugar beet':2, 'Small-flowered Cranesbill':3,
             'Common Chickweed':4, 'Common wheat':5, 'Maize':6, 'Cleavers':7, 'Scentless Mayweed':8,
             'Fat Hen':9, 'Black-grass':10, 'Shepherds Purse':11}
for i in os.listdir(path):
    label_number = label2num[i]
    new_path = path+i+'/'
    for j in fnmatch.filter(os.listdir(new_path), '*.png'):
        temp_img = image.load_img(new_path+j, target_size=(128,128))
        train_label.append(label_number)
        temp_img = image.img_to_array(temp_img)
        train_img.append(temp_img)

train_img = np.array(train_img)

train_y=pd.get_dummies(train_label)
train_y = np.array(train_y)
train_img=preprocess_input(train_img)
#species = train_y.unique()

print('Training data shape: ', train_img.shape)
print('Training labels shape: ', train_y.shape)

###################################



def vgg16_model(num_classes=None):

    model = VGG16(weights='imagenet', include_top=False,input_shape=(128,128,3))
    model.layers.pop()
    model.layers.pop()
    model.layers.pop()

    model.outputs = [model.layers[-1].output]

    model.layers[-2].outbound_nodes= []
    x=Conv2D(256, kernel_size=(2,2),strides=2)(model.output)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)    
    x=Conv2D(128, kernel_size=(2,2),strides=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax')(x)

    model=Model(model.input,x)

    for layer in model.layers[:15]:

        layer.trainable = False


    return model




########################################################


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def fscore(y_true, y_pred):
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    f_score = 2 * (p * r) / (p + r + K.epsilon())
    return f_score

########################################################


num_classes=12
model = vgg16_model(num_classes)
model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy',fscore])
model.summary()



########################################################


#Split training data into rain set and validation set

X_train, X_valid, Y_train, Y_valid=train_test_split(train_img,train_y,test_size=0.1, random_state=42)

#Data augmentation

from keras.callbacks import ModelCheckpoint
epochs = 80
batch_size = 128
model_checkpoint = ModelCheckpoint('./output/weights.h5', monitor='val_loss', save_best_only=True)

model.fit(X_train,Y_train,
          batch_size=128,
          epochs=epochs,
          verbose=1, shuffle=True, validation_data=(X_valid,Y_valid), callbacks=[model_checkpoint])

########################################################


def plot_model(model):
    plots = [i for i in model.history.history.keys() if i.find('val_') == -1]
    plt.figure(figsize=(15,10))

    for i, p in enumerate(plots):
        plt.subplot(len(plots), 2, i + 1)
        plt.title(p)
        plt.plot(model.history.history[p], label=p)
        plt.plot(model.history.history['val_'+p], label='val_'+p)
        plt.legend()

    #plt.show()
    plt.savefig('./output/Performance-Measures.png')
    
plot_model(model)



########################################################


model.load_weights('./output/weights.h5')


########################################################


prob=[]
num=[]
test_img=[]
test_path = 'data/test/'
test_all = fnmatch.filter(os.listdir(test_path), '*.png')

test_img=[]
for i in range(len(test_all)):
    path=test_path+'/'+test_all[i]
    temp_img=image.load_img(path,target_size=(128,128))
    temp_img=image.img_to_array(temp_img)
    test_img.append(temp_img) 
test_img=np.array(test_img)    
test_img=preprocess_input(test_img)


test_labels=[]
pred=model.predict(test_img)
num2label =  {0:'Loose Silky-bent', 1:'Charlock',2: 'Sugar beet',3: 'Small-flowered Cranesbill',
              4:'Common Chickweed',5: 'Common wheat',6: 'Maize', 7:'Cleavers', 8:'Scentless Mayweed',
             9: 'Fat Hen', 10:'Black-grass', 11:'Shepherds Purse'}
for i in range(len(test_all)):
    max_score =0
    lab=-1
    for j in range(12):
        if pred[i][j]>max_score:
            max_score=pred[i][j]
            lab=j
    test_labels.append(num2label[lab])


d = {'file': test_all, 'species': test_labels}
df = pd.DataFrame(data=d)

#print(df.head(5))

#*************************************
df.to_csv("./output/Test_results.csv",index=False) 
print('Test CSV saved')
#*************************************

########################################################


Y_pred = model.predict(X_valid)
#Return the column position where the max value is
Y_pred_classes = np.argmax(Y_pred,axis = 1)
Y_true = np.argmax(Y_valid,axis = 1)

#labels=['Loose Silky-bent','Charlock','Sugar beet','Small-flowered Cranesbill','Common Chickweed','Common wheat','Maize','Cleavers','Scentless Mayweed','Fat Hen','Black-grass','Shepherds Purse']
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
print (confusion_mtx)
#plt.imshow(confusion_mtx, cmap='binary')
df_cm = pd.DataFrame(confusion_mtx, index = ['Loose Silky-bent','Charlock','Sugar beet','Small-flowered Cranesbill','Common Chickweed','Common wheat','Maize','Cleavers','Scentless Mayweed','Fat Hen','Black-grass','Shepherds Purse'],
                  columns = ['Loose Silky-bent','Charlock','Sugar beet','Small-flowered Cranesbill','Common Chickweed','Common wheat','Maize','Cleavers','Scentless Mayweed','Fat Hen','Black-grass','Shepherds Purse'])

fig = plt.figure()
plt.title('Confusion matrix of the classifier')
plt.figure(figsize = (15,10))
sn.heatmap(df_cm,cmap='Blues', annot=True)
plt.savefig('./output/Plot-Confusion_matrix.png')
########################################################
