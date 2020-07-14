import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import scipy 
import os
import keras
import sklearn
#%matplotlib inline


train=pd.read_csv("Cleaned_Train_Data.csv")
test=pd.read_csv("Cleaned_Test_Data.csv")

train_img_path='Train_Data/Train/'
test_img_path='Test_Data/Test/'

train_img_names=os.listdir(train_img_path)
test_img_names=os.listdir(test_img_path)

def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256))
    return img

from tqdm import tqdm

train_img = []
for img in tqdm(train['Image_Names'].values):
    train_img.append(read_img(train_img_path + img))


X_train=np.array(train_img,np.float32)
X_train=X_train/255.0
print("Train Images Shape:",X_train.shape)
np.save("Train_256_256.npy",X_train)

print("Train Image Saved")
'''
labels= train['Sub_category'].tolist()
Y_train = {k:v+1 for v,k in enumerate(set(labels))}
y_train = [Y_train[k] for k in labels]

'''
test_img = []
for img in tqdm(test['Image_Names'].values):
    test_img.append(read_img(test_img_path + img))

X_test=np.array(test_img,np.float32)/255.0

print("Test Image Shape:",test.shape)
np.save("Test_256_256.npy",X_test)
print("Test Images Saved")
'''
mean_img=np.mean(X_train,axis=0)
std_img=np.std(X_train,axis=0)
X_train_norm=(X_train-mean_img)/std_img
X_test_norm=(X_test-mean_img)/std_img

#y_train=np.array(pd.get_dummies(train['Sub_category']))


from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D
from keras.callbacks import EarlyStopping

model = Sequential()
model.add(Convolution2D(32, (3,3), activation='relu', padding='same',input_shape = (64,64,3))) # if you resize the image above, change the shape
model.add(Convolution2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, (3,3), activation='relu', padding='same'))
model.add(Convolution2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


early_stops = EarlyStopping(patience=3, monitor='val_acc')

model.fit(X_train_norm, y_train, batch_size=100, epochs=10, validation_split=0.3, callbacks=[early_stops])

## Model Saving

from keras.models import model_from_json
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


predictions = model.predict(X_test_norm)
predictions = np.argmax(predictions, axis= 1)


y_maps = dict()
y_maps = {v:k for k, v in Y_train.items()}
pred_labels = [y_maps[k] for k in predictions]


sub = pd.DataFrame({'Image_Name': test['Image_Names'], 'Class':pred_labels})
sub.to_csv('Submission_64_64_69.csv', index=False)
'''
















