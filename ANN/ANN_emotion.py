#=====Image classification using ANN=========#
#===Import library============#
import pandas as pd
import numpy as np
import cv2
import glob
from skimage.feature import hog
from skimage import data, exposure

#============================Reading TRAIN_dataset=====================================================#
angry_images=[]
#====Reading angry image dataset==============#
for image in glob.glob('../Dataset/train_data/angry/*'):
	angry_images.append(image)
	

for i in range(0,len(angry_images)):
	t=cv2.imread(angry_images[i])
	t=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
	t1=cv2.resize(t,(64,64),interpolation=cv2.INTER_AREA)
	
df=pd.DataFrame(t1)
df=df.astype('float32')
df1=df/255
lab1=pd.DataFrame(np.repeat(1,len(df1)))

#====Reading disgust image dataset==============#
disgust_images=[]

for image in glob.glob('C:/Modified/Dataset/train_data/disgust/*'):
	disgust_images.append(image)
	

for i in range(0,len(disgust_images)):
	t=cv2.imread(disgust_images[i])
	t=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
	t1=cv2.resize(t,(64,64),interpolation=cv2.INTER_AREA)
	
df=pd.DataFrame(t1)
df=df.astype('float32')
df2=df/255
lab2=pd.DataFrame(np.repeat(2,len(df2)))

#====Reading fear image dataset==============#
fear_images=[]

for image in glob.glob('C:/Modified/Dataset/train_data/fear/*'):
	fear_images.append(image)
	

for i in range(0,len(fear_images)):
	t=cv2.imread(fear_images[i])
	t=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
	t1=cv2.resize(t,(64,64),interpolation=cv2.INTER_AREA)
	
df=pd.DataFrame(t1)
df=df.astype('float32')
df3=df/255
lab3=pd.DataFrame(np.repeat(3,len(df3)))


#====Reading happy image dataset==============#
happy_images=[]

for image in glob.glob('C:/Modified/Dataset/train_data/happy/*'):
	happy_images.append(image)
	
for i in range(0,len(happy_images)):
	t=cv2.imread(happy_images[i])
	t=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
	t1=cv2.resize(t,(64,64),interpolation=cv2.INTER_AREA)
	
df=pd.DataFrame(t1)
df=df.astype('float32')
df4=df/255
lab4=pd.DataFrame(np.repeat(4,len(df4)))

#====Reading sad image dataset==============#
sad_images=[]

for image in glob.glob('C:/Modified/Dataset/train_data/sad/*'):
	sad_images.append(image)
	

for i in range(0,len(sad_images)):
	t=cv2.imread(sad_images[i])
	t=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
	t1=cv2.resize(t,(64,64),interpolation=cv2.INTER_AREA)
	
df=pd.DataFrame(t1)
df=df.astype('float32')
df5=df/255
lab5=pd.DataFrame(np.repeat(5,len(df5)))

#====Reading surprise image dataset==============#
surprise_images=[]

for image in glob.glob('C:/Modified/Dataset/train_data/surprise/*'):
	surprise_images.append(image)
	

for i in range(0,len(surprise_images)):
	t=cv2.imread(surprise_images[i])
	t=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
	t1=cv2.resize(t,(64,64),interpolation=cv2.INTER_AREA)
	
df=pd.DataFrame(t1)
df=df.astype('float32')
df6=df/255
lab6=pd.DataFrame(np.repeat(6,len(df6)))

#===Constructing Common Train data_set========#

train_data=np.array(pd.concat([df1,df2,df3,df4,df5,df6],axis=0))
labels=np.array(pd.concat([lab1,lab2,lab3,lab4,lab5,lab6],axis=0))



#=====Implementing HOG for Visualization using Matplotlib=======================#

import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure


image =train_data

fd, hog_image = hog(image, orientations=8, pixels_per_cell=(32, 32),
                    cells_per_block=(1, 1), visualize=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')


#=====Rescale histogram for better display=====================================#
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.savefig('Visualization.png')
#plt.show()


#======Spliting train_data into two part=================#
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
X_train,X_test,y_train,y_test=train_test_split(train_data,labels,test_size=0.33,random_state=42)
y_train=np_utils.to_categorical(y_train)
#labels=np_utils.to_categorical(labels)
from keras.models import Sequential
model = Sequential()
from keras.layers import Dense
model.add(Dense(units=4096, input_dim=64,activation='relu'))  #Input_diamension is number of columns in dataset.
model.add(Dense(4096,activation='relu'))
model.add(Dense(120,activation='relu'))

model.add(Dense(7,activation='softmax'))    #Here the last layer of neural_network must contain number of classes tobe predict
#=====================compile_model=========================================================
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#====================Split_data_into_train_test_split=========================================
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(X_train,y_train, epochs=100, batch_size=32)
#======Saving model==================#
from keras.models import load_model

model.save("saved_model/my_ANN.h5")
model=load_model('saved_model/my_ANN.h5')

#====================evaluation_of_model=====================================================
scores=model.evaluate(X_train,y_train,batch_size=20)
print("Model_evaluation_score is:",scores)
#=====================prediction=============================================================
submit=model.predict(X_test)
pred= submit.argmax(axis=-1)

from sklearn.metrics import accuracy_score
print(accuracy_score(pred,y_test))







