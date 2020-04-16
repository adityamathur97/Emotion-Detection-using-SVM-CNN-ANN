#===Building CNN=========#
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#====Initializing CNN======#
classifier=Sequential()

#==== 1) Building Convolution layer=======#

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))

#=====2) Pooling==========================#

classifier.add(MaxPooling2D(pool_size=(2,2)))

#===Adding second CNN layer==================#

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#=====3) Flattening=======================#

classifier.add(Flatten())

#=====4) Full Connection===================#

classifier.add(Dense(output_dim= 128,activation='relu'))
classifier.add(Dense(output_dim= 6,activation='softmax'))

#===Compiling the CNN=======================#

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#==@ Fitting the CNN to the images============#

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('../Dataset/train_data',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


test_set = test_datagen.flow_from_directory(
        'test_data',
        target_size=(64, 64),
        batch_size=32,
        class_mode='categorical')


#classifier.fit_generator(training_set,samples_per_epoch=2000,nb_epoch=100)
classifier.fit_generator(training_set,
                         steps_per_epoch = 100,
                         epochs = 100,
                         validation_data = test_set,
                         validation_steps = 100)


from keras.models import load_model

classifier.save("saved_model/my_model_CNN_aditya.h5")
print("Training Done")







