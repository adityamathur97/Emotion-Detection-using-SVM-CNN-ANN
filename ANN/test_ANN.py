#=====Test images on ANN=====#
from keras.models import load_model
import cv2

model=load_model("saved_model/my_ANN_aditya.h5")


#=====Making prediction of new image=================#
import numpy as np
from keras.preprocessing import image
test_image=(image.load_img('test_pics/img6.jpeg',target_size=(64,64)))

test_image=image.img_to_array(test_image)
gray=cv2.cvtColor(test_image,cv2.COLOR_RGB2GRAY)

result=model.predict(gray)
result= result.argmax(axis=1)
pred = int(np.mean(result))

if pred==0:
	print('angry')
elif pred==1:
	print('disgust')
elif pred==2:
	print('fear')
elif pred==3:
	print('happy')
elif pred==4:
	print('sad')
else:
	print('surprise')