#====Import libraries=======#

import pickle
import cv2
import pandas as pd

#-----Load trained model-------#
filename = 'saved_model/model_aditya.pkl'
loaded_model = pickle.load(open(filename, 'rb'))

t=cv2.imread('test_pics/img4.png')  #=====Provide input image to be tested====#

t=cv2.cvtColor(t,cv2.COLOR_RGB2GRAY)
t1=cv2.resize(t,(64,64),interpolation=cv2.INTER_AREA)
	
import numpy as np
a=loaded_model.predict(t1)
#prediction=np.bincount(a).argmax()
result = int(np.mean(a))
#===Prediction result======#    1-angry  2-disgust  3-fear  4-happy  5-sad  6-surprise
if result==0:
	print('angry')
elif result==1:
	print('disgust')
elif result==2:
	print('fear')
elif result==3:
	print('happy')
elif result==4:
	print('sad')
else:
	print('surprise')
