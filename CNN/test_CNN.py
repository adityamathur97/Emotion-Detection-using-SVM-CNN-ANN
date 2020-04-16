from keras.models import load_model

model=load_model("saved_model/my_model_CNN_aditya.h5")


#=====Making prediction of new image=================#
import numpy as np
from keras.preprocessing import image
test_image=image.load_img('test_pics/img6.jpeg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)
result=model.predict(test_image)
#print training_set.class_indices
result= result.argmax(axis=-1)


print("Predicted Result :")

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

