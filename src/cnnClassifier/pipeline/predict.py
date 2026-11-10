# load the model and whatever data provided by the user give prediction on the type of data
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image 
import os 

class PredictionPipeline:
    def __init__(self,filename):
        self.filename = filename

    
    def predict(self):
        model = load_model(os.path.join("artifacts", "training", "model.h5"))
        imagename = self.filename
        # load the image
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        # expand the dimensionl because single image ( dl batches mai samjhta hai)
        test_image = np.expand_dims(test_image, axis = 0) # ( 0 ka matlab hua ki 1 row add kr diyae -- 1,224,224,3(rgb))
        result = np.argmax(model.predict(test_image), axis=1) # it return the index of maximum value in column
        print(result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Chest Disease'
            return [{ "image" : prediction}]
