#Library imports
import numpy as np
import streamlit as st
import cv2
import tensorflow.keras
from keras.models import load_model


#Loading the Model
model = load_model('bottle_fill_level.h5')

#Name of Classes
CLASS_NAMES = ['correctly_filled', 'over_filled', 'under_filled']

#Setting Title of App
st.image("bottle.JPG", width = 500, )
html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:red;text-align:center;">Bottle fill level Detection AI App </h2>
    </div>
    """
st.title("Bottle fill level Detection")
st.markdown("Upload an image of the bottle")

#Uploading the bottle image
plant_image = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Predict')
#On predict button click
if submit:


    if plant_image is not None:

        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        st.write(opencv_image.shape)
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]
        st.title(str("The bottle is "+result.split('-')[0]))
        print(result)


if st.button("About Author"):
               st.text("Name : Orimolade Eyimofe && Temitope Osifalujo") 
               st.text("Email : eyimofeOkikiola@gmail.com") 
               
