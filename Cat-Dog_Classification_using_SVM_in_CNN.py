# type streamlit run Cat-Dog_Classification_using_SVM_in_CNN.py to run this code

import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","Prediction"])

# Main Page
if(app_mode=="Home"):
    st.header("PRODIGY INFOTECH INTERNSHIP PROGRAM")
    image_path = "D:\\Prodigy\\Cat_Dog_Classifier\\cat_dog.jpg"
    st.subheader("Cat Dog Classification System")
    st.image(image_path)
    st.header("About Project")
    st.write("This is a Cat Dog Classification System. You can insert any image of the cat and dog and let the model guess whether it is Cat or Dog. This dataset contains images of cats and dogs.")
    st.subheader("Content")
    st.text("This dataset contains two folders:")
    st.text("1. train (10000 images each)")
    st.text("2. test (1000 images each)")

# Prediction Page
elif(app_mode=="Prediction"):
    st.header("Model Prediction")
    test_image = st.file_uploader("Choose an Image:")

    if test_image:
        st.subheader("The Image you selected is displayed below:")
        st.image(test_image,width=4,use_column_width=True)
            
        #Predict button
        if st.button("Predict"):
            
            st.write("Our Prediction")
            model = load_model('model_r_cat_dog.h5')
            Image = image.load_img(test_image, target_size = (64,64))
            Image = image.img_to_array(Image)
            Image=Image/255
            Image = np.expand_dims(Image, axis = 0)
            result = model.predict(Image)
            if result[0]<0:
                st.success("Model is Predicting that it's a Cat")
            else:
                st.success("Model is Predicting that it's a Dog")
            st.snow()
