import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from utils import build_inception_model, build_custom_model




# Function to preprocess the image
def preprocess_image(image):
    img = image.resize((256, 256))  # Resize to model's expected input size
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


# Function to make a prediction
def make_prediction(model, image):
    input_arr = preprocess_image(image)
    pred = model.predict(input_arr)
    result = (pred > 0.5).astype("int32")[0][0]
    return result


# Streamlit app function
def main():
    st.title("Cat Vs Dog Classification")
    st.write("Upload an Image.")


    # Model selection
    model_choice = st.selectbox("Choose a model:", ["Custom CNN", "InceptionV3"])


    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])


    if uploaded_file is not None:
        # Display the image
        image = load_img(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)


        # Load the selected model
        if model_choice == "Custom CNN":
            model = build_custom_model()
        elif model_choice == "InceptionV3":
            model = build_inception_model()


        # Make prediction
        if st.button("Predict"):
            result = make_prediction(model, image)
            if result == 0:
                st.success('This is a cat.')
            else:
                st.error('This is a dog.')


if __name__ == "__main__":
    main()
