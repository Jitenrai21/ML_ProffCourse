from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io


app = Flask(__name__)


# Load updated model
try:
    model = tf.keras.models.load_model(r"C:\Users\ACER\gitClones\ML_ProffCourseModels\models\incep_dog_cat_classifier.h5")
except Exception as e:
    print(f"Error loading model: {e}")


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((256, 256))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400


    image = request.files['image'].read()
    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)[0][0]
        result = {
            'class': 'Dog' if prediction > 0.5 else 'Cat',
            'confidence': float(prediction if prediction > 0.5 else 1 - prediction)
        }
        return jsonify(result)


    except Exception as e:
        return jsonify({'error': f"Prediction error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=False)