from flask import Flask, render_template, request
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import os

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))

model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

@app.route('/')
def index():
    image_files = os.listdir('uploads')
    return render_template('index.html', css_url='style.css', image_files=image_files)

@app.route('/load.html')
def load_html_page():
    image_files = os.listdir('uploads')
    return render_template('load.html', css_url='style1.css', image_files=image_files)

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', css_url='style.css')

    image = request.files['image']
    image_path = f"uploads/{image.filename}"
    image.save(image_path)

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = Image.open(image_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    result = {
        'class_name': class_name,
        'confidence_score': confidence_score
    }

    image_files = os.listdir('uploads')

    return render_template('load.html', result=result, css_url='style1.css', image_files=image_files)


if __name__ == '__main__':
    app.run(debug=True)
