from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from utils.preprocess import prepare_image

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = tf.keras.models.load_model("model/efficientnet_model.h5")
class_names = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse',
               'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, top_preds, image_url = None, None, [], None
    if request.method == "POST":
        file = request.files["image"]
        if file:
            path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(path)
            image_url = path
            image = Image.open(path)
            processed_image = prepare_image(image)
            predictions = model.predict(processed_image)[0]
            prediction = class_names[np.argmax(predictions)]
            confidence = np.max(predictions) * 100
            top_indices = predictions.argsort()[-3:][::-1]
            top_preds = [(class_names[i], predictions[i] * 100) for i in top_indices]
    return render_template("index.html", prediction=prediction,
                           confidence=confidence, top_preds=top_preds, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)