from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras import utils
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'D:\projects\Mini project gec\lung\models\model_vgg16.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model.make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/home')


def model_predict(img_path, model):
    img = utils.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = utils.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/home', methods=['GET'])
def index():
    # Main page
    return render_template('project.html')

@app.route("/knowus.html")
def know():
    return render_template('knowus.html')
@app.route("/HOB.html")
def hob():
    return render_template('HOB.html')
@app.route("/contactus.html")
def con():
    return render_template('contactus.html')
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    print("hi")
    if request.method == 'POST':
        print("SEKHAR")
        # Get the file from post request
        f = request.files['fileq']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(preds)
        print(result)          
        if(str(result)=='[[1. 0.]]'):
            result=" NO PNEMONIA, you are safe"
        else:
            result="PNEMONIA,Please consider doctor"
        return render_template('projectr.html',data=result)
    return None


if __name__ == '__main__':
    app.run(debug=True)
    