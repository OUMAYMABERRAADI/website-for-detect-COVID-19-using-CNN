import os
from flask import Flask,redirect,url_for,render_template,request, send_from_directory, abort, flash
from werkzeug.utils import secure_filename
from skimage.transform import resize
import matplotlib.pyplot as plt

from tensorflow.python.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


import numpy as np

app = Flask(__name__)

print("Loading model")
tf.compat.v1.disable_eager_execution()
global sess
sess = tf.compat.v1.Session()
set_session(sess)
global model
model = load_model('covid19_model.h5')
global graph
graph = tf.compat.v1.get_default_graph()

picFolder="C:/Users/AYOUB/AppData/Local/Programs/Python/Python310/covid-19 website/uploads"

app.config['UPLOAD_FOLDER'] = picFolder
ALLOWED_EXTENSIONS = {'jpeg', 'jpg'}


from PIL import Image

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
import secrets

secret = secrets.token_urlsafe(32)

app.secret_key = secret

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
          filename = secure_filename(file.filename)
          file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
          return redirect(url_for('result', filename=filename))
    return render_template("home.html")


@app.route("/result/<filename>")
def result(filename):
    IMG_SIZE=196
    my_image = plt.imread(os.path.join('uploads', filename))
    my_image_re = resize(my_image, (196, 196, 1))
    with graph.as_default():
      set_session(sess)
      probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
      print(probabilities)
    
    number_to_class= ['NORMAL', 'PNEUMONIA']
    index = np.argsort(probabilities)
    predictions = { "class1":number_to_class[index[0]],
         "class2":number_to_class[index[1]],
         "prob1":probabilities[index[0]],
         "prob2":probabilities[index[1]], }
    c1 = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return render_template("result.html", predictions=predictions)

app.run(host='127.0.0.2',port=8000,debug=True)
