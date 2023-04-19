import datetime
import os
import pickle

import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import minmax_scale
from werkzeug.utils import secure_filename

import module as md

app = Flask(__name__)
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MODEL_FILE'] = 'model/knn_model.pkl'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


with open(app.config['MODEL_FILE'], 'rb') as file:
    knn_model = pickle.load(file)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        image = request.files["image"]
        pakan = request.form["pakan"]
        usia = int(request.form["usia"])
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            md.image_processing(image_path)
            df1 = pd.read_csv('static/result/rgb.csv')
            df2 = pd.read_csv('static/result/first_order.csv')
            df3 = pd.read_csv('static/result/second_order.csv')
            df = pd.concat([df1, df2, df3], axis=1)
            df.insert(0, "Pakan", pakan)
            df.insert(1, "Usia", usia)
            df = pd.DataFrame(minmax_scale(df), columns=df.columns)
            label = knn_model.predict(df)
            label_map = {
                1: "Tidak ada endapan",
                2: "Sedikit endapan",
                3: "Banyak endapan"
            }
            label_text = label_map.get(label[0], "unknown")
            # Calculate shift_pond based on usia and predicted class
            if usia <= 30:
                shift_pond = 30 - usia
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            elif label == 1:
                shift_pond = 4
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            elif label == 2:
                shift_pond = 3
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            elif label == 3:
                shift_pond = 2
                shift_date = datetime.date.today() + datetime.timedelta(days=shift_pond)
            else:
                shift_pond = None
                shift_date = None
            return render_template("prediction.html", result=label_text, label=label, shift_pond=shift_pond, shift_date=shift_date)
        else:
            return render_template("prediction.html", error="Silahkan upload gambar dengan format JPG")
    else:
        return render_template("prediction.html")


if __name__ == "__main__":
    app.run()
