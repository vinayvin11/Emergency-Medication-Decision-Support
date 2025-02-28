from flask import Flask, render_template, request, redirect, url_for, session
import re
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
import tensorflow as tf
from csv import writer
import pandas as pd
from flask_material import Material
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

import joblib
from tensorflow.keras.models import load_model

UPLOAD_FOLDER = 'static/uploads/'

# EDA PKg
import pandas as pd 
import numpy as np 

app = Flask(__name__)
Material(app)

model = joblib.load('DrugRec.pkl')

drug_dict = {
    1: ('Ursodiol', 'Typically used to treat gallstones and primary biliary cirrhosis (PBC), a chronic liver disease.', '250-300 mg twice daily'),
    2: ('Acyclovir', 'Recommended for treating herpes simplex infections (cold sores and genital herpes), chickenpox, and shingles.', '200-800 mg 2 to 5 times daily'),
    3: ('Levofloxacin', 'Used for bacterial infections, including respiratory tract infections, urinary tract infections, and skin infections.', '250-750 mg once daily'),
    4: ('Metformin', 'Primarily prescribed for managing type 2 diabetes by controlling blood sugar levels.', '500-1000 mg 1 to 3 times daily'),
    5: ('Paracetamol', 'Commonly recommended for pain relief and fever reduction in conditions such as headaches, muscle aches, and colds.', '500-1000 mg every 4-6 hours'),
    6: ('Omeprazole', 'Used to treat conditions like gastroesophageal reflux disease (GERD), stomach ulcers, and Zollinger-Ellison syndrome.', '20-40 mg once daily'),
    7: ('Prednisone', 'Indicated for treating inflammation, allergies, autoimmune diseases, and as part of cancer therapy.', '5-60 mg daily'),
    8: ('Loratadine', 'Typically prescribed for allergic rhinitis (hay fever) and chronic urticaria (hives).', '10 mg once daily'),
    9: ('Atovaquone', 'Used for the prevention and treatment of Pneumocystis pneumonia and toxoplasmosis, as well as malaria in combination therapy.', '750-1500 mg once or twice daily'),
    10: ('Fluconazole', 'Recommended for treating fungal infections such as candidiasis, cryptococcal meningitis, and other serious systemic fungal infections.', '150-400 mg daily')
}


@app.route('/')
def index():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        
        if username == "admin" and password == "admin":
            return render_template('index.html')
        else:
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)

@app.route('/upload_image', methods=["POST"])
def upload_image():
    symptoms = [
        'itching', 'Nodal_Skin_Eruptions', 'shivering', 'Stomach_pain', 'Vomiting', 'Chest_pain', 
        'Loss_of_Appetite', 'Yellow_urine', 'Restlessness', 'Excessive_hunger', 'High_fever', 
        'Diarrhoea', 'Red_sports_Over_Body', 'Breathlessness', 'Dark_urine', 'Skin_rash', 
        'Continuous_sneezing', 'Chills', 'Ulcers_On_tounge', 'Cough', 'Yellowish_Skin', 
        'Abdominal_Pain', 'Weight_Loss', 'Irregular_Sugar_level', 'Increased_Appetite', 
        'Headache', 'Muscle_Pain', 'Runny_Nose', 'Fast_Hart_Rate'
    ]
    
    # Extract input features from the form
    input_features = [int(request.form.get(symptom, 0)) for symptom in symptoms]
    
    # Check if all input values are the same (either all 0s or all 1s)
    if all(value == 0 for value in input_features) or all(value == 1 for value in input_features):
        return render_template('contact.html', prediction="Invalid input", description="Please provide a variety of symptoms for an accurate prediction.", dosage="")
    
    input_array = np.array([input_features])
    
    # Make a prediction using the model
    numeric_prediction = model.predict(input_array)[0]
    drug_name, drug_description, drug_dosage = drug_dict.get(numeric_prediction, ("Unknown", "No description available.", "No dosage information available."))
    
    return render_template('contact.html', prediction=drug_name, description=drug_description, dosage=drug_dosage)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
