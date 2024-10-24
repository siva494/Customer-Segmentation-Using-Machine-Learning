from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load project descriptions
with open("../notebooks/descriptions.json") as f:
    segment_descriptions = json.load(f)

segment_descriptions = pd.DataFrame(segment_descriptions.values(), index=segment_descriptions.keys(), columns=["description"])

# Load the model and scaler
model = joblib.load("../models/model.joblib")
scaler = MinMaxScaler()
scaler.min_, scaler.scale_ = np.load('minmax_scaler_params.npy')

# Encoding mappings
encodings = {
    'Married': {'Yes': 1, 'No': 0},
    'Graduated': {'Yes': 1, 'No': 0},
    'Gender': {"Female": 0, "Male": 1},
    'Profession': {'Artist': 0, 'Doctor': 1, 'Engineer': 2, 'Entertainment': 3, 'Executive': 4, 'Healthcare': 5, "Lawyer": 6, "Other": 7},
    'Spending Score': {'Low': 2, 'Average': 0, 'High': 1}
}

# List of features and column names
num_features = ['Family Size', 'Age', 'Work Experience']
cat_features = ['Spending Score', 'Profession', 'Gender', 'Graduated', 'Married']
columns = ['Family_Size', 'Age', 'Work_Experience', 'Spending_Score',
           'Profession_Artist', 'Profession_Doctor', 'Profession_Engineer',
           'Profession_Entertainment', 'Profession_Executive',
           'Profession_Healthcare', 'Profession_Lawyer', 'Profession_Other',
           'Gender', 'Graduated', 'Ever_Married']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    age = request.form['age']
    married = request.form['married']
    work_experience = request.form['work_experience']
    graduated = request.form['graduated']
    work_profession = request.form['profession']
    gender = request.form['gender']
    spending_score = request.form['spending_score']
    family_size = request.form['family_size']

    # Prepare the input dictionary
    inputs = {
        'Family Size': family_size,
        'Age': age,
        'Work Experience': work_experience,
        'Spending Score': spending_score,
        'Profession': work_profession,
        'Gender': gender,
        'Graduated': graduated,
        'Married': married
    }

    # Preprocess numerical inputs
    num_inputs = {k: v for k, v in inputs.items() if k in num_features}
    num_df = pd.DataFrame.from_dict(num_inputs, orient='index').T
    scaled_inputs = scaler.transform(num_df)
    num_df = pd.DataFrame(scaled_inputs)

    # Process categorical inputs
    num_professions = len(set(encodings['Profession'].values()))
    num_onehot_encoded_features = 1
    cat_df = np.zeros((1, len(cat_features) - num_onehot_encoded_features + num_professions))

    for i, feature in enumerate(cat_features):
        if feature == 'Spending Score':
            cat_df[0, i] = encodings[feature][inputs[feature]]
        elif feature == 'Profession':
            profession = np.zeros(num_professions)
            profession[encodings[feature][inputs[feature]]] = 1
            cat_df[:, i:i+num_professions] = profession.reshape(1, num_professions)
        elif feature in ['Married', 'Graduated', 'Gender']:
            cat_df[0, i+num_professions-num_onehot_encoded_features] = encodings[feature][inputs[feature]]
    
    cat_df = pd.DataFrame(cat_df)
    predict_df = pd.concat([num_df, cat_df.add_suffix('_2')], axis=1)
    
    # Rename columns
    predict_df.columns = columns

    # Make the prediction
    prediction = model.predict(predict_df)
    
    # Get the description for the predicted segment
    predicted_segment = prediction[0]
    description = segment_descriptions.loc[predicted_segment]['description']

    # Return the result to the user
    return render_template('result.html', segment=predicted_segment, description=description)

if __name__ == '__main__':
    app.run(debug=True)
