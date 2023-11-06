# Load the model


import pickle5
#import pickle

from flask import Flask
from flask import request
from flask import jsonify

model_file = 'model_h2O_potability.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle5.load(f_in)


app = Flask('water_potability')

@app.route('/predict', methods=['POST'])

def predict():

    water = request.get_json()

    X = dv.transform([water])
    y_pred = model.predict_proba(X)[0, 1]

    #the decision making
    potability = y_pred >= 0.5

    result = {
        'potability_probability': float(y_pred),
        'potability': bool(potability)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)