from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained model
with open('c:\Users\Sumukha\Downloads\lasso_model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('2.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    features = [
        request.form['YEAR'],
        request.form['MONTH'],
        request.form['DAY'],
        request.form['DAY_OF_WEEK'],
        request.form['FLIGHT_NUMBER'],
        request.form['TAIL_NUMBER'],
        request.form['ORIGIN_AIRPORT'],
        request.form['DESTINATION_AIRPORT'],
        request.form['SCHEDULED_DEPARTURE'],
        request.form['DEPARTURE_TIME'],
        request.form['DEPARTURE_DELAY'],
        request.form['TAXI_OUT'],
        request.form['WHEELS_OFF'],
        request.form['SCHEDULED_TIME'],
        request.form['ELAPSED_TIME'],
        request.form['AIR_TIME'],
        request.form['DISTANCE'],
        request.form['WHEELS_ON'],
        request.form['TAXI_IN'],
        request.form['SCHEDULED_ARRIVAL'],
        request.form['ARRIVAL_TIME'],
        request.form['ARRIVAL_DELAY'],
        request.form['DIVERTED'],
        request.form['CANCELLED'],
        request.form['DELAY_LEVEL'],
        request.form['Actual_Departure'],
        request.form['Date'],
        request.form['Scheduled_Arrival'],
        request.form['Scheduled_Departure'],
        request.form['Actual_Arrival'],
        request.form['AIRLINE'],
        request.form['IATA_CODE_x'],
        request.form['AIRPORT_x'],
        request.form['CITY_x'],
        request.form['STATE_x'],
        request.form['COUNTRY_x'],
        request.form['LATITUDE_x'],
        request.form['LONGITUDE_x'],
        request.form['IATA_CODE_y'],
        request.form['AIRPORT_y'],
        request.form['CITY_y'],
        request.form['STATE_y'],
        request.form['COUNTRY_y'],
        request.form['LATITUDE_y'],
        request.form['LONGITUDE_y']
    ]

    # Convert features to the appropriate type (e.g., float)
    features = np.array(features, dtype=float).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    return f'Predicted value: {prediction[0]}'

if __name__ == '__main__':
    app.run(debug=True)
