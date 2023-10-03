from flask import Flask,redirect,url_for,render_template,request
#from datetime import timedelta
from datetime import datetime
import plotly.express as px
import pandas as pd
import numpy as np
import pickle

app=Flask(__name__)
data =pd.read_csv('datae.csv')
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html',data=data)
    

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        return render_template('predict.html')

    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get input data from the for
        vrms_min = float(request.form.get('vrms_min'))
        vrms_avg = float(request.form.get('vrms_avg'))
        vrms_max = float(request.form.get('vrms_max'))
        current_min = float(request.form.get('current_min'))
        current_avg = float(request.form.get('current_avg'))
        current_max = float(request.form.get('current_max'))
       
        # Create a DataFrame with the input data
        #input_data=np.array([[vrms_min,vrms_avg,vrms_max,current_min,current_avg,current_max]])
        input_data = pd.DataFrame({
            
            'Vrms ph-n L1N Min': [vrms_min],
            'Vrms ph-n L1N Avg': [vrms_avg],
            'Vrms ph-n L1N Max': [vrms_max],
            'Current L1 Min': [current_min],
            'Current L1 Avg': [current_avg],
            'Current L1 Max': [current_max]
        })

        # Use the pre-trained model to make predictions
        # scaler = StandardScaler()
        # input_data=np.array(input_data)
        # user_input_scaled = scaler.fit_transform(input_data)
        # input_data = user_input_scaled.reshape(1, input_data.shape[1], 1)
        
        predicted_consumption = model.predict(input_data)

        return render_template('predict.html', prediction_result=predicted_consumption[0])

    return redirect(url_for('prediction'))
@app.route('/dashboard')
def dashboard():
    start_time, end_time = data['Time'].min(), data['Time'].max()
    filtered_df = data[(data['Time'] >= start_time) & (data['Time'] <= end_time)]

    # Create a Line chart using Plotly
    fig_line = px.line(filtered_df, x='Time', y=['Current L1 Min', 'Current L1 Avg', 'Current L1 Max'],
                       color_discrete_sequence=['blue', 'green', 'red'], title='Line Chart')

    # Create an Area chart using Plotly
    fig_area = px.area(filtered_df, x='Time', y='Current L1 Min', title='Area Chart')

    # Convert Plotly charts to JSON
    line_chart_json = fig_line.to_json()
    area_chart_json = fig_area.to_json()

    return render_template('dashbord.html', line_chart=line_chart_json, area_chart=area_chart_json)

if __name__ == '__main__':
    app.run(debug=True)