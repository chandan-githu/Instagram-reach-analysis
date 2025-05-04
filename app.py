from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

# Initialize the Flask application
app = Flask(__name__)

# Check if the model file exists before loading
MODEL_PATH = "engagement_model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Ensure it is in the correct directory.")

# Define the route for the homepage
@app.route('/')
def index():
    return render_template('index.html', form_data={})

# Define the route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        form_data = {
            'caption': request.form.get('caption', '').strip(),
            'hashtags': request.form.get('hashtags', '').strip(),
            'hour': request.form.get('hour', ''),
            'day': request.form.get('day', '').strip(),
            'month': request.form.get('month', ''),
            'year': request.form.get('year', ''),
            'caption_length': request.form.get('caption_length', ''),
            'num_hashtags': request.form.get('num_hashtags', '')
        }

        # Ensure all numeric inputs are valid
        numeric_fields = ['hour', 'month', 'year', 'caption_length', 'num_hashtags']
        for field in numeric_fields:
            if not form_data[field] or not form_data[field].isdigit():
                return render_template('index.html', error=f"Invalid input for {field}. Please enter a valid number.", form_data=form_data)

        # Convert numeric inputs to integers
        form_data['hour'] = int(form_data['hour'])
        form_data['month'] = int(form_data['month'])
        form_data['year'] = int(form_data['year'])
        form_data['caption_length'] = int(form_data['caption_length'])
        form_data['num_hashtags'] = int(form_data['num_hashtags'])

        # Ensure caption length is at least 50
        if form_data['caption_length'] < 50:
            return render_template('index.html', error="Caption must be at least 50 characters long.", form_data=form_data)

        # Create a DataFrame for input data
        new_data = pd.DataFrame([form_data])

        # Predict engagement category (low, medium, high)
        prediction = model.predict(new_data)

        return render_template('index.html', predicted_engagement=prediction[0], form_data=form_data)

    except Exception as e:
        return render_template('index.html', error=f"Error: {str(e)}", form_data=form_data)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
