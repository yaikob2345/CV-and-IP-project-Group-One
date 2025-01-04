from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load dataset and model
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    # Prepare unique values for dropdowns
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        size = request.form.get('size')
        zipcode = request.form.get('zip_code')

        # Validate and convert input values to appropriate types
        bedrooms = int(bedrooms) if bedrooms else 0
        bathrooms = float(bathrooms) if bathrooms else 0.0
        size = float(size) if size else 0.0
        zipcode = int(zipcode) if zipcode else 0

        # Create input DataFrame for the model
        input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                                  columns=['beds', 'baths', 'size', 'zip_code'])

        print("Input Data:")
        print(input_data)

        # Handle unknown categories by replacing with the mode of the dataset
        for column in input_data.columns:
            unknown_categories = set(input_data[column]) - set(data[column].unique())
            if unknown_categories:
                print(f"Unknown categories in {column}: {unknown_categories}")
                input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

        print("Processed Input Data:")
        print(input_data)

        # Predict the house price
        prediction = pipe.predict(input_data)[0]

        return jsonify({'prediction': f"Predicted Price: {prediction}"})

    except Exception as e:
        # Handle errors and return a message
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
