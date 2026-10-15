import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained linear regression model
with open('linear.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML template with embedded modern CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>House Price Prediction</title>
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f3f4f6;
            --card-bg: #ffffff;
            --text-color: #1f2937;
            --border-color: #d1d5db;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: var(--card-bg);
            padding: 30px 40px;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            width: 100%;
            max-width: 650px;
        }
        h2 {
            text-align: center;
            color: var(--primary);
            margin-bottom: 25px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group.full-width {
            grid-column: span 2;
        }
        label {
            display: block;
            margin-bottom: 6px;
            font-weight: 600;
            font-size: 0.9rem;
        }
        input {
            width: 100%;
            padding: 10px 12px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            font-size: 1rem;
            box-sizing: border-box;
            transition: border-color 0.2s, box-shadow 0.2s;
        }
        input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }
        button {
            width: 100%;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 12px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }
        button:hover {
            background-color: var(--primary-hover);
        }
        .result-card {
            margin-top: 25px;
            background-color: #e0e7ff;
            border: 1px solid #c7d2fe;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .result-card h3 {
            margin: 0;
            color: #3730a3;
            font-size: 1.25rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>House Price Prediction</h2>
        <form method="POST">
            <div class="form-grid">
                <div class="form-group">
                    <label for="square_footage">Square Footage</label>
                    <input type="number" step="any" id="square_footage" name="square_footage" placeholder="e.g., 2000" required>
                </div>
                <div class="form-group">
                    <label for="num_bedrooms">Num Bedrooms</label>
                    <input type="number" id="num_bedrooms" name="num_bedrooms" placeholder="e.g., 3" required>
                </div>
                <div class="form-group">
                    <label for="num_bathrooms">Num Bathrooms</label>
                    <input type="number" step="any" id="num_bathrooms" name="num_bathrooms" placeholder="e.g., 2" required>
                </div>
                <div class="form-group">
                    <label for="year_built">Year Built</label>
                    <input type="number" id="year_built" name="year_built" placeholder="e.g., 2010" required>
                </div>
                <div class="form-group">
                    <label for="lot_size">Lot Size</label>
                    <input type="number" step="any" id="lot_size" name="lot_size" placeholder="e.g., 5000" required>
                </div>
                <div class="form-group">
                    <label for="garage_size">Garage Size</label>
                    <input type="number" step="any" id="garage_size" name="garage_size" placeholder="e.g., 2" required>
                </div>
                <div class="form-group full-width">
                    <label for="neighborhood_quality">Neighborhood Quality</label>
                    <input type="number" step="any" id="neighborhood_quality" name="neighborhood_quality" placeholder="e.g., 8" required>
                </div>
            </div>
            <button type="submit">Predict Price</button>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Estimated Price: ${{ prediction }}</h3>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def index():
  prediction = None
  if request.method == 'POST':
    try:
      # Collecting inputs matching model features
      sq_ft = float(request.form['square_footage'])
      bedrooms = float(request.form['num_bedrooms'])
      bathrooms = float(request.form['num_bathrooms'])
      year_built = float(request.form['year_built'])
      lot_size = float(request.form['lot_size'])
      garage_size = float(request.form['garage_size'])
      neighborhood_quality = float(request.form['neighborhood_quality'])

      # Arrange features into an array matching the training layout
      features = np.array([[
          sq_ft,
          bedrooms,
          bathrooms,
          year_built,
          lot_size,
          garage_size,
          neighborhood_quality,
      ]])

      pred = model.predict(features)[0]
      prediction = f'{pred:,.2f}'
    except Exception as e:
      prediction = f'Error: {str(e)}'

  return render_template_string(HTML_TEMPLATE, prediction=prediction)


if __name__ == '__main__':
  app.run(debug=True)
