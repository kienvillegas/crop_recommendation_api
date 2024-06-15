from flask import Flask, request, jsonify
import numpy as np
import pickle

# Load your model
with open('xgboost_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

# Create a mapping dictionary from numerical labels to crop names
crop_mapping = {
    0: "Wheat",
    1: "Rice",
    2: "Maize",
    3: "Barley",
    4: "Soybean",
    5: "Peanut",
    6: "Sunflower",
    7: "Cotton",
    8: "Sugarcane",
    9: "Tobacco",
    10: "Potato",
    11: "Tomato",
    12: "Cabbage",
    13: "Onion",
    14: "Carrot",
    15: "Cauliflower"
}

@app.route('/')
def index():
    return "Hello world"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get values from form data or query parameters
        N = request.form.get("N") or request.args.get("N")
        P = request.form.get("P") or request.args.get("P")
        K = request.form.get("K") or request.args.get("K")
        
        if N is None or P is None or K is None:
            raise ValueError("Missing input values")
        
        # Convert inputs to float
        N = float(N)
        P = float(P)
        K = float(K)
        
        print(f"Received N: {N}, P: {P}, K: {K}")  # Debug print
    except (TypeError, ValueError) as e:
        print(f"Input conversion error: {e}")  # Debug print
        return jsonify({'error': 'Invalid input'}), 400
    
    input_query = np.array([[N, P, K]])
    print(f"Input query: {input_query}")  # Debug print
    result_index = model.predict(input_query)[0]
    print(f"Model result index: {result_index}")  # Debug print

    # Map the numerical result to crop name
    crop_name = crop_mapping.get(result_index, "Unknown crop")
    print(f"Predicted crop name: {crop_name}")  # Debug print
    
    return jsonify({'placement': crop_name})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

