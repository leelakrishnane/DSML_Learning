from flask import Flask, jsonify, request
import pandas as pd
import pickle

with open("diamonds_linear_trained_model.pkl", "rb") as file_reading_obj:
    train_model = pickle.load(file_reading_obj)

diamond_app = Flask(__name__)   

@diamond_app.route('/linear_model_predict', methods=['POST'])
def linear_predicttion():
    data = request.get_json()
    
    data_from_user = data.get("carat","cut","depth","table","x","y","z")
    if not data_from_user or not isinstance(data_from_user, list):
        return jsonify({"error": "Invalid input, please validate the input."}), 400
    data_input = pd.DataFrame({"carat": data_from_user[0], "cut": data_from_user[1], "depth": data_from_user[2], "table": data_from_user[3], "x": data_from_user[4], "y": data_from_user[5], "z": data_from_user[6]})
    predicted_output = train_model.predict(data_input)
    return jsonify({'data': predicted_output.tolist()})

if __name__ == '__main__':
    diamond_app.run(debug=True, port=5060)