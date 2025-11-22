from flask import Flask, request, jsonify
import pandas as pd
import pickle

with open("uptor_linear_trained_model.pkl", "rb") as file_reading_obj:
    train_model = pickle.load(file_reading_obj)

my_app = Flask(__name__)


@my_app.route('/linear_model_predict', methods=['POST'])
def linear_predicttion():
    data = request.get_json()
    year_from_user = data.get("year")
    if not year_from_user or not isinstance(year_from_user, list):
        return jsonify({"error": "Invalid input, please validate the input."}), 400
    data_input = pd.DataFrame({"year": year_from_user})
    predicted_output = train_model.predict(data_input[["year"]])
    return jsonify({'data': predicted_output.tolist()})

@my_app.route('/')
def landing():
    return "Welcome to my Flask app!"

@my_app.route('/login')
def login():
    return "This is the login page."


if __name__ == '__main__':
    my_app.run(debug=True, port=5050)


