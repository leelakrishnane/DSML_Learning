# app.py
from flask import Flask, request, jsonify, render_template
from ml_api import predict_cancer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("form.html")

@app.route("/predict_form", methods=["POST"])
def predict_form():
    try:
        # Extract 30 input values from HTML form
        features = []
        for i in range(30):
            value = float(request.form.get(f"f{i}"))
            features.append(value)

        result, pred_int = predict_cancer(features)

        return render_template("form.html",
                               prediction=result,
                               raw_pred=pred_int)

    except Exception as e:
        return render_template("form.html", error=str(e))

if __name__ == "__main__":
    app.run(debug=True)
