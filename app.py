from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction</title>
    <style>
        body{
            font-family: Arial;
            background:#f4f4f4;
            padding:40px;
        }
        .container{
            max-width:500px;
            margin:auto;
            background:white;
            padding:20px;
            border-radius:10px;
            box-shadow:0 0 10px rgba(0,0,0,0.1);
        }
        input{
            width:100%;
            padding:10px;
            margin:8px 0;
        }
        button{
            width:100%;
            padding:10px;
            background:#007BFF;
            color:white;
            border:none;
            cursor:pointer;
        }
        h2{
            text-align:center;
        }
        .result{
            margin-top:20px;
            text-align:center;
            font-size:20px;
            color:green;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>House Price Prediction</h2>

    <form method="POST">

        <input type="number" name="beds" placeholder="Beds" required>

        <input type="number" name="baths" placeholder="Baths" required>

        <input type="number" name="size" placeholder="Size (sq ft)" required>

        <input type="number" name="lot_size" placeholder="Lot Size" required>

        <input type="number" name="zip_code" placeholder="Zip Code" required>

        <button type="submit">Predict</button>

    </form>

    {% if prediction %}
    <div class="result">
        Predicted Price: {{ prediction }}
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        beds = float(request.form["beds"])
        baths = float(request.form["baths"])
        size = float(request.form["size"])
        lot_size = float(request.form["lot_size"])
        zip_code = float(request.form["zip_code"])

        features = np.array(
            [[beds, baths, size, lot_size, zip_code]]
        )

        pred = model.predict(features)[0]

        prediction = f"${pred:,.2f}"

    return render_template_string(
        HTML,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
