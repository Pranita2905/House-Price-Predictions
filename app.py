from flask import Flask, request, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
model = pickle.load(open("model.pkl", "rb"))

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction</title>

    <link rel="stylesheet"
    href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">

    <style>

    *{
        margin:0;
        padding:0;
        box-sizing:border-box;
        font-family:'Segoe UI',sans-serif;
    }

    body{
        min-height:100vh;
        display:flex;
        justify-content:center;
        align-items:center;
        background:linear-gradient(135deg,#0f172a,#1e293b,#2563eb);
        padding:20px;
    }

    .container{
        width:100%;
        max-width:900px;
        background:rgba(255,255,255,0.1);
        backdrop-filter:blur(15px);
        border-radius:20px;
        padding:40px;
        box-shadow:0 8px 32px rgba(0,0,0,.3);
        color:white;
    }

    .header{
        text-align:center;
        margin-bottom:30px;
    }

    .header h1{
        font-size:2.5rem;
        margin-bottom:10px;
    }

    .header p{
        color:#dbeafe;
    }

    .form-grid{
        display:grid;
        grid-template-columns:1fr 1fr;
        gap:20px;
    }

    .input-box{
        position:relative;
    }

    .input-box i{
        position:absolute;
        left:15px;
        top:16px;
        color:#2563eb;
    }

    input{
        width:100%;
        padding:14px 14px 14px 45px;
        border:none;
        border-radius:12px;
        outline:none;
        font-size:15px;
    }

    .btn{
        margin-top:25px;
        width:100%;
        padding:15px;
        border:none;
        border-radius:12px;
        font-size:18px;
        font-weight:bold;
        cursor:pointer;
        background:#2563eb;
        color:white;
        transition:0.3s;
    }

    .btn:hover{
        background:#1d4ed8;
        transform:translateY(-2px);
    }

    .result{
        margin-top:30px;
        padding:25px;
        border-radius:15px;
        text-align:center;
        background:rgba(255,255,255,0.15);
    }

    .result h2{
        margin-bottom:10px;
    }

    .price{
        font-size:2rem;
        color:#4ade80;
        font-weight:bold;
    }

    .footer{
        margin-top:30px;
        text-align:center;
        color:#cbd5e1;
        font-size:14px;
    }

    @media(max-width:768px){
        .form-grid{
            grid-template-columns:1fr;
        }

        .header h1{
            font-size:2rem;
        }
    }

    </style>
</head>

<body>

<div class="container">

    <div class="header">
        <h1><i class="fas fa-house"></i> House Price Prediction</h1>
        <p>Machine Learning Powered Real Estate Valuation System</p>
    </div>

    <form method="POST">

        <div class="form-grid">

            <div class="input-box">
                <i class="fas fa-bed"></i>
                <input type="number" name="beds"
                placeholder="Number of Bedrooms" required>
            </div>

            <div class="input-box">
                <i class="fas fa-bath"></i>
                <input type="number" name="baths"
                placeholder="Number of Bathrooms" required>
            </div>

            <div class="input-box">
                <i class="fas fa-ruler-combined"></i>
                <input type="number" name="size"
                placeholder="House Size (sq ft)" required>
            </div>

            <div class="input-box">
                <i class="fas fa-map"></i>
                <input type="number" name="lot_size"
                placeholder="Lot Size" required>
            </div>

            <div class="input-box">
                <i class="fas fa-location-dot"></i>
                <input type="number" name="zip_code"
                placeholder="Zip Code" required>
            </div>

        </div>

        <button type="submit" class="btn">
            <i class="fas fa-chart-line"></i>
            Predict House Price
        </button>

    </form>

    {% if prediction %}

    <div class="result">
        <h2>Estimated Property Value</h2>
        <div class="price">
            ${{ prediction }}
        </div>
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

        prediction = format(round(pred, 2), ",")

    return render_template_string(
        HTML,
        prediction=prediction
    )

if __name__ == "__main__":
    app.run(debug=True)
