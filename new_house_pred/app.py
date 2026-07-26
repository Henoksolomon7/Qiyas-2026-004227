import sqlite3
from datetime import datetime
from pathlib import Path



from flask import Flask, jsonify, request, render_template,flash, redirect

import joblib
app = Flask(__name__)
app.secret_key = "secret"

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "house_predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            rooms INTEGER NOT NULL,
            site_area REAL NOT NULL,
            built_area REAL NOT NULL,
            property_years INTEGER NOT NULL,
            proximity_cbd REAL NOT NULL,
            proximity_bus REAL NOT NULL,
            proximity_schools REAL NOT NULL,
            housing_typology TEXT NOT NULL,
            land_value_grading TEXT NOT NULL,
            road_type TEXT NOT NULL,
            construction_material TEXT NOT NULL,
            predicted_price REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()

try:
    model = joblib.load('gbm.pkl')
    
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model files: {e}")



@app.route('/')
def home():
    return render_template('index.html')


@app.route('/history')
def history():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    predictions = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return render_template('History.html', predictions=predictions)


@app.route('/house_price',methods=['POST'])
def house_price():
 
   # Numerical Inputs
    rooms = int(request.form["Number_of_Rooms"])
    site = float(request.form["Site_Area_sqm"])
    built = float(request.form["Built_Area_sqm"])
    years = int(request.form["Property_Years"])
    cbd = float(request.form["Proximity_to_CBD_km"])
    bus = float(request.form["Proximity_to_Bus_Station_km"])
    school = float(request.form["Proximity_to_Schools_km"])

    # Categorical Inputs
    housing = request.form["Housing_Typology"]
    land = request.form["Land_Value_Grading"]
    road = request.form["Type_of_Nearest_Road"]
    material = request.form["Construction_Materials"]

    # -------- One-Hot Encoding --------

    # Housing Typology
    housing_condo = 1 if housing == "Condominium" else 0
    housing_detached = 1 if housing == "Detached" else 0
    housing_semi = 1 if housing == "Semi-detached" else 0

    # Land Value
    land_high = 1 if land == "High" else 0
    land_low = 1 if land == "Low" else 0
    land_medium = 1 if land == "Medium" else 0

    # Road Type
    road_asphalt = 1 if road == "Asphalt" else 0
    road_gravel = 1 if road == "Gravel" else 0

    # Construction Material
    material_concrete = 1 if material == "Concrete" else 0
    material_mudwood = 1 if material == "Mud&Wood" else 0

    # Arrange in the SAME ORDER used during training
    features = [[
        rooms,
        site,
        built,
        years,
        cbd,
        bus,
        school,
        housing_condo,
        housing_detached,
        housing_semi,
        land_high,
        land_low,
        land_medium,
        road_asphalt,
        road_gravel,
        material_concrete,
        material_mudwood
    ]]

    # Predict
    #features_scaled = scaler.transform(features)

    prediction = model.predict(features)[0]
    predicted_price = round(float(prediction), 2)
    print(predicted_price)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO predictions (
            created_at, rooms, site_area, built_area, property_years,
            proximity_cbd, proximity_bus, proximity_schools,
            housing_typology, land_value_grading, road_type,
            construction_material, predicted_price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            rooms, site, built, years,
            cbd, bus, school,
            housing, land, road, material,
            predicted_price
        )
    )
    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        predicted_price=predicted_price
    )



if __name__ == "__main__":
    app.run(debug=True,port=5001)