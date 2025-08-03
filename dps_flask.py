from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model and encoders
with open("diseasepredictjsonfile.pkl", "rb") as f:
    model, feature_encoders, target_encoder = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    ans = None
    error = None

    if request.method == "POST":
        try:
            # Get form values
            age = request.form.get("Age")
            gender = request.form.get("Gender")
            symptoms = request.form.get("Symptoms")
            duration = request.form.get("Duration")
            family = request.form.get("Family History")
            smoking = request.form.get("Smoking History")
            alcohol = request.form.get("Alcohol Consumption")
            blood = request.form.get("Blood Pressure Level")
            sugar = request.form.get("Blood Sugar Level")
            cholestrol = request.form.get("Cholesterol Level")
            bodytm = request.form.get("Body Temperature")
            heartrt = request.form.get("Heart Rate")

            # Validate numeric fields
            if not age or not duration or not bodytm or not heartrt:
                error = "Please fill all numeric fields!"
                return render_template("dps.html", ans=error)

            age = int(age)
            duration = int(duration)
            bodytm = float(bodytm)
            heartrt = int(heartrt)

            # Encode categorical features
            gender_en = feature_encoders["Gender"].transform([gender])[0]
            symptoms_en = feature_encoders["Symptoms"].transform([symptoms])[0]
            family_en = feature_encoders["Family History"].transform([family])[0]
            smoking_en = feature_encoders["Smoking History"].transform([smoking])[0]
            alcohol_en = feature_encoders["Alcohol Consumption"].transform([alcohol])[0]
            blood_en = feature_encoders["Blood Pressure Level"].transform([blood])[0]
            sugar_en = feature_encoders["Blood Sugar Level"].transform([sugar])[0]
            cholestrol_en = feature_encoders["Cholesterol Level"].transform([cholestrol])[0]

            # Combine features
            features = np.array([[age, gender_en, symptoms_en, duration, family_en,
                                   smoking_en, alcohol_en, blood_en, sugar_en,
                                   cholestrol_en, bodytm, heartrt]])

            # Predict
            prediction = model.predict(features)[0]

            # Decode prediction back to disease name
            ans = target_encoder.inverse_transform([prediction])[0]

        except Exception as e:
            ans = f"Prediction Error: {str(e)}"

    return render_template("dps.html", ans=ans)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
