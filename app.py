from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Define phone data
data_dict = {
    "Feature": ["RAM", "Storage (ROM)", "Processor Speed (GHz)", "Battery", "Display", "Camera", "Refresh Rate (Hz)", "Charging Speed (W)", "Weight (g)", "Rating", "Price (₹)"],
    "Samsung Galaxy S23 5G": [8, 128, 3.36, 3900, 6.1, 50, 120, 25, 168, 4.5, 40635],
    "Apple iPhone 13": [4, 128, 3.1, 3240, 6.1, 12, 60, 20, 174, 4.6, 46216],
    "Xiaomi 14": [12, 512, 3.3, 4610, 6.36, 50, 120, 90, 193, 4.7, 47972],
    "Xiaomi 14 Civi": [8, 256, 2.84, 4700, 6.55, 50, 120, 67, 172, 4.4, 47660],
}
data = pd.DataFrame(data_dict).set_index("Feature").transpose()

# Prediction Function
def predict_viability_years(phone_specs):
    training_data = {
        "Processor Speed (GHz)": [2.73, 3.0, 3.36, 3.2, 3.3],
        "RAM (GB)": [8, 6, 12, 8, 16],
        "Battery Capacity (mAh)": [4000, 3500, 4500, 5000, 5500],
        "Camera Megapixels": [12, 12, 50, 48, 108],
        "Price (INR)": [40000, 42000, 45000, 48000, 50000],
        "User Rating (out of 5)": [4.2, 4.5, 4.7, 4.6, 4.8],
        "Viability Years": [2, 3, 4, 5, 6]
    }

    df = pd.DataFrame(training_data)
    X = df[["Processor Speed (GHz)", "RAM (GB)", "Battery Capacity (mAh)", "Camera Megapixels", "Price (INR)", "User Rating (out of 5)"]]
    y = df["Viability Years"]

    model = LinearRegression()
    model.fit(X, y)

    input_features = pd.DataFrame({
        "Processor Speed (GHz)": [phone_specs["Processor Speed (GHz)"]],
        "RAM (GB)": [phone_specs["RAM"]],
        "Battery Capacity (mAh)": [phone_specs["Battery"]],
        "Camera Megapixels": [50],  # Assuming 50 MP for simplicity
        "Price (INR)": [phone_specs["Price (₹)"]],
        "User Rating (out of 5)": [phone_specs.get("Rating", 4.5)]  # Default rating
    })

    predicted_years = model.predict(input_features)[0]
    return round(predicted_years, 2)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            price_min = int(request.form["price_min"])
            price_max = int(request.form["price_max"])
            usage_type = request.form["usage_type"]

            # User preferences
            user_preferences = {
                feature: int(request.form.get(feature, 0))
                for feature in ["RAM", "Storage (ROM)", "Battery", "Display", "Camera"]
            }

            # Filter phones within price range
            filtered_data = data[(data["Price (₹)"] >= price_min) & (data["Price (₹)"] <= price_max)].copy()
            if filtered_data.empty:
                return render_template("index.html", error="No phones found within this price range.")

            # Normalize relevant numeric columns
            numeric_columns = ["RAM", "Storage (ROM)", "Battery", "Display", "Camera", "Processor Speed (GHz)", "Refresh Rate (Hz)", "Rating"]
            scaler = MinMaxScaler()
            normalized_features = scaler.fit_transform(filtered_data[numeric_columns])
            normalized_df = pd.DataFrame(normalized_features, columns=numeric_columns, index=filtered_data.index)

            # System weights for usage type
            weights = {
                "gaming": {"RAM": 0.2, "Processor Speed (GHz)": 0.25, "Battery": 0.15, "Refresh Rate (Hz)": 0.2, "Rating": 0.2},
                "casual": {"Display": 0.2, "Camera": 0.15, "Battery": 0.15, "RAM": 0.1, "Rating": 0.4}
            }
            system_weights = weights.get(usage_type, weights["casual"])
            system_scores = normalized_df[list(system_weights.keys())].multiply(list(system_weights.values())).sum(axis=1)

            # User recommendation scores
            user_scores = normalized_df.multiply(pd.Series(user_preferences), axis=1).sum(axis=1)

            # Top recommendations
            filtered_data["User Recommendation Score"] = user_scores
            filtered_data["System Recommendation Score"] = system_scores

            top_user_recommendation_row = filtered_data.sort_values(by="User Recommendation Score", ascending=False).iloc[0]
            top_system_recommendation_row = filtered_data.sort_values(by="System Recommendation Score", ascending=False).iloc[0]

            # Extract details
            top_user_recommendation = top_user_recommendation_row.name
            user_score = round(top_user_recommendation_row["User Recommendation Score"], 2)
            top_system_recommendation = top_system_recommendation_row.name
            system_score = round(top_system_recommendation_row["System Recommendation Score"], 2)

            # Viability prediction
            phone_specs = filtered_data.loc[top_system_recommendation]
            viability_years = predict_viability_years({
                "Processor Speed (GHz)": phone_specs["Processor Speed (GHz)"],
                "RAM": phone_specs["RAM"],
                "Battery": phone_specs["Battery"],
                "Price (₹)": phone_specs["Price (₹)"],
                "Rating": phone_specs["Rating"]
            })

            return render_template(
                "index.html",
                top_user_recommendation=top_user_recommendation,
                user_score=user_score,
                top_system_recommendation=top_system_recommendation,
                system_score=system_score,
                viability_years=viability_years
            )
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
