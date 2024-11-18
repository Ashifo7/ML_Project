import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

def predict_viability_years(phone_specs):
    """
    Predict the viability of the phone for the next number of years based on its specifications.
    :param phone_specs: A dictionary containing phone specifications (e.g., processor, RAM, battery, etc.)
    :return: Predicted number of years the phone will be viable and the path to the graph image
    """
    # Example training data
    training_data = {
        "Processor Speed (GHz)": [2.73, 3.0, 3.36, 3.2, 3.3],
        "RAM (GB)": [8, 6, 12, 8, 16],
        "Battery Capacity (mAh)": [4000, 3500, 4500, 5000, 5500],
        "Camera Megapixels": [12, 12, 50, 48, 108],
        "Price (INR)": [40000, 42000, 45000, 48000, 50000],
        "User Rating (out of 5)": [4.2, 4.5, 4.7, 4.6, 4.8],
        "Viability Years": [2, 3, 4, 5, 6]
    }

    # Create DataFrame from training data
    df = pd.DataFrame(training_data)

    # Define features and target
    X = df[["Processor Speed (GHz)", "RAM (GB)", "Battery Capacity (mAh)", "Camera Megapixels", "Price (INR)", "User Rating (out of 5)"]]
    y = df["Viability Years"]

    # Train the model
    model = LinearRegression()
    model.fit(X, y)

    # Prepare input features
    input_features = pd.DataFrame({
        "Processor Speed (GHz)": [phone_specs["Processor Speed (GHz)"]],
        "RAM (GB)": [phone_specs["RAM (GB)"]],
        "Battery Capacity (mAh)": [phone_specs["Battery Capacity (mAh)"]],
        "Camera Megapixels": [phone_specs["Camera Megapixels"]],
        "Price (INR)": [phone_specs["Price (INR)"]],
        "User Rating (out of 5)": [phone_specs["User Rating (out of 5)"]]
    })

    # Make prediction
    predicted_years = model.predict(input_features)[0]

    # Generate and save the graph
    plt.figure(figsize=(6, 4))
    plt.bar(["Predicted Viability"], [predicted_years], color="blue")
    plt.ylabel("Years of Viability")
    plt.title("Phone Viability Prediction")
    plt.ylim(0, 10)

    # Save the graph
    graph_path = os.path.join("static", "viability_graph.png")
    plt.savefig(graph_path)
    plt.close()

    return round(predicted_years, 2), graph_path
