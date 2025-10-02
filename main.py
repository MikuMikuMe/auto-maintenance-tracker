Creating an auto-maintenance tracker using machine learning involves several steps. I'll provide a basic Python program outline for building such an application. The program will:

1. Use a simple dataset of vehicles and their maintenance history.
2. Train a machine learning model to predict the next maintenance date/type based on historical data.
3. Provide functionality to query maintenance schedules.

Due to the complexity and breadth of this task, we'll be using a simplified example. The machine learning part will utilize sklearn's random forest for demonstration purposes.

Here's a complete example:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta
import logging

# Setting up logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Sample dataset: 'vehicle_id', 'last_maintenance_date', 'mileage_since_last', 'issue_detected', 'next_service_type'
data = {
    'vehicle_id': ['V001', 'V002', 'V003', 'V004'],
    'last_maintenance_date': ['2023-01-10', '2022-12-15', '2023-02-02', '2023-03-21'],
    'mileage_since_last': [1500, 3000, 2000, 1000],
    'issue_detected': [0, 1, 0, 1],  # 0 - No issue, 1 - Issue Detected
    'next_service_type': [0, 1, 0, 1]  # 0 - Basic, 1 - Comprehensive
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocess the data
def preprocess_data(df):
    try:
        df['last_maintenance_date'] = pd.to_datetime(df['last_maintenance_date'])
        df['days_since_last'] = (datetime.now() - df['last_maintenance_date']).dt.days
        return df.drop(['vehicle_id', 'last_maintenance_date'], axis=1)
    except Exception as e:
        logging.error("Error in preprocessing data: %s", e)
        return None

# Prepare data for training
try:
    df_processed = preprocess_data(df)
    if df_processed is None:
        raise ValueError("Processed DataFrame is None, check preprocessing step")
    
    X = df_processed[['mileage_since_last', 'issue_detected', 'days_since_last']]
    y = df_processed['next_service_type']

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except Exception as e:
    logging.error("Error preparing data for training: %s", e)

# Train the model
try:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training completed.")
except Exception as e:
    logging.error("Error during model training: %s", e)

# Evaluate the model
try:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logging.info("Model accuracy: %.2f", accuracy)
except Exception as e:
    logging.error("Error during model evaluation: %s", e)

# Function to predict the next service type
def predict_next_service(mileage_since_last, issue_detected, last_maintenance_date):
    try:
        days_since_last = (datetime.now() - pd.to_datetime(last_maintenance_date)).days
        input_data = pd.DataFrame({
            'mileage_since_last': [mileage_since_last],
            'issue_detected': [issue_detected],
            'days_since_last': [days_since_last]
        })
        prediction = model.predict(input_data)[0]
        service_type = "Basic" if prediction == 0 else "Comprehensive"
        logging.info("Next service type prediction: %s", service_type)
        return service_type
    except Exception as e:
        logging.error("Error predicting next service type: %s", e)
        return "Error in prediction"

# Example query
vehicle_id = 'V005'
last_maintenance_date = '2023-04-01'
mileage_since_last = 2500
issue_detected = 1

predicted_service = predict_next_service(mileage_since_last, issue_detected, last_maintenance_date)
print(f"Predicted next service for vehicle {vehicle_id} is {predicted_service}.")
```

### Key Points:
- **Dataset Preparation**: This example uses a small hardcoded dataset. In a real-world application, you would use a much larger and more detailed dataset, possibly stored in a database.
- **Preprocessing**: Convert dates to a suitable format and calculate derived features such as `days_since_last`.
- **Machine Learning Model**: We use a simple Random Forest classifier. Depending on the application, you might need to explore other models and fine-tune hyperparameters.
- **Error Handling and Logging**: Integrate error handling and logging to make the application robust and easier to debug.

### Next Steps for Real-world Application:
- Interface with a database to store vehicle data and retrieve historical records.
- Implement a more sophisticated machine learning pipeline with regular updates as more data becomes available.
- Develop a user-friendly interface to interact with the system (e.g., a web interface with Flask or Django).
- Integrate vehicle telematics for real-time data acquisition.