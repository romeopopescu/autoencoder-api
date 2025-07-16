# Anomaly Detection API for Android Launcher

This is the backend server for my thesis project, the "Android Launcher for Behavioural Profile." It uses a TensorFlow autoencoder model to learn a user's app usage patterns and detect anomalies in real-time.

The API receives usage data from the Android app, trains a personalized model, and provides an anomaly score for new activity.

The anomaly score is calculated using the Autoencoder neural network.

## Tech Stack

-   Python 3
-   FastAPI
-   TensorFlow / Keras
-   Uvicorn
-   scikit-learn, pandas, numpy

---
## Setup

First, install the required Python packages.

```bash
pip install fastapi uvicorn tensorflow scikit-learn pandas numpy
```

## Running the Server

To start the server, run the next command from the project's root directory. The server will be accessible on port 8000.

```bash
uvicorn neural-network:app --host 0.0.0.0 --port 8000
```

## API Endpoints
The main endpoints are:
- GET /: See the current status of the model
- GET /docs: FastApi Swagger UI to see all available endpoints and values.
- POST /train: Trains the autoencoder model with the app usage data. Example of request body:
```json
{
  "usage_data": [
    {
      "app": "com.whatsapp",
      "date": "2025-07-16",
      "first_hour": 9,
      "last_hour": 22,
      "launch_count": 50,
      "total_time_in_foreground": 3600
    },
    {
      "app": "com.android.chrome",
      "date": "2025-07-16",
      "first_hour": 10,
      "last_hour": 18,
      "launch_count": 15,
      "total_time_in_foreground": 1800
    }
  ],
  "epochs": 50,
  "validation_split": 0.2
}
```
- POST /detect: Receives recent usage data and returns an anomaly score and risk level based on the trained model.
Example of request body:
```json
{
  "usage_data": [
    {
      "app": "com.whatsapp",
      "date": "2025-07-16",
      "first_hour": 19,
      "last_hour": 19,
      "launch_count": 5,
      "total_time_in_foreground": 300
    }
  ]
}
```
- POST /generate-sample-data: Create sample data for testing the training process.



