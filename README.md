# Anomaly Detection API for Android Launcher

This is the backend server for my thesis project, the "Android Launcher for Behavioural Profile." It uses a TensorFlow autoencoder model to learn a user's app usage patterns and detect anomalies in real-time.

The API receives usage data from the Android app, trains a personalized model, and provides an anomaly score for new activity.

## Tech Stack

-   Python 3
-   FastAPI
-   TensorFlow / Keras
-   Uvicorn (for serving)
-   scikit-learn, pandas, numpy (for data handling)

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
- POST /train: Trains the autoencoder model with the app usage data.
- POST /detect: Receives recent usage data and returns an anomaly score and risk level based on the trained model.
- POST /generate-sample-data: Create sample data for testing the training process.
