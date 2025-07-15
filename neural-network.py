from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import joblib
import os
from datetime import datetime
import json
import logging

# dai run cu asta:
# uvicorn neural-network:app --host 0.0.0.0 --port 8000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Behavioral Anomaly Detection API",
    description="Android launcher security API for detecting anomalous app usage patterns",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AppUsageData(BaseModel):
    app: str
    date: str  
    first_hour: int
    last_hour: int
    launch_count: int
    total_time_in_foreground: int  
    
    @validator('first_hour', 'last_hour')
    def validate_hours(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('Hours must be between 0 and 23')
        return v
    
    @validator('launch_count')
    def validate_launch_count(cls, v):
        if v < 0:
            raise ValueError('Launch count cannot be negative')
        return v
    
    @validator('total_time_in_foreground')
    def validate_foreground_time(cls, v):
        if v < 0:
            raise ValueError('Foreground time cannot be negative')
        if v > 86400:  
            raise ValueError('Foreground time cannot exceed 24 hours')
        return v

class TrainingRequest(BaseModel):
    usage_data: List[AppUsageData]
    epochs: Optional[int] = 50
    validation_split: Optional[float] = 0.2

class AnomalyDetectionRequest(BaseModel):
    usage_data: List[AppUsageData]

class AnomalyResult(BaseModel):
    app: str
    date: str
    is_anomaly: bool
    anomaly_score: float
    confidence_percent: float
    risk_level: str  
    details: Dict[str, Any]

class TrainingResponse(BaseModel):
    success: bool
    message: str
    model_id: str
    threshold: float
    training_samples: int

class AnomalyDetectionResponse(BaseModel):
    success: bool
    results: List[AnomalyResult]
    overall_risk_level: str
    timestamp: str

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.model_id = None
        self.feature_names = ['first_hour_norm', 'last_hour_norm', 'log_launch_count', 'usage_span_norm', 'log_foreground_time']
    
    def preprocess_data(self, usage_data: List[AppUsageData]):
        """Convert API data to features"""
        data_dicts = [item.dict() for item in usage_data]
        df = pd.DataFrame(data_dicts)
        
        df['first_hour_norm'] = df['first_hour'] / 23.0
        df['last_hour_norm'] = df['last_hour'] / 23.0
        df['log_launch_count'] = np.log(df['launch_count'] + 1)
        
        df['usage_span'] = df['last_hour'] - df['first_hour']
        df['usage_span_norm'] = df['usage_span'] / 24.0
        
        df.loc[df['usage_span'] < 0, 'usage_span_norm'] = (24 + df['usage_span']) / 24.0
        
        df['log_foreground_time'] = np.log(df['total_time_in_foreground'] + 1)  
        
        features = df[self.feature_names].values
        return features, df
    
    def build_autoencoder(self, input_dim):
        """Build the autoencoder model"""
        input_layer = keras.Input(shape=(input_dim,))
        
        encoded = layers.Dense(20, activation='relu', name='encoder_1')(input_layer)
        encoded = layers.Dropout(0.1)(encoded)
        encoded = layers.Dense(10, activation='relu', name='encoder_2')(encoded)
        encoded = layers.Dense(5, activation='relu', name='bottleneck')(encoded)
        
        decoded = layers.Dense(10, activation='relu', name='decoder_1')(encoded)
        decoded = layers.Dropout(0.1)(decoded)
        decoded = layers.Dense(20, activation='relu', name='decoder_2')(decoded)
        decoded = layers.Dense(input_dim, activation='linear', name='output')(decoded)
        
        autoencoder = keras.Model(input_layer, decoded, name='behavioral_autoencoder')
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return autoencoder
    
    def train(self, usage_data: List[AppUsageData], epochs: int = 50, validation_split: float = 0.2):
        """Train the model"""
        try:
            features, df = self.preprocess_data(usage_data)
            
            if len(features) < 10:
                raise ValueError("Need at least 10 samples for training")
            
            features_scaled = self.scaler.fit_transform(features)
            
            self.model = self.build_autoencoder(features_scaled.shape[1])
            
            history = self.model.fit(
                features_scaled, features_scaled,
                epochs=epochs,
                batch_size=min(32, len(features) // 2),
                validation_split=validation_split,
                shuffle=True,
                verbose=0
            )
            
            predictions = self.model.predict(features_scaled, verbose=0)
            mse_scores = np.mean(np.square(features_scaled - predictions), axis=1)
            self.threshold = np.percentile(mse_scores, 95)
            
            self.model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info(f"Model trained successfully. Threshold: {self.threshold:.4f}")
            return True, len(features)
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return False, 0
    
    def predict(self, usage_data: List[AppUsageData]):
        """Detect anomalies"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        features, df = self.preprocess_data(usage_data)
        features_scaled = self.scaler.transform(features)
        
        predictions = self.model.predict(features_scaled, verbose=0)
        mse_scores = np.mean(np.square(features_scaled - predictions), axis=1)
        
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            score = mse_scores[i]
            is_anomaly = score > self.threshold
            confidence = min(100, (score / self.threshold) * 100) if self.threshold > 0 else 0
            
            if score > self.threshold * 2:
                risk_level = "CRITICAL"
            elif score > self.threshold * 1.5:
                risk_level = "HIGH"
            elif is_anomaly:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"
            
            results.append(AnomalyResult(
                app=row['app'],
                date=row['date'],
                is_anomaly=is_anomaly,
                anomaly_score=float(score),
                confidence_percent=float(confidence),
                risk_level=risk_level,
                details={
                    'first_hour': int(row['first_hour']),
                    'last_hour': int(row['last_hour']),
                    'launch_count': int(row['launch_count']),
                    'total_time_in_foreground': int(row['total_time_in_foreground']),
                    'threshold': float(self.threshold)
                }
            ))
        
        return results
    
    def save_model(self, path: str = "models"):
        """Save model to disk"""
        if not os.path.exists(path):
            os.makedirs(path)
        
        if self.model:
            self.model.save(f"{path}/autoencoder_{self.model_id}.h5")
            joblib.dump(self.scaler, f"{path}/scaler_{self.model_id}.pkl")
            
            metadata = {
                'model_id': self.model_id,
                'threshold': float(self.threshold),
                'feature_names': self.feature_names
            }
            with open(f"{path}/metadata_{self.model_id}.json", 'w') as f:
                json.dump(metadata, f)
model_manager = ModelManager()

@app.get("/")
async def root():
    return {
        "message": "Behavioral Anomaly Detection API",
        "status": "running",
        "model_trained": model_manager.model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_ready": model_manager.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """Train the behavioral anomaly detection model"""
    try:
        success, sample_count = model_manager.train(
            request.usage_data,
            request.epochs,
            request.validation_split
        )
        
        if success:
            model_manager.save_model()
            
            return TrainingResponse(
                success=True,
                message="Model trained successfully",
                model_id=model_manager.model_id,
                threshold=model_manager.threshold,
                training_samples=sample_count
            )
        else:
            raise HTTPException(status_code=400, detail="Training failed")
            
    except Exception as e:
        logger.error(f"Training endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """Detect anomalies in app usage patterns"""
    try:
        if model_manager.model is None:
            raise HTTPException(status_code=400, detail="Model not trained. Please train first.")
        
        results = model_manager.predict(request.usage_data)
        
        risk_levels = [r.risk_level for r in results]
        if "CRITICAL" in risk_levels:
            overall_risk = "CRITICAL"
        elif "HIGH" in risk_levels:
            overall_risk = "HIGH"
        elif "MEDIUM" in risk_levels:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        return AnomalyDetectionResponse(
            success=True,
            results=results,
            overall_risk_level=overall_risk,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Detection endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get information about the current model"""
    if model_manager.model is None:
        return {"trained": False, "message": "No model trained"}
    
    return {
        "trained": True,
        "model_id": model_manager.model_id,
        "threshold": model_manager.threshold,
        "feature_names": model_manager.feature_names
    }

@app.post("/generate-sample-data")
async def generate_sample_data():
    """Generate sample training data for testing"""
    apps = ['WhatsApp', 'Instagram', 'Chrome', 'Gmail', 'YouTube', 'Maps', 'Camera']
    sample_data = []
    
    for day in range(1, 15):
        date = f"2024-01-{day:02d}"
        
        for app in apps:
            patterns = {
                'WhatsApp': {'first': 8, 'last': 22, 'count': 40, 'avg_foreground': 3600},  
                'Instagram': {'first': 10, 'last': 23, 'count': 15, 'avg_foreground': 2400}, 
                'Chrome': {'first': 9, 'last': 18, 'count': 25, 'avg_foreground': 1800},     
                'Gmail': {'first': 7, 'last': 17, 'count': 8,  'avg_foreground': 600},      
                'YouTube': {'first': 19, 'last': 23, 'count': 12, 'avg_foreground': 4800},  
                'Maps': {'first': 8, 'last': 20, 'count': 5,   'avg_foreground': 300},      
                'Camera': {'first': 12, 'last': 18, 'count': 3, 'avg_foreground': 180}      
            }
            
            base = patterns[app]
            
            first_hour = max(0, base['first'] + np.random.randint(-2, 3))
            last_hour = min(23, base['last'] + np.random.randint(-2, 3))
            launch_count = max(1, base['count'] + np.random.randint(-5, 6))
            

            variation_factor = np.random.uniform(0.7, 1.3)
            total_time_in_foreground = int(base['avg_foreground'] * variation_factor)
            
            total_time_in_foreground = max(30, min(total_time_in_foreground, 14400))  
            
            sample_data.append({
                'app': app,
                'date': date,
                'first_hour': first_hour,
                'last_hour': last_hour,
                'launch_count': launch_count,
                'total_time_in_foreground': total_time_in_foreground
            })
    
    return {
        "message": "Sample data generated with foreground time",
        "data": sample_data,
        "count": len(sample_data)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)