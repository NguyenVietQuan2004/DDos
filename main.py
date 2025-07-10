
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import numpy as np
from utils import preprocess_input_data
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import BinaryCrossentropy
import io
import traceback

app = FastAPI()

# Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class CustomBinaryCrossentropy(BinaryCrossentropy):
    def __init__(self, fn=None, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
# Load mô hình
model = load_model("best_model_30_1.h5", custom_objects={
    'BinaryCrossentropy': CustomBinaryCrossentropy
})

# Load scaler
scaler = joblib.load("scaler.pkl")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Đọc file CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        # Tiền xử lý input
        X = preprocess_input_data(df, scaler, n_past=30)

        # Dự đoán
        predictions = model.predict(X)
        results = (predictions > 0.5).astype(int).flatten().tolist()
        print(predictions)
        return JSONResponse(content={"results": results})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})