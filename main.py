import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import nest_asyncio

nest_asyncio.apply()

# تحميل النموذج
# بدلاً من استخدام المسار المطلق، استخدم مسار نسبي
model = joblib.load(r"rf_model.pkl")


app = FastAPI()

# تعريف نموذج البيانات
class DiabetesData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree: float
    age: int

@app.post("/predict")
def predict(data: DiabetesData):
    insulin_log = np.log(data.insulin + 1)  
    diabetes_pedigree_log = np.log(data.diabetes_pedigree + 1)
    
    features = np.array([[data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
                          insulin_log, data.bmi, diabetes_pedigree_log, data.age]])
    
    prediction = model.predict(features)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    
    return {"prediction": result}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8030)
