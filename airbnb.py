import json
import logging
from fastapi import FastAPI
from inference import Inference
from input_data import InputData

app = FastAPI()

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)

inf = Inference(model_save='model_default', aux_model_save='aux_model_default.ckpt')

@app.get("/api")
def default():
    return "This is an Airbnb price prediction API"

@app.get("/api/test")
def default_test(): 
    default_features = InputData().dict()
    description = default_features.pop('description', None)
    pred = inf.get_model_pred(default_features, description)
    logger.info(f'default_test predicted price_year_avg: {pred}')
    return {'default_test_pred_price_year_avg': str(pred)}

@app.post("/api/predict")
def predict_price(input_data: InputData):
    input_features = input_data.dict()
    description = input_features.pop('description', None)
    pred = inf.get_model_pred(input_features, description)
    logger.info(f'predicted price_year_avg: {pred}')
    return {'pred_price_year_avg': str(pred)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=9090)

