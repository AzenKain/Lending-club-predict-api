from flask import Flask, request, jsonify
from flask_cors import CORS
from pydantic import BaseModel
import xgboost as xgb
import os
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000"
]}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(BASE_DIR, "model/xgb_model.json"))

MODEL_COLUMNS = [
    'annual_inc', 'pub_rec', 'pub_rec_bankruptcies', 'int_rate', 'mort_acc',
    'funded_amnt_inv', 'loan_amnt', 'dti', 'open_acc', 'installment',
    'revol_bal', 'revol_util', 'fico', 'A1', 'A2', 'A3',
    'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3', 'C4', 'C5',
    'D1', 'D2', 'D3', 'D4', 'D5', 'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2',
    'F3', 'F4', 'F5', 'G1', 'G2', 'G3', 'G4', 'G5', 'MORTGAGE', 'OTHER',
    'OWN', 'RENT', 'Individual', 'Joint App', 'car', 'credit_card',
    'debt_consolidation', 'educational', 'home_improvement', 'house',
    'major_purchase', 'medical', 'moving', 'other', 'renewable_energy',
    'small_business', 'vacation', 'wedding', '36', '60', 'Not Verified',
    'Source Verified', 'Verified'
]

class InputData(BaseModel):
    annual_inc: float
    pub_rec: int
    fico: int
    sub_grade: str
    home_ownership: str
    application_type: str
    loan_amnt: int
    mort_acc: int
    funded_amnt_inv: float
    dti: float
    open_acc: int
    pub_rec_bankruptcies: int
    purpose: str
    term: int
    revol_bal: int
    revol_util: float
    verification_status: str
    int_rate: float
    installment: float

def preprocess_input(data: InputData):
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

    input_mapping = {
        'annual_inc': data.annual_inc,
        'pub_rec': data.pub_rec,
        'pub_rec_bankruptcies': data.pub_rec_bankruptcies,
        'int_rate': data.int_rate,
        'mort_acc': data.mort_acc,
        'funded_amnt_inv': data.funded_amnt_inv,
        'loan_amnt': data.loan_amnt,
        'dti': data.dti,
        'open_acc': data.open_acc,
        'installment': data.installment,
        'revol_bal': data.revol_bal,
        'revol_util': data.revol_util,
        'fico': data.fico,
    }

    float_columns = ["int_rate", "funded_amnt_inv", "installment", "revol_util", "annual_inc"]

    for column in float_columns:
        if column in input_df.columns:
            input_df[column] = input_df[column].astype('float64')

    for key, value in input_mapping.items():
        input_df.at[0, key] = float(value) if key in float_columns else value

    if data.sub_grade in input_df.columns:
        input_df.at[0, data.sub_grade] = 1

    if data.home_ownership in input_df.columns:
        input_df.at[0, data.home_ownership] = 1

    if data.application_type in input_df.columns:
        input_df.at[0, data.application_type] = 1

    if data.purpose in input_df.columns:
        input_df.at[0, data.purpose] = 1

    if str(data.term) in input_df.columns:
        input_df.at[0, str(data.term)] = 1

    if data.verification_status in input_df.columns:
        input_df.at[0, data.verification_status] = 1

    return input_df

@app.route("/predict", methods=["POST"])
def predict():
    try:
        request_data = request.get_json()
        data = InputData(**request_data)
        input_data = preprocess_input(data)
        xgb_prediction = xgb_model.predict(xgb.DMatrix(input_data))
        return jsonify({
            "xgb_prediction": xgb_prediction.tolist()[0]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/")
def main():
    return jsonify({"message": "Hello World"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3434)