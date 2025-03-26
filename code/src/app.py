# main.py
from fastapi import FastAPI
from reconciliation_analysis import analyze_reconciliation_data  # Import from your module
from reconciliation_analysis import ReconciliationData

app = FastAPI()

@app.post("/analyze/")
async def analyze(reconciliation_data: ReconciliationData):
    return await analyze_reconciliation_data(reconciliation_data)