from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.llm_explainer import explain_prediction
from app.ml.features import DEFAULT_VALUES, FEATURE_COLUMNS, FORM_FIELDS
from app.ml.model_service import ModelNotTrainedError, predict_risk


app = FastAPI(title="Классификация девиантного риска")
templates = Jinja2Templates(directory="app/templates")


def form_to_payload(form_data) -> dict:
    return {feature: form_data.get(feature) for feature in FEATURE_COLUMNS}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "fields": FORM_FIELDS,
            "values": DEFAULT_VALUES,
            "result": None,
            "explanation": None,
            "error": None,
        },
    )


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    values = form_to_payload(form_data)
    try:
        result = predict_risk(values)
        error = None
        try:
            explanation = explain_prediction(values, result)
        except Exception as exc:
            explanation = f"LLM-пояснение не сформировано: {exc}"
    except (ValueError, ModelNotTrainedError) as exc:
        result = None
        explanation = None
        error = str(exc)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "fields": FORM_FIELDS,
            "values": values,
            "result": result,
            "explanation": explanation,
            "error": error,
        },
    )


@app.post("/api/predict")
async def api_predict(payload: dict):
    result = predict_risk(payload)
    try:
        result["explanation"] = explain_prediction(payload, result)
    except Exception as exc:
        result["explanation"] = f"LLM-пояснение не сформировано: {exc}"
    return result
