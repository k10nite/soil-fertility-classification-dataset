from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app_final import main as build_recommendation

app = FastAPI(title="Rule-Based Fertilizer Recommendation API")


class RecommendationRequest(BaseModel):
    crop_label: str
    n_status: str
    p_status: str
    k_status: str
    soil_ph: float
    raw_area: float
    area_unit: str = "Square Meters (sqm)"
    selected_inventory_names: Optional[List[str]] = None


# Example request body for POST /recommendation:
# {
#   "crop_label": "Cabbage",
#   "n_status": "L",
#   "p_status": "ML",
#   "k_status": "S",
#   "soil_ph": 5.5,
#   "raw_area": 500.0,
#   "area_unit": "Square Meters (sqm)",
#   "selected_inventory_names": ["Urea", "14-14-14"]
# }
# Accepted area_unit values include:
# - "Square Meters (sqm)"
# - "sqm"
# - "Hectares (ha)"
# - "ha"
@app.post("/recommendation")
def recommendation(request: RecommendationRequest):
    return build_recommendation(
        crop_label=request.crop_label,
        n_status=request.n_status,
        p_status=request.p_status,
        k_status=request.k_status,
        soil_ph=request.soil_ph,
        raw_area=request.raw_area,
        area_unit=request.area_unit,
        selected_inventory_names=request.selected_inventory_names,
    )
