import streamlit as st
import json
from pathlib import Path
import pandas as pd

THESIS_CROP_MAP = {
    "Cabbage": "Cabbage",
    "Potato": "Potato",
    "Carrot": "Carrot",
    "Lettuce": "Lettuce",
    "Broccoli": "Broccoli",
    "Snap Bean": "String Beans",
    "Tomato": "tomatoes"
}

RULE_ARRAY_ORDER = [
    "preprocessing_rules",
    "ph_classification_rules",
    "crop_target_rules",
    "lime_requirement_rules",
    "nutrient_availability_rules",
    "integration_modifiers_for_npk_engine",
]

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    if (current.parent / "data").exists():
        return current.parent
    if (current.parent.parent / "data").exists():
        return current.parent.parent
    return current.parent

def load_assets():
    base_dir = get_project_root()
    data_dir = base_dir / "data"

    with open(data_dir / "fertilizers.json", "r", encoding="utf-8") as f:
        inventory = json.load(f)["inventory"]
    with open(data_dir / "engine_rules.json", "r", encoding="utf-8") as r:
        rules = json.load(r)["engine_logic"]
    with open(data_dir / "crop_npk_rules.json", "r", encoding="utf-8") as c:
        crop_rules = json.load(c)
    with open(data_dir / "ph_rules.json", "r", encoding="utf-8") as p:
        ph_rules = json.load(p)

    return inventory, rules, crop_rules, ph_rules

def parse_target_value(val):
    if val is None: return 0.0
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str) and ("–" in val or "-" in val):
        val = val.replace("–", "-")
        parts = [float(p.strip()) for p in val.split("-")]
        return sum(parts) / len(parts)
    try:
        return float(val)
    except:
        return 0.0

def get_fertilizer_recommendation(crop_name, n_status, p_status, k_status, crop_rules):
    crop_data = crop_rules.get(crop_name)
    if not crop_data: return None
    t_n = parse_target_value(crop_data["N"].get(n_status, 0))
    t_p = parse_target_value(crop_data["P"].get(p_status, 0))
    t_k = parse_target_value(crop_data["K"].get(k_status, 0))
    return t_n, t_p, t_k

def run_ph_engine(data, ph_rules):
    filtered_data = {
        "crop": data["crop"], "soil_ph": data["soil_ph"], "buffer_ph": None,
        "lime_requirement_value": None, "potato_scab_sensitive": data.get("potato_scab_sensitive", None),
        "legume_rotation_present": None,
    }
    context = dict(filtered_data)
    context.update({"warnings": [], "actions": [], "decision_trace": []})
    
    # Simple mock-up of rule execution for brevity in this UI code
    # In your real app, this calls run_rule_array which modifies context
    context["ph_status"] = "acidic" if data["soil_ph"] < 6.0 else "optimal"
    context["target_ph"] = 6.5
    return context

def get_ph_modifiers(ph_result):
    soil_ph = ph_result.get("soil_ph", 0)
    if soil_ph < 5.5: return {"phosphorus_effective_availability_multiplier": 0.75, "nitrogen_use_efficiency_multiplier": 0.85}
    if 5.5 <= soil_ph < 6.0: return {"phosphorus_effective_availability_multiplier": 0.9, "nitrogen_use_efficiency_multiplier": 0.95}
    return {"phosphorus_effective_availability_multiplier": 1.0, "nitrogen_use_efficiency_multiplier": 1.0}

def adjust_targets_with_ph(t_n, t_p, t_k, ph_result):
    modifiers = get_ph_modifiers(ph_result)
    p_m = modifiers.get("phosphorus_effective_availability_multiplier", 1.0)
    n_m = modifiers.get("nitrogen_use_efficiency_multiplier", 1.0)
    adj_n = round(float(t_n) / n_m, 2) if n_m > 0 else t_n
    adj_p = round(float(t_p) / p_m, 2) if p_m > 0 else t_p
    ph_result["integration_modifiers"] = modifiers
    return adj_n, adj_p, float(t_k)

def solve_npk(t_n, t_p, t_k, inventory, rules):
    results = []
    prec = rules["constraints"]["precision_decimals"]
    n_filler = next(f for f in inventory if f["n"] > 0 and f["p"] == 0 and f["k"] == 0)
    k_filler = next(f for f in inventory if f["k"] > 0 and f["n"] == 0 and f["p"] == 0)
    for p_fert in [f for f in inventory if f["p"] > 0]:
        qty_p = (t_p / p_fert["p"]) * 100 if p_fert["p"] > 0 else 0
        n_prov = (qty_p * p_fert["n"]) / 100
        k_prov = (qty_p * p_fert["k"]) / 100
        qty_n = (max(0, t_n - n_prov) / n_filler["n"]) * 100
        qty_k = (max(0, t_k - k_prov) / k_filler["k"]) * 100
        prescription = []
        if qty_n > 0.01: prescription.append(f"{round(qty_n, prec)} kg of {n_filler['name']}")
        if qty_p > 0.01: prescription.append(f"{round(qty_p, prec)} kg of {p_fert['name']}")
        if qty_k > 0.01: prescription.append(f"{round(qty_k, prec)} kg of {k_filler['name']}")
        results.append({"Source": p_fert["name"], "Prescription": prescription, "Total Weight": qty_n + qty_p + qty_k})
    return sorted(results, key=lambda x: x["Total Weight"])[:rules["constraints"]["max_combinations"]]

# --- MAIN UI ---
st.set_page_config(page_title="Rule-Based NPK + pH", layout="centered")
st.title("🌱 Fertilizer Recommendation Engine")

try:
    inventory, rules, crop_rules, ph_rules = load_assets()

    with st.sidebar:
        st.header("1. Land Information")
        # PHILIPPINES LOCALIZATION: Unit selection for sqm or ha
        unit = st.radio("Select Area Unit", ["Square Meters (sqm)", "Hectares (ha)"])
        raw_area = st.number_input(f"Total Area ({unit})", min_value=1.0, value=500.0 if "sqm" in unit else 1.0)
        
        # INTERNAL CONVERSION: Normalize to Hectares for the logic engine
        area_ha = raw_area / 10000 if "sqm" in unit else raw_area
        
        st.write("---")
        st.header("2. Soil & Crop Data")
        selected_crop_label = st.selectbox("Select Crop", options=list(THESIS_CROP_MAP.keys()))
        selected_crop = THESIS_CROP_MAP[selected_crop_label]
        n_lvl = st.selectbox("Nitrogen (N) Status", options=["L", "M", "H", "VH"])
        p_lvl = st.selectbox("Phosphorus (P) Status", options=["L", "ML", "MH", "H", "VH"])
        k_lvl = st.selectbox("Potassium (K) Status", options=["L", "S", "S+", "S++/+++"])
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=5.5, step=0.1)

    # PORTION: Calculation Logic
    rec = get_fertilizer_recommendation(selected_crop, n_lvl, p_lvl, k_lvl, crop_rules)
    
    if rec:
        base_n, base_p, base_k = rec
        ph_res = run_ph_engine({"crop": selected_crop_label.lower(), "soil_ph": soil_ph}, ph_rules)
        adj_n_ha, adj_p_ha, adj_k_ha = adjust_targets_with_ph(base_n, base_p, base_k, ph_res)

        if st.button("Calculate Prescription", type="primary"):
            # 1. SCALING: Multiply rates by actual land size (area_ha)
            t_base_n, t_base_p, t_base_k = base_n * area_ha, base_p * area_ha, base_k * area_ha
            t_adj_n, t_adj_p, t_adj_k = adj_n_ha * area_ha, adj_p_ha * area_ha, adj_k_ha * area_ha

            st.success(f"Results for {raw_area} {unit}")
            
            # 2. SOIL pH ASSESSMENT SECTION
            st.subheader("Soil pH Assessment")
            left, right = st.columns(2)

            # Extracting values from ph_res (calculated before the button press)
            ph_status = ph_res.get("ph_status", "N/A").replace("_", " ").title()
            target_ph = ph_res.get("target_ph", "N/A")
            lime_trigger_ph = ph_res.get("lime_trigger_ph", 6.0) # Default if missing
            within_range = ph_res.get("within_target_range", soil_ph >= 6.0)
            liming_needed = ph_res.get("liming_recommended", soil_ph < 5.5)
            lime_test_needed = ph_res.get("lime_requirement_test_needed", soil_ph < 5.0)

            with left:
                status_key = ph_res.get("ph_status", "")
                if "acidic" in status_key:
                    st.error(f"Soil Status: {ph_status}")
                elif "alkaline" in status_key:
                    st.warning(f"Soil Status: {ph_status}")
                else:
                    st.success(f"Soil Status: {ph_status}")
                st.write(f"**Target pH:** {target_ph}")
                st.write(f"**Lime Trigger pH:** {lime_trigger_ph}")

            with right:
                st.metric("Within Target Range", "Yes" if within_range else "No")
                st.metric("Liming Recommended", "Yes" if liming_needed else "No")
                st.metric("Lime Requirement Test Needed", "Yes" if lime_test_needed else "No")

            # 3. WARNINGS AND IMPACTS
            if ph_res.get("warnings"):
                for warning in ph_res["warnings"]:
                    st.warning(warning)

            effects = ph_res.get("nutrient_availability_effects", [])
            if effects:
                st.markdown("### Nutrient Impact")
                for effect in effects:
                    parameter = effect["parameter"].replace("_", " ").title()
                    impact = effect["effect"].replace("_", " ").title()
                    st.info(f"{parameter}: {impact}")

            st.divider() # Visual separator before NPK results

            # 4. NPK SUMMARY TABLE
            summary_df = pd.DataFrame([
                {"Scenario": "Standard Need (kg)", "N": t_base_n, "P": t_base_p, "K": t_base_k},
                {"Scenario": "pH-Adjusted Need (kg)", "N": t_adj_n, "P": t_adj_p, "K": t_adj_k}
            ]).set_index("Scenario")
            st.table(summary_df.style.format("{:.2f}"))

            # 5. MIX COMPARISON
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Standard Mix")
                base_results = solve_npk(t_base_n, t_base_p, t_base_k, inventory, rules)
                for res in base_results:
                    with st.expander(f"Using {res['Source']}"):
                        for line in res["Prescription"]: st.info(line)
                        st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

            with col2:
                st.markdown("### pH-Adjusted Mix")
                adj_results = solve_npk(t_adj_n, t_adj_p, t_adj_k, inventory, rules)
                for res in adj_results:
                    with st.expander(f"Using {res['Source']}"):
                        for line in res["Prescription"]: st.success(line)
                        st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

except Exception as e:
    st.error(f"Configuration Error: {e}")