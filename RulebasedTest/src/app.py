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
    # extracts contraint rukes from engine_rules.jason
    precision = rules["constraints"]["precision_decimals"]  # specifaclly extract preison decimla point (2)
    allow_over = rules["constraints"]["allow_over_fertilization"] # Extracts allow_over_fertilization - False,over fert is not allowed

    # exrtacting fertilizer with Pure N, K and P Compound & Complete
    n_filler = next(f for f in inventory if f["n"] > 0 and f["p"] == 0 and f["k"] == 0)
    k_filler = next(f for f in inventory if f["k"] > 0 and f["n"] == 0 and f["p"] == 0)
    p_sources = [f for f in inventory if f["p"] > 0] # Compund & Complete

    #Solving Part (Core)
    for p_fert in p_sources:
        qty_p = (t_p / p_fert["p"]) * 100 if p_fert["p"] > 0 else 0  #Solving for Comound || Complete using p_sources

        n_provided = (qty_p * p_fert["n"]) / 100
        p_provided = (qty_p * p_fert["p"]) / 100
        k_provided = (qty_p * p_fert["k"]) / 100

        # Subtracting needed N & K
        rem_n = t_n - n_provided 
        rem_k = t_k - k_provided
        
        #To check whether is over fertilization or not
        if not allow_over and (rem_n < -0.01 or rem_k < -0.01):
            continue

        #if no over fertilizer solve the remaining N &K
        qty_n = (max(0, rem_n) / n_filler["n"]) * 100
        qty_k = (max(0, rem_k) / k_filler["k"]) * 100

        total_n = n_provided + ((qty_n * n_filler["n"]) / 100)
        total_k = k_provided + ((qty_k * k_filler["k"]) / 100)

        fmt = rules["output_format"]
        prescription = []
        if qty_n > 0:
            prescription.append(fmt.format(qty=round(qty_n, precision), fertilizer_name=n_filler["name"]))
        prescription.append(fmt.format(qty=round(qty_p, precision), fertilizer_name=p_fert["name"]))
        if qty_k > 0:
            prescription.append(fmt.format(qty=round(qty_k, precision), fertilizer_name=k_filler["name"]))

        results.append({
            "Source": p_fert["name"],
            "Prescription": prescription,
            "Total Weight": qty_n + qty_p + qty_k,
            "Applied N": total_n,
            "Applied P": p_provided,
            "Applied K": total_k,
        })

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
            # base_n/p/k are the raw kg/ha from the JSON
            t_base_n, t_base_p, t_base_k = base_n * area_ha, base_p * area_ha, base_k * area_ha
            t_adj_n, t_adj_p, t_adj_k = adj_n_ha * area_ha, adj_p_ha * area_ha, adj_k_ha * area_ha

            st.success(f"✅ Results for {raw_area} {unit}")

            # --- NEW SECTION: SCIENTIFIC BASIS ---
            with st.expander("📊 View Nutrient Basis (Reference Rates)", expanded=False):
                st.markdown(f"**Crop:** {selected_crop_label} | **Soil Status:** N:{n_lvl}, P:{p_lvl}, K:{k_lvl}")
                st.write("These values represent the standard recommendation per hectare before land-size scaling or pH adjustment:")
                
                ref_col1, ref_col2, ref_col3 = st.columns(3)
                ref_col1.metric("Target N", f"{base_n} kg/ha")
                ref_col2.metric("Target P₂O₅", f"{base_p} kg/ha")
                ref_col3.metric("Target K₂O", f"{base_k} kg/ha")
                st.caption("Source: crop_npk_rules.json")

            # --- SECTION 1: SOIL pH ASSESSMENT ---
            st.subheader("1. Soil Condition Assessment")
            ph_col1, ph_col2 = st.columns(2)
            ph_status = ph_res.get("ph_status", "N/A").replace("_", " ").title()
            
            with ph_col1:
                if "acidic" in ph_status.lower():
                    st.error(f"Soil Status: {ph_status}")
                else:
                    st.success(f"Soil Status: {ph_status}")
                st.write(f"**Optimal pH Range:** 6.0 – 7.0")

            with ph_col2:
                # Calculating efficiency based on your get_ph_modifiers function
                mods = get_ph_modifiers(ph_res)
                n_eff = mods['nitrogen_use_efficiency_multiplier'] * 100
                p_eff = mods['phosphorus_effective_availability_multiplier'] * 100
                st.write(f"**N Efficiency:** {n_eff}%")
                st.write(f"**P Efficiency:** {p_eff}%")

            st.divider()

            # --- SECTION 2: NUTRIENT REQUIREMENTS (TOTAL) ---
            st.subheader("2. Calculated Requirements for your Land")
            
            summary_df = pd.DataFrame([
                {
                    "Analysis Step": "Theoretical Requirement (Standard)", 
                    "N (kg)": t_base_n, "P (kg)": t_base_p, "K (kg)": t_base_k
                },
                {
                    "Analysis Step": "Field-Adjusted (pH Corrected)", 
                    "N (kg)": t_adj_n, "P (kg)": t_adj_p, "K (kg)": t_adj_k
                }
            ]).set_index("Analysis Step")
            
            st.table(summary_df.style.format("{:.2f}"))
            
            # --- SECTION 3: FERTILIZER MIXES ---
            st.subheader("3. Fertilizer Application Options")
            mix_col1, mix_col2 = st.columns(2)
            
            with mix_col1:
                st.markdown("#### Standard Mix")
                base_results = solve_npk(t_base_n, t_base_p, t_base_k, inventory, rules)
                for res in base_results:
                    with st.expander(f"Using {res['Source']}"):
                        for line in res["Prescription"]: st.info(line)
                        st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

            with mix_col2:
                st.markdown("#### pH-Adjusted Mix (Recommended)")
                adj_results = solve_npk(t_adj_n, t_adj_p, t_adj_k, inventory, rules)
                for res in adj_results:
                    with st.expander(f"Using {res['Source']}"):
                        for line in res["Prescription"]: st.success(line)
                        st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

except Exception as e:
    st.error(f"Configuration Error: {e}")