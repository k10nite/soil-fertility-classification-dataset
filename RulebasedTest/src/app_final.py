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
    """Resolve the project root directory for the rule-based engine.

    This function searches upward from the current file until it finds a
    directory containing the expected "data" folder. It ensures the rest of
    the app can reliably load JSON assets from the rule-based data directory.

    Returns:
        Path: The project root directory containing the "data" folder.
    """
    current = Path(__file__).resolve()
    if (current.parent / "data").exists():
        return current.parent
    if (current.parent.parent / "data").exists():
        return current.parent.parent
    return current.parent

def load_assets():
    """Load rule engine JSON assets from the project data folder.

    Reads the fertilizer inventory, engine rules, crop NPK target rules, and
    pH adjustment rules from JSON files under the configured data directory.

    Returns:
        tuple: A tuple containing (inventory, rules, crop_rules, ph_rules).
    """
    base_dir = get_project_root()
    data_dir = base_dir / "data"

    with open(data_dir / "fertilizers.json", "r", encoding="utf-8") as f:
        inventory = json.load(f)["inventory"]
    with open(data_dir / "engine_rules.json", "r", encoding="utf-8") as r:
        rules = json.load(r)["engine_logic"]
    with open(data_dir / "crop_npk_rules.json", "r", encoding="utf-8") as c:
        crop_rules = json.load(c)
    with open(data_dir / "crop_workaround.json", "r", encoding="utf-8") as c:
        crop_rules_workaround = json.load(c)
    with open(data_dir / "ph_rules.json", "r", encoding="utf-8") as p:
        ph_rules = json.load(p)

    return inventory, rules, crop_rules, crop_rules_workaround, ph_rules

def parse_target_value(val):
    """Normalize a crop nutrient target value into a float.

    This helper accepts numeric values and range strings such as "10-12" or
    "6–8". If a range string is provided, it returns the midpoint.

    Args:
        val: The raw value from crop target rules, which may be int, float, or str.

    Returns:
        float: The parsed numeric target, or 0.0 on failure.
    """
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
    """Compute the base N-P-K target rates for a specific crop and soil status.

    Args:
        crop_name: The normalized crop key used by crop rules.
        n_status: Nitrogen soil status code (e.g. "L", "M", "H", "VH").
        p_status: Phosphorus soil status code (e.g. "L", "ML", "MH", "H", "VH").
        k_status: Potassium soil status code (e.g. "L", "S", "S+", "S++/+++").
        crop_rules: Mapping of crop names to NPK target rule definitions.

    Returns:
        tuple|None: Target rates (N, P, K) in kg/ha or None if crop data is missing.
    """
    crop_data = crop_rules.get(crop_name)
    if not crop_data: return None
    t_n = parse_target_value(crop_data["N"].get(n_status, 0))
    t_p = parse_target_value(crop_data["P"].get(p_status, 0))
    t_k = parse_target_value(crop_data["K"].get(k_status, 0))
    return t_n, t_p, t_k

def run_ph_engine(data, ph_rules):
    """Evaluate the soil pH engine and return pH-related adjustment context.

    This function currently uses a simplified pH rule logic for the UI.
    It prepares the rule input context and returns pH status, target pH, and any
    warning/decision trace placeholders.

    Args:
        data: A dictionary containing crop and soil_ph values.
        ph_rules: Loaded pH rule definitions from ph_rules.json.

    Returns:
        dict: pH engine output including ph_status and target_ph.
    """
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
    """Select pH adjustment multipliers based on soil pH value.

    These modifiers adjust the effective availability of phosphorus and the
    nitrogen use efficiency when the soil is acidic.

    Args:
        ph_result: A pH engine result dictionary containing soil_ph.

    Returns:
        dict: Multipliers for phosphorus availability and nitrogen efficiency.
    """
    soil_ph = ph_result.get("soil_ph", 0)
    if soil_ph < 5.5: return {"phosphorus_effective_availability_multiplier": 0.75, "nitrogen_use_efficiency_multiplier": 0.85}
    if 5.5 <= soil_ph < 6.0: return {"phosphorus_effective_availability_multiplier": 0.9, "nitrogen_use_efficiency_multiplier": 0.95}
    return {"phosphorus_effective_availability_multiplier": 1.0, "nitrogen_use_efficiency_multiplier": 1.0}

def adjust_targets_with_ph(t_n, t_p, t_k, ph_result):
    """Apply pH-based multipliers to nutrient targets.

    The method adjusts the raw N and P targets using pH modifiers, while K is
    passed through unchanged. Adjusted targets are used for field-level
    prescription calculations.

    Args:
        t_n: Base nitrogen target in kg/ha.
        t_p: Base phosphorus target in kg/ha.
        t_k: Base potassium target in kg/ha.
        ph_result: Result from the pH engine, used to derive modifiers.

    Returns:
        tuple: Adjusted target rates (N, P, K) in kg/ha.
    """
    modifiers = get_ph_modifiers(ph_result)
    p_m = modifiers.get("phosphorus_effective_availability_multiplier", 1.0)
    n_m = modifiers.get("nitrogen_use_efficiency_multiplier", 1.0)
    adj_n = round(float(t_n) / n_m, 2) if n_m > 0 else t_n
    adj_p = round(float(t_p) / p_m, 2) if p_m > 0 else t_p
    ph_result["integration_modifiers"] = modifiers
    return adj_n, adj_p, float(t_k)


def solve_npk(t_n, t_p, t_k, inventory, rules, area, unit_label):
    """Generate fertilizer mix options based on target nutrient requirements.

    This algorithm uses a pure-n fertilizer, a pure-k fertilizer, and one or more
    compound P-bearing fertilizers to generate candidate prescriptions.

    Args:
        t_n: Total nitrogen requirement for the field.
        t_p: Total phosphorus requirement for the field.
        t_k: Total potassium requirement for the field.
        inventory: Loaded fertilizer inventory list.
        rules: Engine rule definitions including constraints and output formatting.
        area: The raw area value entered by the user.
        unit_label: The unit string used for display in prescriptions.

    Returns:
        list: Sorted fertilizer combination results limited by rule constraints.
    """
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
            prescription.append(fmt.format(
                qty=round(qty_n, precision), 
                area=area, 
                unit=unit_label, 
                fertilizer_name=n_filler["name"]
            ))
        prescription.append(fmt.format(
            qty=round(qty_p, precision), 
            area=area, 
            unit=unit_label, 
            fertilizer_name=p_fert["name"]
        ))
        if qty_k > 0:
            prescription.append(fmt.format(
                qty=round(qty_k, precision), 
                area=area, 
                unit=unit_label, 
                fertilizer_name=k_filler["name"]
            ))

        results.append({
            "Source": p_fert["name"],
            "Prescription": prescription,
            "Total Weight": qty_n + qty_p + qty_k,
            "Applied N": total_n,
            "Applied P": p_provided,
            "Applied K": total_k,
        })

    return sorted(results, key=lambda x: x["Total Weight"])[:rules["constraints"]["max_combinations"]]


def normalize_area(raw_area, area_unit):
    """Convert user area input into hectares and derive display unit label.

    Args:
        raw_area: The numeric area value entered by the user.
        area_unit: The selected area unit string, either "sqm"/"Square Meters (sqm)" or
            "ha"/"Hectares (ha)".

    Returns:
        tuple: The converted area in hectares and the normalized unit label.

    Raises:
        ValueError: If the area_unit is not recognized.
    """
    normalized = area_unit.strip().lower()
    if "sqm" in normalized or "square meter" in normalized:
        return raw_area / 10000.0, "sqm"
    if "ha" in normalized or "hectare" in normalized:
        return raw_area, "ha"
    raise ValueError(
        f"Unsupported area_unit '{area_unit}'. Use 'sqm' or 'ha', or values like 'Square Meters (sqm)' or 'Hectares (ha)'."
    )


def ph_adjusted_recommendation(crop_label, n_status, p_status, k_status, soil_ph, raw_area,
                               area_unit="Square Meters (sqm)"):
    """Calculate pH-adjusted nutrient targets and return recommended fertilizer list.

    This method handles the entire pH adjustment workflow: loading assets, computing
    base nutrient targets, running the pH engine, applying pH-based multipliers,
    and generating fertilizer mix recommendations.

    Args:
        crop_label: User-facing crop label (e.g. "Cabbage").
        n_status: Nitrogen status code.
        p_status: Phosphorus status code.
        k_status: Potassium status code.
        soil_ph: Measured soil pH value.
        raw_area: Numeric area from the user input.
        area_unit: The area unit string, defaulting to square meters.

    Returns:
        list: Recommended fertilizer mixes sorted by total weight. Each item contains
              Source, Prescription, Total Weight, Applied N/P/K.

    Raises:
        ValueError: If crop is not configured or unable to compute recommendation.
    """
    inventory, rules, crop_rules, ph_rules = load_assets()
    selected_crop = THESIS_CROP_MAP.get(crop_label, crop_label)

    if selected_crop not in crop_rules:
        raise ValueError(f"Crop '{selected_crop}' is not configured in crop_npk_rules.json")

    area_ha, unit_label = normalize_area(raw_area, area_unit)
    
    rec = get_fertilizer_recommendation(selected_crop, n_status, p_status, k_status, crop_rules)
    if rec is None:
        raise ValueError("Unable to compute fertilizer recommendation for the selected crop and soil status.")

    base_n, base_p, base_k = rec
    ph_res = run_ph_engine({"crop": selected_crop.lower(), "soil_ph": soil_ph}, ph_rules)
    adj_n_ha, adj_p_ha, adj_k_ha = adjust_targets_with_ph(base_n, base_p, base_k, ph_res)

    t_adj_n, t_adj_p, t_adj_k = adj_n_ha * area_ha, adj_p_ha * area_ha, adj_k_ha * area_ha

    adjusted_mix = solve_npk(t_adj_n, t_adj_p, t_adj_k, inventory, rules, raw_area, unit_label)
    
    return {
        "adjusted_targets_per_ha": {"N": adj_n_ha, "P": adj_p_ha, "K": adj_k_ha},
        "total_adjusted": {"N": t_adj_n, "P": t_adj_p, "K": t_adj_k},
        "ph_result": ph_res,
        "adjusted_mix": adjusted_mix,
    }


def build_recommendation(crop_label, n_status, p_status, k_status, soil_ph, raw_area,
                         area_unit="Square Meters (sqm)", selected_inventory_names=None):
    """Build a full fertilizer recommendation payload for external use.

    This method loads the engine assets, resolves crop targets, applies pH
    adjustments, scales values by land area, and computes both standard and
    adjusted fertilizer mix recommendations.

    Args:
        crop_label: User-facing crop label (e.g. "Cabbage").
        n_status: Nitrogen status code.
        p_status: Phosphorus status code.
        k_status: Potassium status code.
        soil_ph: Measured soil pH value.
        raw_area: Numeric area from the user input.
        area_unit: The area unit string, defaulting to square meters.
        selected_inventory_names: Optional list of fertilizer names the user has.

    Returns:
        dict: Recommendation results including targets, mixes, pH output, and sufficiency state.
    """
    inventory, rules, crop_rules, ph_rules = load_assets()
    selected_crop = THESIS_CROP_MAP.get(crop_label, crop_label)

    if selected_crop not in crop_rules:
        raise ValueError(f"Crop '{selected_crop}' is not configured in crop_npk_rules.json")

    area_ha, unit_label = normalize_area(raw_area, area_unit)
    
    rec = get_fertilizer_recommendation(selected_crop, n_status, p_status, k_status, crop_rules)
    if rec is None:
        raise ValueError("Unable to compute fertilizer recommendation for the selected crop and soil status.")

    base_n, base_p, base_k = rec
    ph_res = run_ph_engine({"crop": selected_crop.lower(), "soil_ph": soil_ph}, ph_rules)

    t_base_n, t_base_p, t_base_k = base_n * area_ha, base_p * area_ha, base_k * area_ha

    selected_inventory_names = selected_inventory_names or []
    user_inventory = [f for f in inventory if f["name"] in selected_inventory_names]

    has_n = any(f["n"] > 0 for f in user_inventory)
    has_p = any(f["p"] > 0 for f in user_inventory)
    has_k = any(f["k"] > 0 for f in user_inventory)
    missing_nutrients = []
    if t_base_n > 0 and not has_n: missing_nutrients.append("Nitrogen (N)")
    if t_base_p > 0 and not has_p: missing_nutrients.append("Phosphorus (P)")
    if t_base_k > 0 and not has_k: missing_nutrients.append("Potassium (K)")

    base_mix = solve_npk(t_base_n, t_base_p, t_base_k, inventory, rules, raw_area, unit_label)

    return {
        "inventory": inventory,
        "rules": rules,
        "crop_rules": crop_rules,
        "ph_rules": ph_rules,
        "selected_crop_label": crop_label,
        "selected_crop": selected_crop,
        "area_ha": area_ha,
        "unit_label": unit_label,
        "raw_area": raw_area,
        "base_targets_per_ha": {"N": base_n, "P": base_p, "K": base_k},
        "total_base": {"N": t_base_n, "P": t_base_p, "K": t_base_k},
        "ph_result": ph_res,
        "user_inventory": user_inventory,
        "inventory_sufficiency": {
            "has_n": has_n,
            "has_p": has_p,
            "has_k": has_k,
            "missing_nutrients": missing_nutrients,
        },
        "standard_mix": base_mix,
    }


def run_ui():
    st.set_page_config(page_title="Rule-Based NPK + pH", layout="centered")
    st.title("🌱 Fertilizer Recommendation Engine")

    try:
        inventory, rules, crop_rules, ph_rules = load_assets()

        with st.sidebar:
            st.header("1. Land Information")
            unit = st.radio("Select Area Unit", ["Square Meters (sqm)", "Hectares (ha)"])
            raw_area = st.number_input(f"Total Area ({unit})", min_value=1.0, value=500.0 if "sqm" in unit else 1.0)
            
            # INTERNAL CONVERSION
            area_ha = raw_area / 10000 if "sqm" in unit else raw_area
            
            st.write("---")
            st.header("2. Soil & Crop Data")
            selected_crop_label = st.selectbox("Select Crop", options=list(THESIS_CROP_MAP.keys()))
            selected_crop = THESIS_CROP_MAP[selected_crop_label]
            n_lvl = st.selectbox("Nitrogen (N) Status", options=["L", "M", "H", "VH"])
            p_lvl = st.selectbox("Phosphorus (P) Status", options=["L", "ML", "MH", "H", "VH"])
            k_lvl = st.selectbox("Potassium (K) Status", options=["L", "S", "S+", "S++/+++"])
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=5.5, step=0.1)

            st.write("---")
            st.header("3. Inventory Management")
            with st.expander("🛒 Plan Your Purchase / Select Inventory", expanded=True):
                fert_names = [f["name"] for f in inventory]
                # Changed 'default=fert_names' to 'default=[]'
                user_selection = st.multiselect(
                    "Select fertilizers you plan to buy or use:",
                    options=fert_names,
                    default=[], 
                    help="Start typing to search for fertilizers like Urea, 14-14-14, etc."
                )

        # PORTION: Calculation Logic
        rec = get_fertilizer_recommendation(selected_crop, n_lvl, p_lvl, k_lvl, crop_rules)
        
        if rec:
            base_n, base_p, base_k = rec
            ph_res = run_ph_engine({"crop": selected_crop_label.lower(), "soil_ph": soil_ph}, ph_rules)
            adj_n_ha, adj_p_ha, adj_k_ha = adjust_targets_with_ph(base_n, base_p, base_k, ph_res)

            if st.button("Calculate Prescription", type="primary"):
                # INITIALIZE USER INVENTORY FROM SELECTION
                user_inventory = [f for f in inventory if f["name"] in user_selection]

                # 1. SCALING
                t_base_n, t_base_p, t_base_k = base_n * area_ha, base_p * area_ha, base_k * area_ha
                t_adj_n, t_adj_p, t_adj_k = adj_n_ha * area_ha, adj_p_ha * area_ha, adj_k_ha * area_ha

                st.success(f"✅ Results for {raw_area} {unit}")

                # --- SECTION: SCIENTIFIC BASIS ---
                with st.expander("📊 View Nutrient Basis (Reference Rates)", expanded=False):
                    st.markdown(f"**Crop:** {selected_crop_label} | **Soil Status:** N:{n_lvl}, P:{p_lvl}, K:{k_lvl}")
                    st.write("Standard recommendation per hectare (Source: crop_npk_rules.json):")
                    ref_col1, ref_col2, ref_col3 = st.columns(3)
                    ref_col1.metric("Target N", f"{base_n} kg/ha")
                    ref_col2.metric("Target P₂O₅", f"{base_p} kg/ha")
                    ref_col3.metric("Target K₂O", f"{base_k} kg/ha")

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
                    mods = get_ph_modifiers(ph_res)
                    n_eff = mods['nitrogen_use_efficiency_multiplier'] * 100
                    p_eff = mods['phosphorus_effective_availability_multiplier'] * 100
                    st.write(f"**N Efficiency:** {n_eff}%")
                    st.write(f"**P Efficiency:** {p_eff}%")

                st.divider()
                st.subheader("Inventory Suitability Report")
                
                # Filter inventory based on sidebar expander selection
                user_inventory = [f for f in inventory if f["name"] in user_selection]

                # Check if the selected inventory can provide the REQUIRED nutrients
                has_n = any(f["n"] > 0 for f in user_inventory)
                has_p = any(f["p"] > 0 for f in user_inventory)
                has_k = any(f["k"] > 0 for f in user_inventory)

                # GAP ANALYSIS: Specifically looking at Adjusted Targets (t_adj)
                missing_nutrients = []
                if t_adj_n > 0 and not has_n: missing_nutrients.append("Nitrogen (N)")
                if t_adj_p > 0 and not has_p: missing_nutrients.append("Phosphorus (P)")
                if t_adj_k > 0 and not has_k: missing_nutrients.append("Potassium (K)")

                if not missing_nutrients:
                    st.info(f"✅ **Status: Just Right.** Your inventory can fulfill the **{raw_area} {unit}** adjusted requirements.")
                else:
                    st.warning(f"⚠️ **Status: Insufficient.** Missing: **{', '.join(missing_nutrients)}**.")
                    
                    # THE "WHY" - Explaining the Adjusted Requirement
                    st.write(f"**Why this matters:** Because your soil pH is **{soil_ph}**, the {selected_crop_label} "
                             f"requires a higher adjusted amount of nutrients to overcome soil locking. "
                             f"Without a source of **{missing_nutrients[0]}**, you will not meet the target "
                             f"yield for this land area.")
                    
                    # Recommendations based on what is missing
                    if "Nitrogen (N)" in missing_nutrients:
                        st.caption("💡 *Tip: Consider adding Urea (46-0-0) or Ammonium Sulfate.*")
                    if "Phosphorus (P)" in missing_nutrients:
                        st.caption("💡 *Tip: Consider adding Solophos (0-20-0) or 16-20-0.*")

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

                display_unit = "sqm" if "sqm" in unit else "ha"
                
                with mix_col1:
                    st.markdown("#### Standard Mix")
                    base_results = solve_npk(t_base_n, t_base_p, t_base_k, inventory, rules, raw_area, display_unit)
                    for res in base_results:
                        with st.expander(f"Using {res['Source']}"):
                            for line in res["Prescription"]: st.info(line)
                            st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

                with mix_col2:
                    st.markdown("#### pH-Adjusted Mix (Recommended)")
                    adj_results = solve_npk(t_base_n, t_base_p, t_base_k, inventory, rules, raw_area, display_unit)
                    for res in adj_results:
                        with st.expander(f"Using {res['Source']}"):
                            for line in res["Prescription"]: st.success(line)
                            st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

    except Exception as e:
        st.error(f"Configuration Error: {e}")


def run_ui_workaround():
    st.set_page_config(page_title="Workaround NPK + pH", layout="centered")
    st.title("🌱 Fertilizer Recommendation Engine (Workaround)")

    try:
        inventory, rules, crop_rules, crop_rules_workaround, ph_rules = load_assets()

        with st.sidebar:
            st.header("1. Land Information")
            unit = st.radio("Select Area Unit", ["Square Meters (sqm)", "Hectares (ha)"])
            raw_area = st.number_input(f"Total Area ({unit})", min_value=1.0, value=500.0 if "sqm" in unit else 1.0)
            
            # INTERNAL CONVERSION
            area_ha = raw_area / 10000 if "sqm" in unit else raw_area
            
            st.write("---")
            st.header("2. Soil & Crop Data")
            selected_crop_label = st.selectbox("Select Crop", options=list(THESIS_CROP_MAP.keys()))
            selected_crop = THESIS_CROP_MAP[selected_crop_label]
            n_lvl = st.selectbox("Nitrogen (N) Status", options=["Low", "Medium", "High"])
            p_lvl = st.selectbox("Phosphorus (P) Status", options=["Low", "Medium", "High"])
            k_lvl = st.selectbox("Potassium (K) Status", options=["Low", "Medium", "High"])
            soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=5.5, step=0.1)

            st.write("---")
            st.header("3. Inventory Management")
            with st.expander("🛒 Plan Your Purchase / Select Inventory", expanded=True):
                fert_names = [f["name"] for f in inventory]
                user_selection = st.multiselect(
                    "Select fertilizers you plan to buy or use:",
                    options=fert_names,
                    default=[], 
                    help="Start typing to search for fertilizers like Urea, 14-14-14, etc."
                )

        # PORTION: Calculation Logic
        rec = get_fertilizer_recommendation(selected_crop, n_lvl, p_lvl, k_lvl, crop_rules_workaround)
        
        if rec:
            base_n, base_p, base_k = rec
            ph_res = run_ph_engine({"crop": selected_crop_label.lower(), "soil_ph": soil_ph}, ph_rules)
            adj_n_ha, adj_p_ha, adj_k_ha = adjust_targets_with_ph(base_n, base_p, base_k, ph_res)

            if st.button("Calculate Prescription", type="primary"):
                # INITIALIZE USER INVENTORY FROM SELECTION
                user_inventory = [f for f in inventory if f["name"] in user_selection]

                # 1. SCALING
                t_base_n, t_base_p, t_base_k = base_n * area_ha, base_p * area_ha, base_k * area_ha
                t_adj_n, t_adj_p, t_adj_k = adj_n_ha * area_ha, adj_p_ha * area_ha, adj_k_ha * area_ha

                st.success(f"✅ Results for {raw_area} {unit}")

                # --- SECTION: SCIENTIFIC BASIS ---
                with st.expander("📊 View Nutrient Basis (Reference Rates)", expanded=False):
                    st.markdown(f"**Crop:** {selected_crop_label} | **Soil Status:** N:{n_lvl}, P:{p_lvl}, K:{k_lvl}")
                    st.write("Standard recommendation per hectare (Source: crop_workaround.json):")
                    ref_col1, ref_col2, ref_col3 = st.columns(3)
                    ref_col1.metric("Target N", f"{base_n} kg/ha")
                    ref_col2.metric("Target P₂O₅", f"{base_p} kg/ha")
                    ref_col3.metric("Target K₂O", f"{base_k} kg/ha")

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
                    mods = get_ph_modifiers(ph_res)
                    n_eff = mods['nitrogen_use_efficiency_multiplier'] * 100
                    p_eff = mods['phosphorus_effective_availability_multiplier'] * 100
                    st.write(f"**N Efficiency:** {n_eff}%")
                    st.write(f"**P Efficiency:** {p_eff}%")

                st.divider()
                st.subheader("Inventory Suitability Report")
                
                # Filter inventory based on sidebar expander selection
                user_inventory = [f for f in inventory if f["name"] in user_selection]

                # Check if the selected inventory can provide the REQUIRED nutrients
                has_n = any(f["n"] > 0 for f in user_inventory)
                has_p = any(f["p"] > 0 for f in user_inventory)
                has_k = any(f["k"] > 0 for f in user_inventory)

                # GAP ANALYSIS: Specifically looking at Adjusted Targets (t_adj)
                missing_nutrients = []
                if t_adj_n > 0 and not has_n: missing_nutrients.append("Nitrogen (N)")
                if t_adj_p > 0 and not has_p: missing_nutrients.append("Phosphorus (P)")
                if t_adj_k > 0 and not has_k: missing_nutrients.append("Potassium (K)")

                if not missing_nutrients:
                    st.info(f"✅ **Status: Just Right.** Your inventory can fulfill the **{raw_area} {unit}** adjusted requirements.")
                else:
                    st.warning(f"⚠️ **Status: Insufficient.** Missing: **{', '.join(missing_nutrients)}**.")
                    
                    # THE "WHY" - Explaining the Adjusted Requirement
                    st.write(f"**Why this matters:** Because your soil pH is **{soil_ph}**, the {selected_crop_label} "
                             f"requires a higher adjusted amount of nutrients to overcome soil locking. "
                             f"Without a source of **{missing_nutrients[0]}**, you will not meet the target "
                             f"yield for this land area.")
                    
                    # Recommendations based on what is missing
                    if "Nitrogen (N)" in missing_nutrients:
                        st.caption("💡 *Tip: Consider adding Urea (46-0-0) or Ammonium Sulfate.*")
                    if "Phosphorus (P)" in missing_nutrients:
                        st.caption("💡 *Tip: Consider adding Solophos (0-20-0) or 16-20-0.*")

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

                display_unit = "sqm" if "sqm" in unit else "ha"
                
                with mix_col1:
                    st.markdown("#### Standard Mix")
                    base_results = solve_npk(t_base_n, t_base_p, t_base_k, inventory, rules, raw_area, display_unit)
                    for res in base_results:
                        with st.expander(f"Using {res['Source']}"):
                            for line in res["Prescription"]: st.info(line)
                            st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

                with mix_col2:
                    st.markdown("#### pH-Adjusted Mix (Recommended)")
                    adj_results = solve_npk(t_base_n, t_base_p, t_base_k, inventory, rules, raw_area, display_unit)
                    for res in adj_results:
                        with st.expander(f"Using {res['Source']}"):
                            for line in res["Prescription"]: st.success(line)
                            st.metric("Total Weight", f"{res['Total Weight']:.2f} kg")

    except Exception as e:
        st.error(f"Configuration Error: {e}")
        
def main():
    run_ui_workaround()  

if __name__ == "__main__":
    run_ui_workaround()
