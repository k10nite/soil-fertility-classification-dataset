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
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str) and ("–" in val or "-" in val):
        val = val.replace("–", "-")
        parts = [float(p.strip()) for p in val.split("-")]
        return sum(parts) / len(parts)
    try:
        return float(val)
    except Exception:
        return 0.0

def get_fertilizer_recommendation(crop_name, n_status, p_status, k_status, crop_rules):
    crop_data = crop_rules.get(crop_name)
    if not crop_data:
        return None

    t_n = parse_target_value(crop_data["N"].get(n_status, 0))
    t_p = parse_target_value(crop_data["P"].get(p_status, 0))
    t_k = parse_target_value(crop_data["K"].get(k_status, 0))
    return t_n, t_p, t_k

def merge_value(context, key, value):
    if key in context and isinstance(context[key], list) and isinstance(value, list):
        for item in value:
            if item not in context[key]:
                context[key].append(item)
    else:
        context[key] = value

def get_context_value(field_name, data, context):
    if field_name in context:
        return context[field_name]
    return data.get(field_name)

def evaluate_operator(actual, operator, expected, data, context):
    if operator == "lt":
        return actual < expected
    if operator == "lte":
        return actual <= expected
    if operator == "gt":
        return actual > expected
    if operator == "gte":
        return actual >= expected
    if operator == "lt_field":
        return actual < get_context_value(expected, data, context)
    if operator == "lte_field":
        return actual <= get_context_value(expected, data, context)
    if operator == "gt_field":
        return actual > get_context_value(expected, data, context)
    if operator == "gte_field":
        return actual >= get_context_value(expected, data, context)
    raise ValueError(f"Unsupported operator: {operator}")

def check_condition(condition, data, context, ph_rules):
    for key, expected in condition.items():
        if key == "crop_not_supported":
            supported = ph_rules["scope"]["supported_crops"]
            actual_result = data.get("crop") not in supported
            if actual_result != expected:
                return False
            continue

        actual = get_context_value(key, data, context)

        if not isinstance(expected, dict):
            if actual != expected:
                return False
            continue

        if actual is None:
            return False

        for op, op_value in expected.items():
            if not evaluate_operator(actual, op, op_value, data, context):
                return False

    return True

def apply_then(then_block, context):
    for key, value in then_block.items():
        merge_value(context, key, value)

def run_logic(rule, data, context):
    if rule["id"] == "TARGET_001":
        soil_ph = get_context_value("soil_ph", data, context)
        lime_trigger_ph = get_context_value("lime_trigger_ph", data, context)
        target_ph = get_context_value("target_ph", data, context)

        if soil_ph is None or lime_trigger_ph is None or target_ph is None:
            context["within_target_range"] = False
        else:
            context["within_target_range"] = (
                soil_ph >= lime_trigger_ph and soil_ph <= (target_ph + 0.5)
            )

def run_rule_array(array_name, data, context, ph_rules):
    for rule in ph_rules.get(array_name, []):
        matched = False

        if "if" in rule:
            matched = check_condition(rule["if"], data, context, ph_rules)
            if matched and "then" in rule:
                apply_then(rule["then"], context)
        elif "logic" in rule:
            run_logic(rule, data, context)
            matched = True

        if matched:
            context["decision_trace"].append({
                "array": array_name,
                "rule_id": rule["id"]
            })

def run_ph_engine(data, ph_rules):
    filtered_data = {
        "crop": data["crop"],
        "soil_ph": data["soil_ph"],
        # optional inputs not yet verified by specialist
        "buffer_ph": None,
        "lime_requirement_value": None,
        "potato_scab_sensitive": data.get("potato_scab_sensitive", None),
        "legume_rotation_present": None,
    }

    context = dict(filtered_data)
    context.setdefault("warnings", [])
    context.setdefault("actions", [])
    context.setdefault("decision_trace", [])

    for array_name in RULE_ARRAY_ORDER:
        run_rule_array(array_name, filtered_data, context, ph_rules)

    return context

def get_ph_modifiers(ph_result):
    soil_ph = ph_result.get("soil_ph", 0)

    if soil_ph < 5.5:
        return {
            "phosphorus_effective_availability_multiplier": 0.75,
            "nitrogen_use_efficiency_multiplier": 0.85,
            "confidence_penalty": "high",
        }
    if 5.5 <= soil_ph < 6.0:
        return {
            "phosphorus_effective_availability_multiplier": 0.9,
            "nitrogen_use_efficiency_multiplier": 0.95,
            "confidence_penalty": "medium",
        }
    if 6.0 <= soil_ph <= 7.0:
        return {
            "phosphorus_effective_availability_multiplier": 1.0,
            "nitrogen_use_efficiency_multiplier": 1.0,
            "confidence_penalty": "none",
        }
    if soil_ph > 7.5:
        return {
            "phosphorus_effective_availability_multiplier": 0.9,
            "confidence_penalty": "medium",
            "micronutrient_watchlist": ["iron", "zinc", "manganese"],
        }
    return {}

def adjust_targets_with_ph(t_n, t_p, t_k, ph_result):
    adjusted_n = float(t_n)
    adjusted_p = float(t_p)
    adjusted_k = float(t_k)

    modifiers = get_ph_modifiers(ph_result)

    p_mult = modifiers.get("phosphorus_effective_availability_multiplier", 1.0)
    n_mult = modifiers.get("nitrogen_use_efficiency_multiplier", 1.0)

    if n_mult > 0:
        adjusted_n = round(adjusted_n / n_mult, 2)
    if p_mult > 0:
        adjusted_p = round(adjusted_p / p_mult, 2)

    ph_result["integration_modifiers"] = modifiers
    return adjusted_n, adjusted_p, adjusted_k

def ph_multiplier_triggered(ph_result):
    modifiers = ph_result.get("integration_modifiers", {})
    return (
        modifiers.get("phosphorus_effective_availability_multiplier") not in (None, 1.0)
        or modifiers.get("nitrogen_use_efficiency_multiplier") not in (None, 1.0)
    )

def solve_npk(t_n, t_p, t_k, inventory, rules):
    results = []
    precision = rules["constraints"]["precision_decimals"]
    allow_over = rules["constraints"]["allow_over_fertilization"]

    n_filler = next(f for f in inventory if f["n"] > 0 and f["p"] == 0 and f["k"] == 0)
    k_filler = next(f for f in inventory if f["k"] > 0 and f["n"] == 0 and f["p"] == 0)
    p_sources = [f for f in inventory if f["p"] > 0]

    for p_fert in p_sources:
        qty_p = (t_p / p_fert["p"]) * 100 if p_fert["p"] > 0 else 0

        n_provided = (qty_p * p_fert["n"]) / 100
        p_provided = (qty_p * p_fert["p"]) / 100
        k_provided = (qty_p * p_fert["k"]) / 100

        rem_n = t_n - n_provided
        rem_k = t_k - k_provided

        if not allow_over and (rem_n < -0.01 or rem_k < -0.01):
            continue

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

def build_zero_results():
    return [{
        "Source": "No pH multiplier applied",
        "Prescription": ["0kg/ha of Urea", "0kg/ha of Phosphorus Source", "0kg/ha of Muriate of Potash"],
        "Total Weight": 0.0,
        "Applied N": 0.0,
        "Applied P": 0.0,
        "Applied K": 0.0,
    }]

st.set_page_config(page_title="Rule-Based NPK + pH", layout="centered")
st.title("🌱 Fertilizer Recommendation Engine")

st.markdown("""
<style>
.block-container {
    max-width: 75%;
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)


try:
    inventory, rules, crop_rules, ph_rules = load_assets()

    with st.sidebar:
        st.header("Input Parameters")
        selected_crop_label = st.selectbox("Select Crop", options=list(THESIS_CROP_MAP.keys()))
        selected_crop = THESIS_CROP_MAP[selected_crop_label]

        st.write("---")
        st.write("**Soil Analysis Results**")
        n_lvl = st.selectbox("Nitrogen (N) Status", options=["L", "M", "H", "VH"])
        p_lvl = st.selectbox("Phosphorus (P) Status", options=["L", "ML", "MH", "H", "VH"])
        k_lvl = st.selectbox("Potassium (K) Status", options=["L", "S", "S+", "S++/+++"])
        soil_ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=5.4, step=0.1)

        potato_scab_sensitive = None
        if selected_crop_label == "Potato":
            potato_scab_sensitive = st.toggle("Potato Scab Sensitive", value=False)

    recommendation = get_fertilizer_recommendation(selected_crop, n_lvl, p_lvl, k_lvl, crop_rules)

    if recommendation is None:
        st.error(f"No crop rule found for {selected_crop_label}.")
    else:
        base_n, base_p, base_k = recommendation

        ph_input = {
            "crop": selected_crop_label.lower().replace(" ", "_"),
            "soil_ph": soil_ph,
            "potato_scab_sensitive": potato_scab_sensitive,
        }
        ph_result = run_ph_engine(ph_input, ph_rules)
        adjusted_n, adjusted_p, adjusted_k = adjust_targets_with_ph(base_n, base_p, base_k, ph_result)
        multiplier_used = ph_multiplier_triggered(ph_result)

        if st.button("Calculate Prescription", type="primary"):
            base_results = solve_npk(base_n, base_p, base_k, inventory, rules)
            adjusted_results = solve_npk(adjusted_n, adjusted_p, adjusted_k, inventory, rules) if multiplier_used else build_zero_results()

            st.subheader("Soil pH Assessment")
            left, right = st.columns(2)

            ph_status = ph_result.get("ph_status", "N/A").replace("_", " ").title()
            target_ph = ph_result.get("target_ph", "N/A")
            lime_trigger_ph = ph_result.get("lime_trigger_ph", "N/A")
            within_range = ph_result.get("within_target_range", False)
            liming_needed = ph_result.get("liming_recommended", False)
            lime_test_needed = ph_result.get("lime_requirement_test_needed", False)

            with left:
                status_key = ph_result.get("ph_status", "")
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

            if ph_result.get("warnings"):
                for warning in ph_result["warnings"]:
                    st.warning(warning)

            effects = ph_result.get("nutrient_availability_effects", [])
            if effects:
                st.markdown("### Nutrient Impact")
                for effect in effects:
                    parameter = effect["parameter"].replace("_", " ").title()
                    impact = effect["effect"].replace("_", " ").title()
                    st.info(f"{parameter}: {impact}")

            st.subheader("Target Recommendation Summary")
            summary_df = pd.DataFrame([
                {
                    "Scenario": "Base",
                    "N (kg/ha)": round(base_n, 2),
                    "P (kg/ha)": round(base_p, 2),
                    "K (kg/ha)": round(base_k, 2),
                },
                {
                    "Scenario": "pH Multiplier Effect" if multiplier_used else "pH Multiplier Effect (Not Triggered)",
                    "N (kg/ha)": round(adjusted_n, 2) if multiplier_used else 0.0,
                    "P (kg/ha)": round(adjusted_p, 2) if multiplier_used else 0.0,
                    "K (kg/ha)": round(adjusted_k, 2) if multiplier_used else 0.0,
                }
            ])
            st.table(
                summary_df.style.format({
                    "N (kg/ha)": "{:.2f}",
                    "P (kg/ha)": "{:.2f}",
                    "K (kg/ha)": "{:.2f}",
                })
            )
            if multiplier_used:
                st.caption("Second row shows pH-adjusted nutrient requirements after applying the pH multiplier.")
            else:
                st.caption("No pH multiplier was triggered, so the pH-adjusted row is shown as zero.")

            st.subheader("Optimized Fertilizer Mixes")
            col_base, col_adj = st.columns(2)

            with col_base:
                st.markdown("#### Base Recommendation")
                if not base_results:
                    st.warning("No combinations found for the base recommendation.")
                else:
                    for i, res in enumerate(base_results):
                        with st.expander(f"Option {i+1}: {res['Source']} Base", expanded=(i == 0)):
                            for line in res["Prescription"]:
                                st.success(line)
                            st.metric("Total Weight", f"{res['Total Weight']:.2f} kg/ha")
                            st.caption(
                                f"Applied N-P-K: "
                                f"{res['Applied N']:.2f}-{res['Applied P']:.2f}-{res['Applied K']:.2f}"
                            )

            with col_adj:
                st.markdown("#### pH-Adjusted Recommendation")
                if not adjusted_results:
                    st.warning("No combinations found for the pH-adjusted recommendation.")
                else:
                    for i, res in enumerate(adjusted_results):
                        with st.expander(f"Option {i+1}: {res['Source']}", expanded=(i == 0)):
                            for line in res["Prescription"]:
                                st.success(line)
                            st.metric("Total Weight", f"{res['Total Weight']:.2f} kg/ha")
                            st.caption(
                                f"Applied N-P-K: "
                                f"{res['Applied N']:.2f}-{res['Applied P']:.2f}-{res['Applied K']:.2f}"
                            )

            with st.expander("Decision Trace"):
                st.json(ph_result.get("decision_trace", []))

except Exception as e:
    st.error(f"Configuration Error: {e}")
    st.info("Ensure all JSON files are present inside the data folder.")
