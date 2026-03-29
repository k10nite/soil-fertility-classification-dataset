import streamlit as st
import json
import pandas as pd

def load_assets():
    with open('fertilizers.json', 'r') as f:
        inventory = json.load(f)['inventory']
    with open('engine_rules.json', 'r') as r:
        rules = json.load(r)['engine_logic']
    return inventory, rules

def solve_npk(t_n, t_p, t_k, inventory, rules):
    results = []
    precision = rules['constraints']['precision_decimals']
    allow_over = rules['constraints']['allow_over_fertilization']
    
    # Identify standard fillers (Singles)
    n_filler = next(f for f in inventory if f['n'] > 0 and f['p'] == 0 and f['k'] == 0)
    k_filler = next(f for f in inventory if f['k'] > 0 and f['n'] == 0 and f['p'] == 0)

    # Step 2 from JSON: SOLVE_FOR_PHOSPHORUS
    p_sources = [f for f in inventory if f['p'] > 0]

    for p_fert in p_sources:
        # Calculate quantity for P
        qty_p = (t_p / p_fert['p']) * 100
        
        # Step 3 from JSON: CALCULATE_REMAINDER logic
        n_provided = (qty_p * p_fert['n']) / 100
        k_provided = (qty_p * p_fert['k']) / 100
        
        rem_n = t_n - n_provided
        rem_k = t_k - k_provided

        # Constraint check: IDENTIFY_LIMITING_NUTRIENT
        # If any remainder is negative and over-fertilization is forbidden, skip this combo
        if not allow_over:
            if rem_n < -0.01 or rem_k < -0.01: # Small margin for float precision
                continue

        # Top up gaps using fillers
        qty_n = (max(0, rem_n) / n_filler['n']) * 100
        qty_k = (max(0, rem_k) / k_filler['k']) * 100

        # Format according to JSON output_format
        fmt = rules['output_format']
        prescription = []
        if qty_n > 0: prescription.append(fmt.format(qty=round(qty_n, precision), fertilizer_name=n_filler['name']))
        prescription.append(fmt.format(qty=round(qty_p, precision), fertilizer_name=p_fert['name']))
        if qty_k > 0: prescription.append(fmt.format(qty=round(qty_k, precision), fertilizer_name=k_filler['name']))

        results.append({
            "Source": p_fert['name'],
            "Prescription": prescription,
            "Total Weight": qty_n + qty_p + qty_k
        })

    return sorted(results, key=lambda x: x['Total Weight'])[:rules['constraints']['max_combinations']]

# --- UI Implementation ---
st.set_page_config(page_title="Rule-Based NPK", layout="centered")
st.title("🌱 NPK Fertilizer Engine")

try:
    inventory, rules = load_assets()
    
    with st.form("input_form"):
        st.write("### Enter Target Nutrient Values")
        c1, c2, c3 = st.columns(3)
        in_n = c1.number_input("Target N", value=260.0)
        in_p = c2.number_input("Target P", value=60.0)
        in_k = c3.number_input("Target K", value=60.0)
        submitted = st.form_submit_button("Calculate")

    if submitted:
        results = solve_npk(in_n, in_p, in_k, inventory, rules)
        
        if not results:
            st.warning("No combinations found within the constraints (Over-fertilization is turned OFF).")
        else:
            for i, res in enumerate(results):
                st.markdown(f"#### Option {i+1} (Base: {res['Source']})")
                for line in res['Prescription']:
                    st.success(line)
                st.caption(f"Total applied weight: {res['Total Weight']:.2f} kg/ha")
                st.divider()

except Exception as e:
    st.error(f"Error: {e}")