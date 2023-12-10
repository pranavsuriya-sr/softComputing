import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

frequency = ctrl.Antecedent(np.arange(48, 53.1, 0.1), 'frequency')
demand = ctrl.Antecedent(np.arange(1900, 2301, 1), 'demand')
tariff = ctrl.Consequent(np.arange(0, 101, 1), 'tariff')

frequency['low'] = fuzz.trimf(frequency.universe, [48, 48, 50])
frequency['medium'] = fuzz.trimf(frequency.universe, [48, 50, 52])
frequency['high'] = fuzz.trimf(frequency.universe, [50, 52, 52])

demand['low'] = fuzz.trimf(demand.universe, [1900, 1900, 2100])
demand['medium'] = fuzz.trimf(demand.universe, [1900, 2100, 2300])
demand['high'] = fuzz.trimf(demand.universe, [2100, 2300, 2300])

tariff['low'] = fuzz.trimf(tariff.universe, [0, 0, 50])
tariff['medium'] = fuzz.trimf(tariff.universe, [0, 50, 100])
tariff['high'] = fuzz.trimf(tariff.universe, [50, 100, 100])

rule1 = ctrl.Rule(frequency['low'] & demand['low'], tariff['low'])
rule2 = ctrl.Rule(frequency['low'] & demand['medium'], tariff['medium'])
rule3 = ctrl.Rule(frequency['low'] & demand['high'], tariff['high'])

rule4 = ctrl.Rule(frequency['medium'] & demand['low'], tariff['medium'])
rule5 = ctrl.Rule(frequency['medium'] & demand['medium'], tariff['medium'])
rule6 = ctrl.Rule(frequency['medium'] & demand['high'], tariff['high'])

rule7 = ctrl.Rule(frequency['high'] & demand['low'], tariff['high'])
rule8 = ctrl.Rule(frequency['high'] & demand['medium'], tariff['medium'])
rule9 = ctrl.Rule(frequency['high'] & demand['high'], tariff['low'])

tariff_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
tariff_decision = ctrl.ControlSystemSimulation(tariff_ctrl)

st.title("Tariff Decision System")
user_frequency = st.slider("Select Frequency (Hz):", min_value=48.0, max_value=52.0, step=0.1, value=50.0)
user_demand = st.slider("Select Demand (MW):", min_value=1900, max_value=2300, value=2100)
tariff_decision.input['frequency'] = user_frequency
tariff_decision.input['demand'] = user_demand
tariff_decision.compute()
st.set_option('deprecation.showPyplotGlobalUse', False)
st.subheader("Fuzzy Logic Output:")
st.write(f"Frequency: {user_frequency} Hz")
st.write(f"Demand: {user_demand} MW")
st.write(f"Tariff: {tariff_decision.output['tariff']}")

st.subheader("Membership Functions:")
st.pyplot(frequency.view(sim=tariff_decision))
st.pyplot(demand.view(sim=tariff_decision))
st.pyplot(tariff.view(sim=tariff_decision))

# Copyright text at the bottom - No edit
st.markdown(
    '<div style="text-align:center; margin-top: 42px">'
    '<a href = "https://pranavsuriya-sr.github.io/personalPortfolio/" style = "text-decoration: none;" ><p style="font-size: 10px;">PranavSuriya Devs © 2023 Project Hack Community.</a></p>'
    '<p style="font-size: 10px;">Open Source rights reserved.</p>'
    '</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div style="text-align:center; margin-top: 42px">'
    '<a href = "https://pranavsuriya-sr.github.io/personalPortfolio/" style = "text-decoration: none;" ><p style="font-size: 10px;">PranavSuriya Devs © 2023 Project Hack Community.</a></p>'
    '<p style="font-size: 10px;">Open Source rights reserved.</p>'
    '</div>',
    unsafe_allow_html=True
)