import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Step function as activation
def step_function(activation):
    return 1 if activation > 0 else 0

# Function to calculate the perceptron output
def perceptron_prediction(x, w, bias):
    activation = np.sum(x * w) + bias
    return step_function(activation)

# Adaline activation function (linear activation)
def adaline_activation(activation):
    return activation

# Function to calculate the Adaline output
def adaline_prediction(x, w, bias):
    activation = np.sum(x * w) + bias
    return adaline_activation(activation)

# Madaline activation function (threshold function)
def madaline_activation(activation):
    return 1 if activation >= 0 else -1

# Function to calculate the Madaline output
def madaline_prediction(x, w, bias):
    activation = np.sum(x * w) + bias
    return madaline_activation(activation)

def main():
    st.sidebar.title("Choose Model")
    model_option = st.sidebar.radio("Select Model", ["sugeno", "mamdani","Perceptron", "Adaline/Madaline"])


    if model_option == "Perceptron":
            st.title("Should I go to the Class? : Model Predictions!")
            choices = {
                "Is test there in the class": 0,
                "Is your friend coming": 1,
                "Is the concept taught important": 2,
                "Is there attendance": 3,
                "Are the portions done": 4
            }

            # Display checkboxes for choices
            selected_choices = [st.checkbox(choice) for choice in choices.keys()]

            # Get selected choices
            selected_features = [choice for choice, selected in zip(choices.keys(), selected_choices) if selected]

            # If at least one choice is selected, display sliders
            if selected_features:
                st.subheader("Adjust sliders for selected features:")
            # Initialize weights array with zeros
            weights = np.zeros(len(choices))

            # Update weights for selected features
            for feature in selected_features:
                weights[choices[feature]] = st.slider(f"How important is this parameter? {feature}", 0, 100, 0) / 100

            # Bias slider
            bias = st.slider("Adjust bias", -100, 100, 0) / 100

            # Calculate perceptron output
            x = np.zeros(len(choices))
            for feature in selected_features:
                x[choices[feature]] = 1

            prediction = perceptron_prediction(x, weights, bias)

            # Display result
            st.subheader("Prediction:")
            st.write("Will the person go to class? ", "Yes" if prediction == 1 else "No")

            # Display table with selected features and weights
            st.subheader("Selected Features and Weights:")
            data = {"Feature": selected_features, "Weight": weights[weights != 0]}  # Display only non-zero weights
            st.table(data)
    elif model_option == "Adaline/Madaline":
            st.title("Should I go to the Class? : Model Predictions!")
            choices = {
                "Is test there in the class": 0,
                "Is your friend coming": 1,
                "Is the concept taught important": 2,
                "Is there attendance": 3,
                "Are the portions done": 4
            }

            # Display checkboxes for choices
            selected_choices = [st.checkbox(choice) for choice in choices.keys()]

            # Get selected choices
            selected_features = [choice for choice, selected in zip(choices.keys(), selected_choices) if selected]

            # If at least one choice is selected, display sliders
            if selected_features:
                st.subheader("Adjust sliders for selected features:")
            # Initialize weights array with zeros
            weights = np.zeros(len(choices))

            # Update weights for selected features
            for feature in selected_features:
                weights[choices[feature]] = st.slider(f"How important is this parameter? {feature}", 0, 100, 0) / 100

            # Bias slider
            bias = st.slider("Adjust bias", -100, 100, 0) / 100
            # Initialize weights array with zeros for Adaline
            adaline_weights = np.zeros(len(choices))

            # Update weights for selected features in Adaline
            for feature in selected_features:
                adaline_weights[choices[feature]] = st.slider(f"How important is this parameter? Adaline {feature}", -1.0, 1.0, 0.0, 0.1, key=f"adaline_{feature}")

            # Bias slider for Adaline
            adaline_bias = st.slider("Adjust bias for Adaline", -1.0, 1.0, 0.0, 0.1, key="adaline_bias")

            # Initialize weights array with zeros for Madaline
            madaline_weights = np.zeros(len(choices))

            # Update weights for selected features in Madaline
            for feature in selected_features:
                madaline_weights[choices[feature]] = st.slider(f"How important is this parameter? Madaline {feature}", -1.0, 1.0, 0.0, 0.1, key=f"madaline_{feature}")

            # Bias slider for Madaline
            madaline_bias = st.slider("Adjust bias for Madaline", -1.0, 1.0, 0.0, 0.1, key="madaline_bias")

            # Calculate Adaline output
            x = np.zeros(len(choices))
            for feature in selected_features:
                x[choices[feature]] = 1

            adaline_prediction_value = adaline_prediction(x, adaline_weights, adaline_bias)

            # Calculate Madaline output
            madaline_prediction_value = madaline_prediction(x, madaline_weights, madaline_bias)

            # Display results
            st.subheader("Adaline Prediction:")
            st.write("Will the person go to class? ", "Yes" if adaline_prediction_value == 1 else "No")

            st.subheader("Madaline Prediction:")
            st.write("Will the person go to class? ", "Yes" if madaline_prediction_value == 1 else "No")

            # Display table with selected features and weights for Adaline
            st.subheader("Adaline Selected Features and Weights:")
            adaline_data = {"Feature": selected_features, "Weight": adaline_weights}
            st.table(adaline_data)

            # Display table with selected features and weights for Madaline
            st.subheader("Madaline Selected Features and Weights:")
            madaline_data = {"Feature": selected_features, "Weight": madaline_weights}
            st.table(madaline_data)

    elif model_option == "mamdani":
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
    elif model_option == "sugeno":
            frequency = ctrl.Antecedent(np.arange(48, 53.1, 0.1), 'frequency')
            demand = ctrl.Antecedent(np.arange(1900, 2301, 1), 'demand')
            tariff = ctrl.Consequent(np.arange(0, 101, 1), 'tariff')

            # Membership functions for antecedents
            frequency['low'] = fuzz.trimf(frequency.universe, [48, 48, 50])
            frequency['medium'] = fuzz.trimf(frequency.universe, [48, 50, 52])
            frequency['high'] = fuzz.trimf(frequency.universe, [50, 52, 52])

            demand['low'] = fuzz.trimf(demand.universe, [1900, 1900, 2100])
            demand['medium'] = fuzz.trimf(demand.universe, [1900, 2100, 2300])
            demand['high'] = fuzz.trimf(demand.universe, [2100, 2300, 2300])

            # Define custom linear functions for the consequent (Sugeno model)
            def tariff_low(x):
                return max(0, min(1, 0))

            def tariff_medium(x):
                return max(0, min(1, x / 50))

            def tariff_high(x):
                return max(0, min(1, 1 - x / 50))

            # Apply the custom functions to the consequent
            tariff['low'] = fuzz.interp_membership(tariff.universe, np.vectorize(tariff_low)(tariff.universe), tariff.universe)
            tariff['medium'] = fuzz.interp_membership(tariff.universe, np.vectorize(tariff_medium)(tariff.universe), tariff.universe)
            tariff['high'] = fuzz.interp_membership(tariff.universe, np.vectorize(tariff_high)(tariff.universe), tariff.universe)

            # Rules
            rule1 = ctrl.Rule(frequency['low'] & demand['low'], tariff['low'])
            rule2 = ctrl.Rule(frequency['low'] & demand['medium'], tariff['medium'])
            rule3 = ctrl.Rule(frequency['low'] & demand['high'], tariff['high'])

            rule4 = ctrl.Rule(frequency['medium'] & demand['low'], tariff['medium'])
            rule5 = ctrl.Rule(frequency['medium'] & demand['medium'], tariff['medium'])
            rule6 = ctrl.Rule(frequency['medium'] & demand['high'], tariff['high'])

            rule7 = ctrl.Rule(frequency['high'] & demand['low'], tariff['high'])
            rule8 = ctrl.Rule(frequency['high'] & demand['medium'], tariff['medium'])
            rule9 = ctrl.Rule(frequency['high'] & demand['high'], tariff['low'])

            # Control System
            tariff_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
            tariff_decision = ctrl.ControlSystemSimulation(tariff_ctrl)

            # Streamlit App
            st.title("Tariff Decision System (Sugeno Model)")

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


if __name__ == "__main__":
    main()


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
        '<p style="font-size: 10px;">Nikil SenthilKumar</p>'
    '<p style="font-size: 10px;">Jairam Vikkranth</p>'
    '<p style="font-size: 10px;">Sanraj S</p>'
    '</div>',
    unsafe_allow_html=True
)