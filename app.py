import os

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_PATH = "iris_model.joblib"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found."
        )

    return joblib.load(MODEL_PATH)


# ============================================================
# APPLICATION TITLE
# ============================================================

st.title("🌸 Iris Flower Classifier")

st.write(
    """
    Enter the measurements of an Iris flower below.
    The trained Decision Tree model will predict
    the flower species.
    """
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    bundle = load_model()

    model = bundle["model"]
    class_names = bundle["class_names"]
    feature_names = bundle["feature_names"]

except Exception as error:

    st.error(
        "The machine learning model could not be loaded."
    )

    st.exception(error)

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("About the Model")

st.sidebar.write(
    "**Algorithm:** Decision Tree Classifier"
)

st.sidebar.write(
    "**Dataset:** Iris Dataset"
)

st.sidebar.write(
    "**Classes:** Setosa, Versicolor, Virginica"
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("🌿 Flower Measurements")

col1, col2 = st.columns(2)


with col1:

    sepal_length = st.slider(
        "Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.8,
        step=0.1
    )

    sepal_width = st.slider(
        "Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.0,
        step=0.1
    )


with col2:

    petal_length = st.slider(
        "Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=4.0,
        step=0.1
    )

    petal_width = st.slider(
        "Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.2,
        step=0.1
    )


# ============================================================
# PREPARE INPUT
# ============================================================

input_data = pd.DataFrame(
    [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]],
    columns=feature_names
)


# ============================================================
# DISPLAY INPUT
# ============================================================

with st.expander("View Input Data"):

    st.dataframe(
        input_data,
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔮 Predict Species",
    type="primary",
    use_container_width=True
):

    try:

        prediction = model.predict(
            input_data
        )[0]

        predicted_species = class_names[
            int(prediction)
        ]

        st.success(
            f"### Predicted Species: "
            f"{predicted_species.title()}"
        )


        # ----------------------------------------------------
        # PREDICTION PROBABILITIES
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_data
            )[0]

            probability_df = pd.DataFrame(
                {
                    "Species": [
                        name.title()
                        for name in class_names
                    ],
                    "Probability": probabilities * 100
                }
            )

            probability_df = (
                probability_df
                .sort_values(
                    "Probability",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            st.subheader(
                "📊 Prediction Probabilities"
            )

            st.dataframe(
                probability_df.style.format(
                    {
                        "Probability": "{:.2f}%"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                probability_df.set_index(
                    "Species"
                )["Probability"]
            )


    except Exception as error:

        st.error(
            "An error occurred during prediction."
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Machine Learning Portfolio Project | "
    "Decision Tree + Streamlit"
)
