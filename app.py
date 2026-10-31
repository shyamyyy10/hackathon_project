import streamlit as st
import pandas as pd
from PIL import Image
from pathlib import Path
from classifier import load_classifier, predict_image, CLASS_NAMES
from decision_engine import recommend_action
from impact_tracker import log_action, load_actions, summarize_actions

st.set_page_config(page_title="ReLoop AI", page_icon="♻️", layout="wide")
st.title("♻️ ReLoop AI")
st.subheader("From waste identification to smarter reuse and recycling")
st.caption("AI-assisted waste guidance. Always follow your local waste collection rules.")

MODEL_PATH = Path("models/waste_classifier.keras")
model = load_classifier(MODEL_PATH)

with st.sidebar:
    st.header("About ReLoop AI")
    st.write("1. Upload a waste-item photo.")
    st.write("2. Review the predicted category and confidence.")
    st.write("3. Add condition details for a reuse-first recommendation.")
    st.write("4. Record the action to update your impact dashboard.")
    st.warning("The model is a prototype. Image predictions can be wrong.")

tab_scan, tab_impact, tab_about = st.tabs(["🔎 Scan an item", "📊 Impact dashboard", "ℹ️ About"])

with tab_scan:
    uploaded = st.file_uploader("Upload a clear photo of one waste item", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        left, right = st.columns([1, 1])
        with left:
            st.image(image, caption="Uploaded item", use_container_width=True)
        with right:
            if model is None:
                st.error("No trained waste model found yet.")
                st.write("To enable AI predictions:")
                st.code("python train_model.py --data_dir data/garbage_classification")
                st.write("Place the trained model at `models/waste_classifier.keras`.")
                st.info("You can still test the recommendation flow by selecting a category below.")
                predicted = st.selectbox("Test recommendation with category", CLASS_NAMES)
                confidence = None
            else:
                predicted, confidence = predict_image(model, image)
                st.metric("AI prediction", predicted.title())
                st.metric("Model confidence", f"{confidence:.0%}")
                if confidence < 0.60:
                    st.warning("Low confidence: check the prediction or select the correct category below.")
            category = st.selectbox(
                "Confirm material category",
                CLASS_NAMES,
                index=CLASS_NAMES.index(predicted) if predicted in CLASS_NAMES else 0,
                help="Correcting the model's prediction improves the recommendation."
            )
            condition = st.selectbox("Item condition", ["Clean and intact", "Dirty or contaminated", "Damaged", "Unsure"])
            location_rule = st.selectbox("Local collection guidance", [
                "Check local rules", "Accepted for recycling locally", "Not accepted for recycling locally"
            ])
            result = recommend_action(category, condition, location_rule)
            st.markdown("### Recommendation")
            st.success(result["action"])
            st.write(result["reason"])
            st.caption("This is guidance, not a guarantee of local recyclability.")
            if st.button("Record this action", type="primary"):
                log_action(category, condition, result["action"])
                st.success("Action recorded in the local demo log.")

with tab_impact:
    actions = load_actions()
    summary = summarize_actions(actions)
    c1, c2, c3 = st.columns(3)
    c1.metric("Actions recorded", summary["total"])
    c2.metric("Reuse actions", summary["reuse"])
    c3.metric("Recycling actions", summary["recycle"])
    if not actions.empty:
        st.bar_chart(actions["category"].value_counts())
        st.dataframe(actions.tail(20), use_container_width=True)
        st.caption("These are recorded actions, not independently verified environmental savings.")
        if st.button("Clear demo action history"):
            Path("data/actions.csv").unlink(missing_ok=True)
            st.rerun()
    else:
        st.info("No actions recorded yet. Scan an item and record your first action.")

with tab_about:
    st.markdown("""
    **What it does**
    - Classifies common waste categories using a transfer-learning image model.
    - Combines the prediction with user-confirmed condition and local rules.
    - Gives an explainable reuse/recycling recommendation.
    - Tracks user-recorded actions without inventing carbon-saving claims.

    **Prototype limitations**
    - Image classification does not reliably detect contamination or prove an item is recyclable.
    - Local collection rules vary; choose the correct local option before following advice.
    - Impact metrics are activity counts unless supported by verified measurement data.
    """)
