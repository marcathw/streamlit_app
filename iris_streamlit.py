import streamlit as st
import joblib
import numpy as np

model = joblib.load("irisdataset.pkl")

def main():
    st.title("Machine Learning Model Deployment")

    sl = st.slider("sepal_length", min_value = 0.0, max_value = 10.0, value = 0.0)
    sw = st.slider("sepal_width", min_value = 0.0, max_value = 10.0, value = 0.0)
    pl = st.slider("petal_length", min_value = 0.0, max_value = 10.0, value = 0.0)
    pw = st.slider("petal_width", min_value = 0.0, max_value = 10.0, value = 0.0)
    
    if st.button("Make Prediction"):
        features = [sl, sw, pl, pw]
        result = make_prediction(features)
        st.success(f"The prediction is: {result}")

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == "__main__":
    main()