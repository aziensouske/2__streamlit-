# -------------------------------
# IMPORT LIBRARIES
# -------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(page_title="Housing Prediction", layout="wide")
st.title("🏡 Housing Price Prediction (Decision Tree)")
st.write("This app predicts house prices using a Decision Tree model.")


# -------------------------------
# LOAD DATASET
# -------------------------------
data = pd.read_csv("housing.csv")

# Show original dataset
st.subheader("📊 Original Dataset")
st.write(data.head())


# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
st.subheader("🧹 Missing Value Handling")

# Show missing values before
missing_before = data.isnull().sum()
st.write("🔴 Missing Values Before:")
st.write(missing_before[missing_before > 0])

# Fill missing values
data['total_bedrooms'] = data['total_bedrooms'].fillna(538)

# Show missing values after
missing_after = data.isnull().sum()
st.write("🟢 Missing Values After:")
st.write(missing_after[missing_after > 0])

st.success("Missing values handled successfully ✅")


# -------------------------------
# DROP UNNECESSARY COLUMNS
# -------------------------------
data = data.drop(["longitude", "latitude"], axis=1)


# -------------------------------
# ENCODING (CATEGORICAL → NUMERIC)
# -------------------------------
st.subheader("🔄 Encoding Process")

# Show before encoding
st.write("Before Encoding:")
st.write(data[['ocean_proximity']].head())

# Apply LabelEncoder
encoder = LabelEncoder()
data['ocean_proximity'] = encoder.fit_transform(data['ocean_proximity'])

# Show encoding mapping
encoding_map = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
st.write("Encoding Mapping:")
st.write(encoding_map)

# Show after encoding
st.write("After Encoding:")
st.write(data[['ocean_proximity']].head())

st.success("Categorical data encoded successfully ✅")


# -------------------------------
# SPLIT DATA INTO FEATURES & TARGET
# -------------------------------
x = data.drop(columns="median_house_value")   # features
y = data["median_house_value"]                # target

columns = x.columns.tolist()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42
)


# -------------------------------
# TRAIN MODEL
# -------------------------------
model = DecisionTreeRegressor(max_depth=5)
model.fit(x_train, y_train)

st.success("Model trained successfully ✅")


# -------------------------------
# MODEL PERFORMANCE
# -------------------------------
st.subheader("📈 Model Performance")

st.write("Training Score:", model.score(x_train, y_train))
st.write("Testing Score:", model.score(x_test, y_test))


# -------------------------------
# ACTUAL VS PREDICTED GRAPH
# -------------------------------
st.subheader("📉 Actual vs Predicted")

y_pred = model.predict(x_test)

compare_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.write(compare_df.head())

fig1, ax1 = plt.subplots()
ax1.scatter(compare_df["Actual"], compare_df["Predicted"])
ax1.set_xlabel("Actual Price")
ax1.set_ylabel("Predicted Price")
ax1.set_title("Actual vs Predicted")

st.pyplot(fig1)


# -------------------------------
# FEATURE IMPORTANCE GRAPH
# -------------------------------
st.subheader("📊 Feature Importance")

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": x.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.write(feature_df)

fig2, ax2 = plt.subplots()
ax2.barh(feature_df["Feature"], feature_df["Importance"])
ax2.set_xlabel("Importance")
ax2.set_ylabel("Feature")
ax2.invert_yaxis()

st.pyplot(fig2)


# -------------------------------
# BASIC DATA VISUALIZATION
# -------------------------------
st.subheader("📊 Data Visualization")

fig3, ax3 = plt.subplots(figsize=(16,4))
ax3.hist(x['total_rooms'], bins=20)
ax3.set_xlabel("Total Rooms")
ax3.set_ylabel("Frequency")

st.pyplot(fig3)


# -------------------------------
# SIDEBAR INPUT (USER DATA)
# -------------------------------
st.sidebar.header("🎛️ Enter House Details")

# Initialize session state
if "inputs" not in st.session_state:
    st.session_state.inputs = {}
    for col in columns:
        if col == "ocean_proximity":
            st.session_state.inputs[col] = encoder.classes_[0]
        else:
            st.session_state.inputs[col] = float(x[col].mean())

# Create sidebar inputs
for col in columns:
    if col == "ocean_proximity":
        st.session_state.inputs[col] = st.sidebar.selectbox(
            col,
            options=list(encoder.classes_),
            index=list(encoder.classes_).index(st.session_state.inputs[col])
        )
    else:
        st.session_state.inputs[col] = st.sidebar.slider(
            col,
            float(x[col].min()),
            float(x[col].max()),
            float(st.session_state.inputs[col])
        )


# -------------------------------
# SIDEBAR BUTTONS
# -------------------------------
if st.sidebar.button("🔄 Reset"):
    for col in columns:
        if col == "ocean_proximity":
            st.session_state.inputs[col] = encoder.classes_[0]
        else:
            st.session_state.inputs[col] = float(x[col].mean())
    st.rerun()

predict_btn = st.sidebar.button("🔮 Predict")


# -------------------------------
# PREDICTION
# -------------------------------
input_df = pd.DataFrame([st.session_state.inputs])

# Encode categorical input
input_df['ocean_proximity'] = encoder.transform(input_df['ocean_proximity'])

if predict_btn:
    prediction = model.predict(input_df)
    st.success(f"🏠 Predicted Price: {prediction[0]:,.2f}")


# -------------------------------
# DOWNLOAD REPORT
# -------------------------------
st.subheader("💾 Download Prediction Report")

full_pred = model.predict(x)

report_df = x.copy()
report_df["Actual Price"] = y
report_df["Predicted Price"] = full_pred

st.write(report_df.head())

csv = report_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download CSV Report",
    data=csv,
    file_name="housing_predictions.csv",
    mime="text/csv"
)