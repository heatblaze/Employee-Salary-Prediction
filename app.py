import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv(r'D:/daw n/Employee Salary Prediction/adult 3.csv')

# Clean data
for col in df.columns:
    df[col] = df[col].replace(' ?', np.nan)
df.dropna(inplace=True)

# Encode categorical
label_encoders = {}
categorical_cols = df.select_dtypes(include='object').columns.drop('income')
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

income_le = LabelEncoder()
df['income'] = income_le.fit_transform(df['income'])

X = df.drop('income', axis=1)
y = df['income']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

joblib.dump(model, 'salary_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(income_le, 'income_encoder.pkl')

print("Model and encoders saved successfully.")

# Streamlit UI starts here
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>Employee Income Prediction</h1>
    <hr style='border: 1px solid #4CAF50;'>
""", unsafe_allow_html=True)

st.subheader('Enter Employee Details Below')

user_data = {}

for col in X.columns:
    if col in categorical_cols:
        options = label_encoders[col].classes_.tolist()
        user_data[col] = st.selectbox(f'{col}', options)
    else:
        if col == 'age':
            user_data[col] = st.slider('Age', 18, 90, 30)
        elif col == 'hours-per-week':
            user_data[col] = st.slider('Hours per Week', 1, 100, 40)
        elif col == 'educational-num':
            user_data[col] = st.slider('Educational Number', 1, 16, 9)
        else:
            user_data[col] = st.number_input(f'{col}', min_value=0)

if st.button('Predict Income'):
    input_df = pd.DataFrame([user_data])
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    prediction = model.predict(input_df)
    income_result = income_le.inverse_transform(prediction)[0]

    st.success(f'**Predicted Income Range: {income_result}**')

    # Estimate annual salary
    if income_result == '>50K':
        estimated_salary = np.random.randint(51000, 100000)
    else:
        estimated_salary = np.random.randint(20000, 49000)

    # Estimate income tax (assume 20% for >50K and 10% for <=50K)
    if income_result == '>50K':
        estimated_tax = estimated_salary * 0.2
    else:
        estimated_tax = estimated_salary * 0.1

    st.info(f"Estimated Annual Salary: ${estimated_salary:,}")
    st.info(f"Estimated Income Tax: ${estimated_tax:,.2f}")

    st.balloons()

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("Model trained on Adult Census Data | 2025 © Employee Income Predictor")
