# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Water Potability Analysis",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a5276;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2e86c1;
        padding: 0.5rem 0;
    }
    .metric-card {
        background-color: #f0f3f7;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-potable {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .prediction-not-potable {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .stButton > button {
        width: 100%;
        background-color: #2e86c1;
        color: white;
        font-weight: 600;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background-color: #1a5276;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and feature names
@st.cache_resource
def load_model():
    model_data = joblib.load("random_forest_potability_model.pkl")
    return model_data["model"], model_data["features"]

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("water_potability.csv")
    return df

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/000000/water--v1.png", width=80)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📊 Exploratory Data Analysis", "🔮 Predict Potability", "📈 Forecasting", "📋 Model Performance"]
)

# Load model and data
try:
    model, feature_names = load_model()
    df = load_data()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.sidebar.error("⚠️ Model not found. Please train and save the model first.")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.markdown('<p class="main-header">💧 Water Potability Analysis System</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dataset Size</h3>
            <h2>{:,}</h2>
            <p>Total Records</p>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        potable_count = df['Potability'].value_counts()
        potable_pct = (potable_count[1] / len(df) * 100) if 1 in potable_count else 0
        st.markdown("""
        <div class="metric-card">
            <h3>✅ Potable Water</h3>
            <h2>{:.1f}%</h2>
            <p>of samples are potable</p>
        </div>
        """.format(potable_pct), unsafe_allow_html=True)
    
    with col3:
        features_count = len(df.columns) - 1
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Features</h3>
            <h2>{}</h2>
            <p>Water Quality Parameters</p>
        </div>
        """.format(features_count), unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="sub-header">📖 About This Application</p>', unsafe_allow_html=True)
        st.write("""
        This application uses machine learning to predict the potability of water based on various water quality parameters.
        
        **Features include:**
        - **Exploratory Data Analysis** - Visualize and understand the water quality dataset
        - **Potability Prediction** - Input water quality parameters to get real-time predictions
        - **Time Series Forecasting** - Predict future water quality using ARIMA models
        - **Model Performance** - Evaluate model accuracy and metrics
        
        The Random Forest classifier achieves **~77% accuracy** in predicting water potability.
        """)
    
    with col2:
        st.markdown('<p class="sub-header">📊 Quick Stats</p>', unsafe_allow_html=True)
        
        # Sample statistics
        st.metric("pH Range", f"{df['ph'].min():.2f} - {df['ph'].max():.2f}")
        st.metric("Hardness Range", f"{df['Hardness'].min():.2f} - {df['Hardness'].max():.2f}")
        st.metric("Turbidity Range", f"{df['Turbidity'].min():.2f} - {df['Turbidity'].max():.2f}")
    
    st.markdown("---")
    st.info("💡 **Tip:** Navigate using the sidebar to explore different features of the application.")

# ==================== EDA PAGE ====================
elif page == "📊 Exploratory Data Analysis":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    # Dataset Overview
    with st.expander("📋 Dataset Overview", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**DataFrame Info**")
            st.dataframe(df.head(10))
        with col2:
            st.write("**Statistical Summary**")
            st.dataframe(df.describe().round(2))
    
    # Missing Values
    with st.expander("🔍 Missing Values Analysis", expanded=False):
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Feature': missing.index,
            'Missing Count': missing.values,
            'Missing %': (missing.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        
        if len(missing_df) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            bars = ax.barh(missing_df['Feature'], missing_df['Missing %'], color='#e74c3c')
            ax.set_xlabel('Missing Percentage (%)')
            ax.set_title('Missing Values per Feature')
            for bar, val in zip(bars, missing_df['Missing %']):
                ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                        f'{val:.1f}%', va='center')
            st.pyplot(fig)
        else:
            st.success("✅ No missing values found in the dataset.")
    
    # Distribution Plots
    st.markdown('<p class="sub-header">📈 Feature Distributions</p>', unsafe_allow_html=True)
    
    numeric_cols = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
    
    cols_per_row = 3
    for i in range(0, len(numeric_cols), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(numeric_cols):
                feature = numeric_cols[i + j]
                with col:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    df[feature].hist(bins=30, ax=ax, color='#3498db', edgecolor='black')
                    ax.set_title(f'Distribution of {feature}')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('Frequency')
                    st.pyplot(fig)
    
    # Correlation Matrix
    st.markdown('<p class="sub-header">🔗 Correlation Matrix</p>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df[numeric_cols + ['Potability']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix of Water Quality Parameters')
    st.pyplot(fig)
    
    # Potability Distribution
    st.markdown('<p class="sub-header">📊 Potability Distribution</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        potability_counts = df['Potability'].value_counts().sort_index()
        bars = ax.bar(['Not Potable', 'Potable'], potability_counts.values, 
                     color=['#e74c3c', '#2ecc71'])
        ax.set_title('Water Potability Distribution')
        ax.set_ylabel('Number of Samples')
        for bar, val in zip(bars, potability_counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   str(val), ha='center', va='bottom')
        st.pyplot(fig)
    
    with col2:
        st.write("**Summary:**")
        st.write(f"- Total Samples: {len(df):,}")
        st.write(f"- Potable: {potability_counts[1] if 1 in potability_counts else 0:,} ({potability_counts[1]/len(df)*100:.1f}%)")
        st.write(f"- Not Potable: {potability_counts[0] if 0 in potability_counts else 0:,} ({potability_counts[0]/len(df)*100:.1f}%)")
        st.write(f"- Class Imbalance: {'Significant' if abs(potability_counts[0] - potability_counts[1]) > len(df)*0.2 else 'Balanced'}")

# ==================== PREDICTION PAGE ====================
elif page == "🔮 Predict Potability":
    st.markdown('<p class="main-header">🔮 Predict Water Potability</p>', unsafe_allow_html=True)
    
    if not model_loaded:
        st.error("⚠️ Model not loaded. Please ensure the model file exists.")
        st.stop()
    
    st.write("Enter the water quality parameters below to predict if the water is potable.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ph = st.slider("pH", 0.0, 14.0, 7.0, 0.01)
        hardness = st.slider("Hardness (mg/L)", 0.0, 350.0, 200.0, 0.1)
        solids = st.slider("Total Dissolved Solids (ppm)", 0.0, 65000.0, 22000.0, 100.0)
    
    with col2:
        chloramines = st.slider("Chloramines (ppm)", 0.0, 14.0, 7.0, 0.01)
        sulfate = st.slider("Sulfate (mg/L)", 0.0, 500.0, 333.0, 0.1)
        conductivity = st.slider("Conductivity (μS/cm)", 0.0, 800.0, 426.0, 0.1)
    
    with col3:
        organic_carbon = st.slider("Organic Carbon (mg/L)", 0.0, 30.0, 14.0, 0.1)
        trihalomethanes = st.slider("Trihalomethanes (μg/L)", 0.0, 130.0, 66.0, 0.1)
        turbidity = st.slider("Turbidity (NTU)", 0.0, 7.0, 4.0, 0.01)
    
    # Predict button
    if st.button("🔬 Predict Potability", use_container_width=True):
        # Create input DataFrame
        input_data = pd.DataFrame([[
            ph, hardness, solids, chloramines, sulfate,
            conductivity, organic_carbon, trihalomethanes, turbidity
        ]], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if prediction[0] == 1:
                st.markdown("""
                <div class="prediction-card prediction-potable">
                    <h1 style="color: #28a745;">✅ POTABLE</h1>
                    <p style="font-size: 1.2rem;">The water is safe for drinking</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="prediction-card prediction-not-potable">
                    <h1 style="color: #dc3545;">❌ NOT POTABLE</h1>
                    <p style="font-size: 1.2rem;">The water is NOT safe for drinking</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.write("**Probability Scores:**")
            st.metric("Not Potable", f"{probability[0] * 100:.2f}%")
            st.metric("Potable", f"{probability[1] * 100:.2f}%")
            
            # Simple gauge visualization
            fig, ax = plt.subplots(figsize=(4, 2))
            colors = ['#dc3545' if probability[1] < 0.5 else '#28a745']
            ax.barh([''], [probability[1] * 100], color=colors[0])
            ax.set_xlim(0, 100)
            ax.set_xlabel('Probability of Potable (%)')
            ax.set_title('Potability Probability')
            st.pyplot(fig)
        
        # Input Summary
        with st.expander("📋 Input Parameters Summary"):
            input_summary = pd.DataFrame({
                'Parameter': feature_names,
                'Value': input_data.values[0]
            })
            st.dataframe(input_summary)

# ==================== FORECASTING PAGE ====================
elif page == "📈 Forecasting":
    st.markdown('<p class="main-header">📈 Water Quality Forecasting</p>', unsafe_allow_html=True)
    
    st.write("""
    This section uses ARIMA time series models to forecast future water quality parameters 
    based on historical Brisbane water quality data.
    """)
    
    # Load Brisbane data
    try:
        df_brisbane = pd.read_csv("brisbane_water_quality.csv")
        df_brisbane['Timestamp'] = pd.to_datetime(df_brisbane['Timestamp'])
        
        # Process data
        quality_cols = [col for col in df_brisbane.columns if "quality" in col.lower()]
        df_brisbane_clean = df_brisbane.drop(columns=quality_cols + ["Record number"])
        df_brisbane_clean = df_brisbane_clean.groupby("Timestamp", as_index=False).mean()
        df_brisbane_clean = df_brisbane_clean.sort_values("Timestamp").reset_index(drop=True)
        
        # Resample daily
        df_daily = df_brisbane_clean.set_index("Timestamp")[["pH", "Turbidity"]]
        df_daily = df_daily.resample("D").mean()
        df_daily = df_daily.interpolate(method="linear")
        
        st.success(f"✅ Loaded {len(df_daily):,} days of water quality data from Brisbane")
        
        # Display time series plots
        st.markdown('<p class="sub-header">📈 Historical Time Series</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_daily.index, df_daily['pH'], color='#2e86c1', linewidth=1)
            ax.set_title('pH Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('pH')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df_daily.index, df_daily['Turbidity'], color='#e67e22', linewidth=1)
            ax.set_title('Turbidity Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel('Turbidity (NTU)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Forecast
        st.markdown('<p class="sub-header">🔮 Forecast Tomorrow\'s Water Quality</p>', unsafe_allow_html=True)
        
        if st.button("📊 Generate Forecast", use_container_width=True):
            with st.spinner("Generating forecast... This may take a moment."):
                try:
                    from statsmodels.tsa.arima.model import ARIMA
                    
                    # Fit ARIMA models
                    model_ph = ARIMA(df_daily['pH'], order=(2, 1, 2)).fit()
                    model_turbidity = ARIMA(df_daily['Turbidity'], order=(2, 1, 2)).fit()
                    
                    # Forecast
                    forecast_ph = model_ph.forecast(steps=1).iloc[0]
                    forecast_turbidity = model_turbidity.forecast(steps=1).iloc[0]
                    
                    # Get current values
                    current_ph = df_daily['pH'].iloc[-1]
                    current_turbidity = df_daily['Turbidity'].iloc[-1]
                    
                    # Display forecast results
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col2:
                        st.markdown("""
                        <div style="background-color: #f0f3f7; padding: 1.5rem; border-radius: 10px; text-align: center;">
                            <h3>📊 Forecast for Tomorrow</h3>
                            <hr>
                            <div style="display: flex; justify-content: space-around; padding: 1rem 0;">
                                <div>
                                    <h4 style="color: #2e86c1;">pH</h4>
                                    <p style="font-size: 1.5rem; font-weight: 700;">{:.4f}</p>
                                    <p style="color: gray;">Current: {:.4f}</p>
                                    <p style="color: {};">Change: {:.4f}</p>
                                </div>
                                <div>
                                    <h4 style="color: #e67e22;">Turbidity</h4>
                                    <p style="font-size: 1.5rem; font-weight: 700;">{:.4f}</p>
                                    <p style="color: gray;">Current: {:.4f}</p>
                                    <p style="color: {};">Change: {:.4f}</p>
                                </div>
                            </div>
                        </div>
                        """.format(
                            forecast_ph, current_ph, 
                            '#28a745' if forecast_ph > current_ph else '#dc3545', 
                            forecast_ph - current_ph,
                            forecast_turbidity, current_turbidity,
                            '#28a745' if forecast_turbidity < current_turbidity else '#dc3545',
                            forecast_turbidity - current_turbidity
                        ), unsafe_allow_html=True)
                    
                    # Predict potability from forecast
                    other_features_avg = df[['Hardness', 'Solids', 'Chloramines', 'Sulfate', 
                                            'Conductivity', 'Organic_carbon', 'Trihalomethanes']].mean()
                    
                    new_water_forecast = pd.DataFrame([[
                        forecast_ph,
                        other_features_avg['Hardness'],
                        other_features_avg['Solids'],
                        other_features_avg['Chloramines'],
                        other_features_avg['Sulfate'],
                        other_features_avg['Conductivity'],
                        other_features_avg['Organic_carbon'],
                        other_features_avg['Trihalomethanes'],
                        forecast_turbidity
                    ]], columns=feature_names)
                    
                    prediction = model.predict(new_water_forecast)
                    probability = model.predict_proba(new_water_forecast)[0]
                    
                    st.markdown("---")
                    st.markdown('<p class="sub-header">💧 Potability Forecast</p>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if prediction[0] == 1:
                            st.success("✅ **Forecast: POTABLE**")
                            st.write("The predicted water quality is safe for drinking.")
                        else:
                            st.error("❌ **Forecast: NOT POTABLE**")
                            st.write("The predicted water quality is NOT safe for drinking.")
                    
                    with col2:
                        st.write(f"**Confidence:**")
                        st.metric("Not Potable", f"{probability[0] * 100:.2f}%")
                        st.metric("Potable", f"{probability[1] * 100:.2f}%")
                
                except Exception as e:
                    st.error(f"⚠️ Error generating forecast: {str(e)}")
                    st.info("Please ensure statsmodels is installed and data is properly formatted.")
    
    except FileNotFoundError:
        st.warning("⚠️ Brisbane water quality data not found. Please load the dataset to enable forecasting.")
        st.info("Expected file: brisbane_water_quality.csv")

# ==================== MODEL PERFORMANCE PAGE ====================
elif page == "📋 Model Performance":
    st.markdown('<p class="main-header">📋 Model Performance</p>', unsafe_allow_html=True)
    
    if not model_loaded:
        st.error("⚠️ Model not loaded. Please ensure the model file exists.")
        st.stop()
    
    st.write("""
    The Random Forest Classifier was trained on water quality data to predict potability.
    Below are the performance metrics and model evaluation.
    """)
    
    # Model metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Accuracy", "76.83%")
    with col2:
        st.metric("📊 Precision", "81% (Potable)")
    with col3:
        st.metric("🔄 Recall", "71% (Potable)")
    with col4:
        st.metric("📈 ROC-AUC", "0.857")
    
    # Feature Importance
    st.markdown('<p class="sub-header">🔑 Feature Importance</p>', unsafe_allow_html=True)
    
    # Get feature importance from model
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], 
                   color='#3498db')
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance - Random Forest')
    for bar, val in zip(bars, feature_importance_df['Importance']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.3f}', va='center', fontsize=9)
    st.pyplot(fig)
    
    # Confusion Matrix
    st.markdown('<p class="sub-header">📊 Confusion Matrix</p>', unsafe_allow_html=True)
    
    # Use the model to predict on test set
    X = df[feature_names]
    y = df['Potability']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Balance training data
    from sklearn.utils import resample
    train_data = pd.concat([X_train, y_train], axis=1)
    not_potable = train_data[train_data['Potability'] == 0]
    potable = train_data[train_data['Potability'] == 1]
    max_count = max(len(not_potable), len(potable))
    not_potable_upsampled = resample(not_potable, replace=True, n_samples=max_count, random_state=42)
    potable_upsampled = resample(potable, replace=True, n_samples=max_count, random_state=42)
    train_balanced = pd.concat([not_potable_upsampled, potable_upsampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    X_train_balanced = train_balanced.drop('Potability', axis=1)
    y_train_balanced = train_balanced['Potability']
    
    # Train and predict
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=20, 
                                min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X_train_balanced, y_train_balanced)
    y_pred = rf.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Potable', 'Potable'],
                yticklabels=['Not Potable', 'Potable'],
                ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    
    # Classification Report
    st.markdown('<p class="sub-header">📋 Classification Report</p>', unsafe_allow_html=True)
    
    report = classification_report(y_test, y_pred, target_names=['Not Potable', 'Potable'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3))
    
    # ROC Curve
    st.markdown('<p class="sub-header">📈 ROC Curve</p>', unsafe_allow_html=True)
    
    y_prob = rf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})', color='#2e86c1', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random Guessing', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Water Potability Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.caption("💧 Water Potability Analysis System | Built with Streamlit & Scikit-learn")