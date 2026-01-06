import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
import warnings 
from scipy import stats

warnings.filterwarnings("ignore")

# 1. Page Configuration
st.set_page_config(
    page_title="AI Student Score Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS
custom_css = """
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        color: #f1f5f9;
    }
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px rgba(99, 102, 241, 0.3);
        text-align: center;
    }
    .prediction-card {
        background: rgba(255, 255, 255, 0.05);
        border-left: 8px solid #6366f1;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
    }
    .score-display {
        font-size: 3rem;
        font-weight: 800;
        color: #ffffff;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%) !important;
        color: white !important;
        padding: 12px 32px !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        width: 100% !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🎓 AI Student Score Predictor</h1><p>Predicting exact final scores based on study habits and AI usage</p></div>', unsafe_allow_html=True)

# 3. Data Loading
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif os.path.exists('ai_impact_student_performance_dataset.csv'):
        return pd.read_csv('ai_impact_student_performance_dataset.csv')
    return None

with st.sidebar:
    st.header("📂 Data Configuration")
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
    
df = load_data(uploaded_file)

if df is None:
    st.warning("⚠️ Dataset not found. Please upload 'ai_impact_student_performance_dataset.csv' in the sidebar.")
    st.stop()

# 4. Regression Model Training
@st.cache_resource
def train_model(df):
    if 'final_score' not in df.columns:
        return None, None, None, None, None, None

    X = df.drop(['final_score'], axis=1)
    y = df['final_score']
    
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    X = X.fillna(X.mean(numeric_only=True))
    
    scaler = StandardScaler()
    feature_names = X.columns.tolist()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred
    }
    
    return model, feature_names, label_encoders, metrics, test_data, scaler

# Unpack results safely
model, feature_names, label_encoders, metrics, test_data, scaler = train_model(df)

if model is None:
    st.error("❌ Error: The dataset must contain a 'final_score' column.")
    st.stop()

def prepare_input(input_df):
    X_input = input_df.copy()
    for col in label_encoders:
        if col in X_input.columns:
            try:
                val = X_input[col].iloc[0]
                if val in label_encoders[col].classes_:
                    X_input[col] = label_encoders[col].transform(X_input[col].astype(str))
                else:
                    X_input[col] = 0 
            except:
                X_input[col] = 0
    X_input = X_input.fillna(0)
    for feat in feature_names:
        if feat not in X_input.columns:
            X_input[feat] = 0
    X_input = X_input[feature_names]
    X_scaled = scaler.transform(X_input)
    return pd.DataFrame(X_scaled, columns=feature_names)

def get_top_features(n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:n]
    return [(feature_names[i], importances[i]) for i in indices]

def add_trendline(fig, x, y):
    """Add trendline to scatter plot without statsmodels"""
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_trend = np.array(sorted(x))
    y_trend = p(x_trend)
    fig.add_trace(go.Scatter(
        x=x_trend, y=y_trend,
        mode='lines',
        name='Trendline',
        line=dict(color='#ff6b6b', width=2, dash='dash')
    ))
    return fig

# 5. UI Layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction", "📈 Analytics", "📚 Features", "ℹ️ About"])

with tab1:
    st.header("👤 Student Profile")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📋 Demographics")
        grade = st.selectbox("Grade Level", ['10th', '11th', '12th', '1st Year', '2nd Year', '3rd Year'])
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        age = st.number_input("Age", 15, 35, 20)
        improve = st.number_input("Improvement Rate (%)", -20.0, 50.0, 10.0)
    
    with col2:
        st.subheader("🏥 Lifestyle")
        study = st.number_input("Study Hours/Day", 0.0, 15.0, 3.5)
        sleep = st.number_input("Sleep Hours/Day", 0.0, 12.0, 7.0)
        social = st.number_input("Social Media Hours/Day", 0.0, 12.0, 2.5)
        concept = st.slider("Concept Understanding", 1, 10, 6)
    
    with col3:
        st.subheader("🤖 AI Usage")
        ai_time = st.number_input("AI Usage (Min/Day)", 0, 300, 60)
        ai_content = st.slider("AI Generated %", 0, 100, 30)
        ai_depend = st.slider("AI Dependency", 1, 10, 5)
        particip = st.slider("Class Participation", 1, 10, 6)

    st.divider()
    
    input_data = pd.DataFrame({
        'age': [age], 'gender': [gender], 'grade_level': [grade],
        'study_hours_per_day': [study], 'uses_ai': [1],
        'ai_usage_time_minutes': [ai_time], 'ai_tools_used': ['ChatGPT'],
        'ai_usage_purpose': ['Notes'], 'ai_dependency_score': [ai_depend],
        'ai_generated_content_percentage': [ai_content], 'ai_prompts_per_week': [50],
        'ai_ethics_score': [5], 'concept_understanding_score': [concept],
        'study_consistency_index': [5.5], 'improvement_rate': [improve],
        'sleep_hours': [sleep], 'social_media_hours': [social],
        'tutoring_hours': [2.0], 'class_participation_score': [particip]
    })
    
    if st.button("🚀 Predict Final Score", use_container_width=True):
        prepared = prepare_input(input_data)
        predicted_score = model.predict(prepared)[0]
        
        # Ensure score is between 0-100
        predicted_score = max(0, min(100, predicted_score))
        
        st.markdown(f"""
        <div class="prediction-card">
            <h3>✨ Predicted Final Score</h3>
            <div class="score-display">{predicted_score:.1f} / 100</div>
            <p style="font-size: 1.2em;">Status: {"🟢 PASS" if predicted_score >= 50 else "🔴 AT RISK (FAIL)"}</p>
        </div>
        """, unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Score Gauge"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#6366f1"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(16, 185, 129, 0.2)"}
                ]
            }
        ))
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📊 Regression Analysis")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 R-Squared", f"{metrics['r2']:.3f}")
    m2.metric("📏 Mean Absolute Error", f"{metrics['mae']:.2f} pts")
    m3.metric("📊 RMSE", f"{metrics['rmse']:.2f} pts")
    
    st.divider()
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("📍 Actual vs. Predicted")
        plot_df = pd.DataFrame({
            'Actual': test_data['y_test'].values, 
            'Predicted': test_data['y_pred']
        })
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scatter(
            x=plot_df['Actual'],
            y=plot_df['Predicted'],
            mode='markers',
            marker=dict(size=8, color='#6366f1', opacity=0.6),
            name='Predictions',
            text=[f"Actual: {a:.1f}, Predicted: {p:.1f}" for a, p in zip(plot_df['Actual'], plot_df['Predicted'])],
            hoverinfo='text'
        ))
        
        # Add perfect prediction line
        min_val = min(plot_df['Actual'].min(), plot_df['Predicted'].min())
        max_val = max(plot_df['Actual'].max(), plot_df['Predicted'].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='#ff6b6b', width=2, dash='dash'),
            name='Perfect Prediction'
        ))
        
        fig_scatter.update_layout(
            title="Actual vs. Predicted Scores",
            xaxis_title="Actual Score",
            yaxis_title="Predicted Score",
            template="plotly_dark",
            height=400,
            hovermode='closest'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with c_right:
        st.subheader("⭐ Top Score Influencers")
        top_all = get_top_features(10)
        
        fig_imp = go.Figure(data=[go.Bar(
            y=[f[0].replace('_',' ').title() for f in top_all], 
            x=[f[1] for f in top_all], 
            orientation='h', 
            marker_color='#6366f1',
            text=[f"{f[1]:.3f}" for f in top_all],
            textposition='auto'
        )])
        fig_imp.update_layout(
            title="Top 10 Features by Importance",
            xaxis_title="Importance Score",
            height=400, 
            template="plotly_dark", 
            yaxis=dict(autorange="reversed"),
            showlegend=False
        )
        st.plotly_chart(fig_imp, use_container_width=True)

with tab3:
    st.header("📚 Dataset Features")
    st.write("Model predicts the `final_score` (0-100) using behavioral and academic features.")
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Dataset Statistics")
    st.write(df.describe())

with tab4:
    st.header("ℹ️ About the Model")
    st.markdown("""
    ### Model Information
    - **Model Type**: 🌲 Random Forest Regressor
    - **Target Variable**: `final_score` (0-100 scale)
    - **Training/Testing Split**: 80% / 20%
    - **Number of Trees**: 200
    - **Max Depth**: 15
    - **Purpose**: Predict numerical academic performance outcomes
    
    ### How It Works
    1. The model analyzes 19 different student behavioral and academic features
    2. Uses ensemble learning with multiple decision trees
    3. Each tree learns different patterns in the data
    4. Final prediction is the average of all trees
    
    ### Key Metrics Explained
    - **R-Squared**: How well the model explains variance (0-1, higher is better)
    - **MAE**: Average absolute error in points
    - **RMSE**: Root mean squared error (penalizes larger errors)
    """)
