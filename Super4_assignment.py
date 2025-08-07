import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')



# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="Page Configuration",
    layout="wide"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page:", [
    "Data Import and Overview",
    "Data Preprocessing", 
    "Model Training",
    "Model Evaluation",
    "Prediction Page",
    "Interpretation and Conclusions"
])

# Helper functions
def load_sample_data():
    """Load sample telco customer churn data"""
    # This would typically load from your CSV file
    # For demo purposes, creating sample data structure
    try:
        # Try to load the actual file if available
        data = pd.read_csv('C:/Users/User/Downloads/Telco-Customer-Churn.csv.csv')
        return data
    except:
        st.error("Sample data file not found. Please upload your own dataset.")
        return None

def preprocess_data(dataset):
    """Preprocess the data for modeling"""
    dataset_processed = dataset.copy()
    
    # Handle TotalCharges column (convert to numeric)
    dataset_processed['TotalCharges'] = pd.to_numeric(dataset_processed['TotalCharges'], errors='coerce')
    dataset_processed['TotalCharges'].fillna(dataset_processed['TotalCharges'].median(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                          'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod', 'Churn']
    
    for col in categorical_columns:
        if col in dataset_processed.columns:
            dataset_processed[col] = le.fit_transform(dataset_processed[col])
    
    return dataset_processed

# PAGE 1: Data Overview
if page == "Data Overview":
    st.title("Customer Churn Prediction - Data Overview")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your customer churn dataset (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.success("Data uploaded successfully!")
    else:
        # Load sample data button
        if st.button("Load Sample Telco Dataset"):
            st.session_state.data = load_sample_data()
    
    if st.session_state.data is not None:
        dataset = st.session_state.data
        
        # Basic dataset info
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", len(dataset))
        with col2:
            st.metric("Features", len(dataset.columns)-1)
        with col3:
            churned = dataset['Churn'].value_counts().get('Yes', 0)
            st.metric("Churned Customers", churned)
        with col4:
            churn_rate = (churned / len(dataset)) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(dataset.head())
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.dataframe(dataset.describe())
        
        # Visualizations
        st.subheader("Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            fig, ax = plt.subplots(figsize=(8, 6))
            churn_counts = dataset['Churn'].value_counts()
            ax.bar(churn_counts.index, churn_counts.values, color=['skyblue', 'salmon'])
            ax.set_title('Churn Distribution')
            ax.set_xlabel('Churn')
            ax.set_ylabel('Count')
            st.pyplot(fig)
        
        with col2:
            # Tenure histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(dataset['tenure'], bins=30, color='lightgreen', alpha=0.7)
            ax.set_title('Customer Tenure Distribution')
            ax.set_xlabel('Tenure (months)')
            ax.set_ylabel('Frequency')
            st.pyplot(fig)
        
        # Correlation matrix for numerical features
        st.subheader("Correlation Matrix")
        numerical_cols = dataset.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = dataset[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix of Numerical Features')
            st.pyplot(fig)

# PAGE 2: Data Preprocessing
elif page == "Data Preprocessing":
    st.title("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload dataset in the 'Data Overview' page.")
    else:
        dataset = st.session_state.data
        
        st.subheader("Raw Data")
        st.write(f"Shape: {dataset.shape}")
        st.dataframe(dataset.head())
        
        # Check for missing values
        st.subheader("Missing Values Analysis")
        missing_values = dataset.isnull().sum()
        if missing_values.sum() > 0:
            st.write("Missing values found:")
            st.dataframe(missing_values[missing_values > 0])
        else:
            st.success("No missing values!")
        
        # Data type information
        st.subheader("Data Types")
        st.write(dataset.dtypes)
        
        # Preprocessing button
        if st.button ("Preprocessing Application"):
            with st.spinner("Processing data..."):
                st.session_state.processed_data = preprocess_data(dataset)
                st.success("Data preprocessing completed!")
        
        # Show processed data
        if st.session_state.processed_data is not None:
            processed_df = st.session_state.processed_data
            
            st.subheader("Processed Data")
            st.write(f"Shape: {processed_df.shape}")
            st.dataframe(processed_df.head())
            
            st.subheader("Preprocessing Steps Applied:")
            st.write("Converted TotalCharges to numeric")
            st.write("Filled missing values with median")
            st.write("Applied Label Encoding to categorical variables")
            st.write("Data is ready for modeling")

# PAGE 3: Model Training
elif page == "Model Training":
    st.title("🤖Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("Please preprocess the data!.")
    else:
        dataset = st.session_state.processed_data
        # Prepare features and target
        X = dataset.drop(['customerID', 'Churn'], axis=1, errors='ignore')
        y = dataset['Churn']
        
        # Train-test split
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.write(f"Training set size: {X_train.shape[0]}")
        st.write(f"Test set size: {X_test.shape[0]}")
        
        # Model training
        if st.button("Model Training"):
            with st.spinner("Training models..."):
                
                # Logistic Regression
                lr_model = LogisticRegression(random_state=random_state, max_iter=1000)
                lr_model.fit(X_train_scaled, y_train)
                
                # Decision Tree
                dt_model = DecisionTreeClassifier(random_state=random_state)
                dt_model.fit(X_train, y_train)
                
                # Store models and data
                st.session_state.models = {
                    'logistic_regression': lr_model,
                    'decision_tree': dt_model,
                    'scaler': scaler,
                    'X_train': X_train,
                    'X_test': X_test,
                    'y_train': y_train,
                    'y_test': y_test,
                    'X_train_scaled': X_train_scaled,
                    'X_test_scaled': X_test_scaled,
                    'feature_names': X.columns.tolist()
                }
                
                st.success("Your Model Has Been Successfully Trained!")
        
        # Display model information
        if st.session_state.models:
            st.subheader("Train Models")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Logistic Regression**")
                lr_model = st.session_state.models['logistic_regression']
                st.write(f"Solver: {lr_model.solver}")
                st.write(f"Max Iterations: {lr_model.max_iter}")
                st.write(f"Features: {len(st.session_state.models['feature_names'])}")
            
            with col2:
                st.write("**Decision Tree Classifier**")
                dt_model = st.session_state.models['decision_tree']
                st.write(f"Max Depth: {dt_model.max_depth}")
                st.write(f"Min Samples Split: {dt_model.min_samples_split}")
                st.write(f"Features: {len(st.session_state.models['feature_names'])}")

# PAGE 4: Model Evaluation
elif page == "Model Evaluation":
    st.title("🦾 Model Evaluation")
    
    if not st.session_state.models:
        st.warning("You need to train the model!")
    else:
        models = st.session_state.models
        
        # Make predictions
        lr_pred = models['logistic_regression'].predict(models['X_test_scaled'])
        dt_pred = models['decision_tree'].predict(models['X_test'])
        
        # Calculate metrics
        def calculate_metrics(y_true, y_pred, model_name):
            return {
                'Model': model_name,
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred),
                'Recall': recall_score(y_true, y_pred),
                'F1-Score': f1_score(y_true, y_pred)
            }
        
        lr_metrics = calculate_metrics(models['y_test'], lr_pred, 'Logistic Regression')
        dt_metrics = calculate_metrics(models['y_test'], dt_pred, 'Decision Tree')
        
        # Comparing Model Performance
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame([lr_metrics, dt_metrics])
        st.dataframe(metrics_df.set_index('Model'))
        
        # Visualize metrics
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        lr_values = [lr_metrics[metric] for metric in metrics_to_plot]
        dt_values = [dt_metrics[metric] for metric in metrics_to_plot]
        
        ax.bar(x - width/2, lr_values, width, label='Logistic Regression', alpha=0.8)
        ax.bar(x + width/2, dt_values, width, label='Decision Tree', alpha=0.8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot)
        ax.legend()
        ax.set_ylim(0, 1)
        
        for i, v in enumerate(lr_values):
            ax.text(i - width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        for i, v in enumerate(dt_values):
            ax.text(i + width/2, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        st.pyplot(fig)
        
        # Confusion matrices
        st.subheader("Confusion Matrices")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Logistic Regression**")
            lr_cm = confusion_matrix(models['y_test'], lr_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Logistic Regression Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        with col2:
            st.write("**Decision Tree**")
            dt_cm = confusion_matrix(models['y_test'], dt_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title('Decision Tree Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
        
        # Store results
        st.session_state.model_results = {
            'lr_metrics': lr_metrics,
            'dt_metrics': dt_metrics,
            'lr_pred': lr_pred,
            'dt_pred': dt_pred
        }

# PAGE 5: Prediction Page
elif page == "Prediction Page":
    st.title("🐝 Super4 Churn Prediction")
    
    if not st.session_state.models:
        st.warning("You need to train the model!")
    else:
        st.subheader("Enter Customer Data")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 1, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
        with col2:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges", 18.0, 120.0, 50.0)
            total_charges = st.number_input("Total Charges", 18.0, 9000.0, 1000.0)
        
        if st.button("Predict Churn"):
            # Prepare input data
            input_data = {
                'gender': 1 if gender == "Male" else 0,
                'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0,
                'Dependents': 1 if dependents == "Yes" else 0,
                'tenure': tenure,
                'PhoneService': 1 if phone_service == "Yes" else 0,
                'MultipleLines': 2 if multiple_lines == "Yes" else (1 if multiple_lines == "No phone service" else 0),
                'InternetService': 1 if internet_service == "Fiber optic" else (0 if internet_service == "DSL" else 2),
                'OnlineSecurity': 1 if online_security == "Yes" else (2 if online_security == "No internet service" else 0),
                'OnlineBackup': 1 if online_backup == "Yes" else (2 if online_backup == "No internet service" else 0),
                'DeviceProtection': 1 if device_protection == "Yes" else (2 if device_protection == "No internet service" else 0),
                'TechSupport': 1 if tech_support == "Yes" else (2 if tech_support == "No internet service" else 0),
                'StreamingTV': 1 if streaming_tv == "Yes" else (2 if streaming_tv == "No internet service" else 0),
                'StreamingMovies': 1 if streaming_movies == "Yes" else (2 if streaming_movies == "No internet service" else 0),
                'Contract': 1 if contract == "One year" else (2 if contract == "Two year" else 0),
                'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
                'PaymentMethod': 3 if payment_method == "Electronic check" else (2 if payment_method == "Mailed check" else (1 if payment_method == "Credit card (automatic)" else 0)),
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Scale input for logistic regression
            input_scaled = st.session_state.models['scaler'].transform(input_df)
            
            # Make predictions
            lr_pred = st.session_state.models['logistic_regression'].predict(input_scaled)[0]
            lr_prob = st.session_state.models['logistic_regression'].predict_proba(input_scaled)[0]
            
            dt_pred = st.session_state.models['decision_tree'].predict(input_df)[0]
            dt_prob = st.session_state.models['decision_tree'].predict_proba(input_df)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Logistic Regression**")
                if lr_pred == 1:
                    st.error("‼️ Halleluyah, Customer is likely to CHURN")
                else:
                    st.success("😃Good news!! Customer will STAY")
                st.write(f"Churn Probability: {lr_prob[1]:.2%}")
                st.write(f"Stay Probability: {lr_prob[0]:.2%}")
            
            with col2:
                st.write("**Decision Tree**")
                if dt_pred == 1:
                    st.error("‼️ Halleluyah, Customer is likely to CHURN")
                else:
                    st.success("😃Good news!! Customer will STAY")
                st.write(f"Churn Probability: {dt_prob[1]:.2%}")
                st.write(f"Stay Probability: {dt_prob[0]:.2%}")

# PAGE 6: Interpretation and Conclusions
elif page == "Interpretation of analysis and Conclusions":
    st.title("🧩 Interpretation and Conclusions")
    
    if not st.session_state.models or not st.session_state.model_results:
        st.warning("You need to train and evaluate the model")
    else:
        models = st.session_state.models
        results = st.session_state.model_results
        
        # Model performance summary
        st.subheader("Model Performance Summary")
        
        metrics_df = pd.DataFrame([results['lr_metrics'], results['dt_metrics']])
        st.dataframe(metrics_df.set_index('Model'))
        
        # Feature importance (Decision Tree)
        st.subheader(" Analysis Of Feature Importance")
        
        dt_model = models['decision_tree']
        feature_importance = pd.DataFrame({
            'Feature': models['feature_names'],
            'Importance': dt_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(10)
        ax.barh(range(len(top_features)), top_features['Importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Most Important Features (Decision Tree)')
        ax.invert_yaxis()
        
        for i, v in enumerate(top_features['Importance']):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center')
        
        st.pyplot(fig)
        
        # Key insights
        st.subheader("Key Insights")
        
        st.write("**Most Predictive Features:**")
        for i, (_, row) in enumerate(top_features.head(5).iterrows(), 1):
            st.write(f"{i}. **{row['Feature']}** - Importance: {row['Importance']:.3f}")
        
        st.subheader("Model Comparison and Trade-offs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Logistic Regression**")
            st.write("✅Estimates probability ")
            st.write("✅Overfitting Is Reduced")
            st.write("✅ Accurate prediction and training")
            st.write(f"Accuracy: {results['lr_metrics']['Accuracy']:.3f}")
        
        with col2:
            st.write("**Decision Tree**")
            st.write("Easy And Simple To Interprate")
            st.write("Handles non-linear relationships")
            st.write("Feature scaling may not be needed")
            st.write(f"Accuracy: {results['dt_metrics']['Accuracy']:.3f}")
        
        # Recommendations
        st.subheader("Business Recommendations")
        
        st.write("**Customer Retention Strategies:**")
        st.write("1. **Focus on high-risk segments** identified by the models")
        st.write("2. **Improve customer service** for customers with short tenure")
        st.write("3. **Offer incentives** for long-term contracts")
        st.write("4. **Enhance digital services** to reduce churn")
        st.write("5. **Monitor monthly charges** relative to customer value")
        
        # Model selection recommendation
        better_model = "Logistic Regression" if results['lr_metrics']['F1-Score'] > results['dt_metrics']['F1-Score'] else "Decision Tree"
        st.subheader("Recommendation For Model Selection")
        st.info(f"**Recommended Model: {better_model}** based on F1-Score performance")
        
        if better_model == "Logistic Regression":
            st.write("Logistic Regression is recommended because it provides better analysis and is more robust for deployment..")
        else:
            st.write("Decision Tree is recommended because it is simple to interpret and handles complex relationships between variables better.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Customer Churn Prediction App - Built with Streamlit")
