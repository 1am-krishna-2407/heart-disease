❤️ Heart Disease Prediction Dashboard

A Machine Learning powered healthcare prediction system that estimates the risk of heart disease based on patient medical attributes.

The project combines data preprocessing, machine learning modeling, and an interactive Streamlit dashboard to provide real-time risk predictions for cardiovascular disease.

This application demonstrates how AI can support early disease detection and preventive healthcare decision making.

🚀 Project Highlights

✔ Machine Learning based heart disease risk prediction
✔ Interactive dashboard built using Streamlit
✔ Real-time predictions using trained model
✔ Feature scaling and preprocessing pipeline
✔ Data visualization using Matplotlib and Seaborn
✔ Clean modular project structure for scalability

🧠 Machine Learning Pipeline

The system follows a standard end-to-end ML workflow.

1️⃣ Data Collection

The model is trained using a heart disease dataset containing clinical attributes of patients.

Typical attributes include:

Age

Blood pressure

Cholesterol

ECG results

Chest pain type

Maximum heart rate

Exercise induced angina

2️⃣ Data Preprocessing

Data preprocessing ensures the model receives normalized inputs.

Steps performed:

Feature selection

Numerical data normalization

Feature scaling using StandardScaler

Preparation of feature matrix and labels

3️⃣ Model Training

A supervised machine learning classification model is trained using Scikit-learn.

Training pipeline includes:

Train-test data split

Model fitting

Performance evaluation

Model serialization using Pickle

Saved components:

heart_model.pkl
scaler.pkl
4️⃣ Model Deployment

The trained model is deployed inside a Streamlit web application.

Users can:

Enter patient medical parameters

Submit the values

Get instant heart disease risk prediction

📊 Input Features

The model uses the following medical attributes:

Feature	Description
age	Age of the patient
sex	Gender (1 = male, 0 = female)
cp	Chest pain type
trestbps	Resting blood pressure
chol	Cholesterol level
fbs	Fasting blood sugar
restecg	Resting ECG results
thalach	Maximum heart rate achieved
exang	Exercise induced angina
oldpeak	ST depression induced by exercise
slope	Slope of peak exercise ST segment
ca	Number of major vessels
thal	Thalassemia status
📂 Project Structure
heart-disease-main
│
├── heart_app.py            # Streamlit dashboard
├── train_model.py          # ML model training script
├── heart_model.pkl         # Trained classification model
├── scaler.pkl              # Feature scaling object
├── feature_importance.csv  # Model feature importance
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
2️⃣ Create Virtual Environment
python -m venv venv

Activate environment

Linux / Mac

source venv/bin/activate

Windows

venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit server:

streamlit run heart_app.py

Open your browser and go to:

http://localhost:8501
📈 Example Workflow

1️⃣ User enters patient medical parameters
2️⃣ Data is scaled using the saved StandardScaler
3️⃣ Input is passed to the trained ML model
4️⃣ Model predicts heart disease risk probability
5️⃣ Dashboard displays prediction result and insights

🛠 Tech Stack

Programming Language

Python

Machine Learning

Scikit-learn

Pandas

NumPy

Visualization

Matplotlib

Seaborn

Deployment

Streamlit

Model Serialization

Pickle

💡 Potential Applications

This system can be expanded into real healthcare solutions such as:

Clinical decision support systems

Hospital triage risk assessment tools

Preventive healthcare screening systems

AI-powered hospital dashboards

Telemedicine diagnostic assistants

🔮 Future Improvements

Possible future enhancements include:

Deep learning models for improved prediction

Model explainability using SHAP or LIME

Integration with hospital electronic health records (EHR)

Cloud deployment (AWS / Streamlit Cloud / HuggingFace Spaces)

Batch prediction for multiple patients

REST API integration

📜 License

This project is developed for educational and research purposes.
