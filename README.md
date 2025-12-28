
# 🎵 Spotify Churn Prediction

## 📌 Project Overview

This project builds a **predictive machine learning system** to identify Spotify users who are likely to **churn (stop using or disengage)** based on survey and behavioral preference data.

It now includes an **interactive UI built with Streamlit** (inside the `predictionui/` folder), allowing users to input custom data and view **churn probability** and **risk segments** dynamically.

The ML model predicts:

* **Churn probability**
* **Churn classification (Yes / No)**
* **Customer risk segments (Low / Medium / High)**

---

# Objectives

* Define churn using business logic
* Train a predictive classification model
* Build an interactive **Streamlit UI** for predictions
* Visualize insights and risk segments
* Save trained model and scaler for deployment

---

# Churn Definition Logic

A user is considered **likely to churn** if:

```text
premium_sub_willingness = "No"
AND
music_recc_rating ≤ 3
```

---

# Machine Learning Pipeline

1. Load & clean data (`data/Spotify_data.csv`)
2. Feature selection
3. Categorical encoding (One-Hot Encoding)
4. Train-test split
5. Feature scaling
6. Train **Logistic Regression** model (`train.py`)
7. Model evaluation (accuracy, ROC-AUC, confusion matrix)
8. Predict churn probability
9. Segment users by risk (Low / Medium / High)
10. Save model and scaler (`spotify_churn_model.pkl`, `spotify_scaler.pkl`)

---

# User Interface (Streamlit)

* Input sliders for Age and Recommendation rating
* Dropdowns for categorical features
* Dynamic prediction of **churn probability**
* Visual cards highlighting **risk segments**
* Real-time interactive UI with dark theme, glow effects, and interactive sliders

**UI Location:** `predictionui/app.py`

**UI Built With:**

* Streamlit
* HTML & CSS (for sliders, glow cards, buttons, dark theme)

---

##  Model Used

**Logistic Regression**

* Interpretable & fast
* Suitable for survey and categorical data
* Provides probability output for segmentation

---

## Visualizations

* Churn distribution (bar chart)
* Confusion matrix
* ROC curve
* Risk segments (Low / Medium / High)

---

## 🧩 Churn Risk Segments

| Probability Range | Segment     |
| ----------------- | ----------- |
| < 0.30            | Low Risk    |
| 0.30 – 0.70       | Medium Risk |
| > 0.70            | High Risk   |

---

## 🛠️ Libraries & Tools Used

### Python & ML

* Python 3.x
* `pandas`
* `numpy`
* `scikit-learn` (Logistic Regression)
* `joblib` (model persistence)

### Visualization

* `matplotlib` (charts, confusion matrix, ROC curve)

### UI / Frontend

* `streamlit` (interactive web app)
* Custom CSS & HTML (for sliders, glow cards, buttons, dark theme)

### Environment Management

* `uv` (virtual environment for Python dependencies)

---

## ⚙️ Environment Setup

### 1️⃣ Create virtual environment

```bash
uv venv
```

### 2️⃣ Activate environment

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

**Linux / Mac:**

```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
uv pip install pandas numpy matplotlib scikit-learn joblib streamlit
```

---

## ▶️ Run the Project

**Train the model (if not already trained):**

```bash
python train.py
```

**Run the Streamlit UI (from project root):**

```bash
streamlit run predictionui/app.py
```

---

## 📁 Project Structure

```
ClusterRecomendation/
│
├── data/
│   └── Spotify_data.csv          # Dataset for training the ML model
├── predictionui/                 # Streamlit UI folder
│   ├── app.py                    # Main Streamlit app
│   ├── assets/
│   │   └── style.css             # Custom CSS for UI
│   ├── spotify_churn_model.pkl   # Saved Logistic Regression model
│   ├── spotify_scaler.pkl        # Saved scaler
│   └── Spotify_data.csv          # Dataset copy for UI if needed
├── main.py                       # Optional main entry file
├── train.py                      # ML training script
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
└── pyproject.toml                # Project config
```

---

##  Outputs

* Model accuracy & classification report
* ROC-AUC & confusion matrix
* Churn probability for each user
* Risk-based segmentation
* Interactive UI predictions

---

##  Resume-Ready Description

> Built an end-to-end **Spotify churn prediction system** using Logistic Regression. Developed an interactive **Streamlit UI** inside the `predictionui/` folder with custom CSS for dynamic user input, visualization of churn probability, and risk segmentation. Saved models and scalers for reproducible deployment.

---

##  Future Improvements

* Implement advanced ML models (XGBoost / Random Forest)
* Deploy as a web app with live Spotify data
* Integrate with Power BI for dashboards
* Automate churn alerts and notifications
* Improve dataset with real-time user activity logs

---

 Author

Podmaraj Boruah
