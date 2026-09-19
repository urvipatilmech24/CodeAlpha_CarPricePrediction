# 🚗 Car Price Prediction Engine

An end-to-end Machine Learning web application that predicts used car market valuations based on vehicle features, built as part of the **CodeAlpha Data Science Internship (Task 3)**.

---

## 📌 Project Overview
Estimating the fair resale price of a used vehicle depends heavily on market factors, depreciation, mileage, and mechanical specifications. This project implements a **Random Forest Regressor** trained on historical car listings to deliver real-time vehicle appraisals through an interactive, glassmorphic web dashboard.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Processing & Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn
* **Data Visualization:** Matplotlib
* **Web UI / Framework:** Streamlit

---

## 📊 Dataset & Preprocessing
The dataset contains vehicle attributes including:
* **Present Price:** Showroom price of the vehicle (in Lakhs ₹).
* **Driven Kilometers:** Total distance accumulated.
* **Fuel Type:** Petrol / Diesel / CNG (One-Hot Encoded).
* **Seller Type:** Dealer / Individual (Encoded).
* **Transmission:** Manual / Automatic (Encoded).
* **Owner Count:** Prior owners.
* **Engineered Feature (`Car_Age`):** Derived from the registration year (`2024 - Year`) to quantify vehicle depreciation.

---

## 📈 Model Performance
The Random Forest Regressor was evaluated on an 80/20 train-test split:

| Metric | Score | Description |
| :--- | :--- | :--- |
| **$R^2$ Score** | **~95.9%** | Explains nearly 96% of variance in vehicle pricing |
| **Mean Absolute Error (MAE)** | **~0.64 Lakhs** | Average error deviation in lakhs |
| **Root Mean Squared Error (RMSE)**| **~0.97 Lakhs** | Standard deviation of prediction residuals |

### Key Feature Impact
* **Present Price:** Dominant pricing predictor (~88% feature weight).
* **Vehicle Age & Mileage:** Primary secondary drivers of value depreciation.

---

## 💻 Local Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/CodeAlpha_CarPricePrediction.git
   cd CodeAlpha_CarPricePrediction
