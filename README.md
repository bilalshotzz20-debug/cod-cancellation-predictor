# 📦 Pakistani E-Commerce COD Cancellation Predictor

An end-to-end Machine Learning project that predicts whether a Cash-on-Delivery (COD) order is likely to be cancelled. Built specifically for the Pakistani e-commerce market to help businesses reduce logistics and return costs.

## 🚀 Live Demo
**Try the app here:** [COD Cancellation Predictor](https://cod-cancellation-predictor-cavrnng5xkgjnaqsmgwvza.streamlit.app/)

## 🎯 The Business Problem
In Pakistan, Cash-on-Delivery (COD) is the most popular payment method. However, e-commerce businesses face massive losses due to order cancellations and refused deliveries.

**My Hypothesis (Domain Knowledge):** 
Through personal experience and market observation, I identified that **late delivery and poor logistics infrastructure** are the primary reasons customers cancel COD orders in Pakistan.

## 🧠 The Solution & Project Workflow
To solve this, I built a complete Machine Learning pipeline to predict cancellation risk before the order is shipped.

1. **Data Generation:** Created a synthetic dataset of 1,000 orders simulating real Pakistani e-commerce data (cities, order value, customer age, and delivery days).
2. **Feature Engineering:** Introduced a `delivery_days` feature to mathematically test my hypothesis.
3. **Exploratory Data Analysis (EDA):** Proved that orders taking **more than 6 days to deliver** have an **80% cancellation rate**, while orders delivered in 1-3 days rarely get cancelled.
4. **Model Building:** Trained a **Random Forest Classifier** using `Scikit-learn` to predict the `is_cancelled` target.
5. **Deployment:** Built an interactive web interface using **Streamlit** and deployed it to the cloud.

## 📊 Key Insights
* **Delivery Time matters:** Cancellation rate jumps from ~10% (1-3 days) to over 80% (7-10 days).
* **City Patterns:** Big cities like Karachi and Peshawar showed higher cancellation rates compared to Rawalpindi.
* **Model Accuracy:** Achieved **73% accuracy** on the test set. *(Note: The accuracy is capped because we simulated human randomness in the data—e.g., even with fast delivery, there is a 10% chance a customer cancels due to a change of mind).*

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Random Forest)
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment:** Streamlit Community Cloud
* **Model Saving:** Joblib

## 💻 How to Run This Project Locally
If you want to run this project on your own machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/bilalshotzz20-debug/cod-cancellation-predictor.git
