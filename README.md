# Clinical Trial Status Prediction Dashboard

## Overview

This project is a Machine Learning web app that predicts the status of a clinical trial using an XGBoost model and a Streamlit frontend.

It takes inputs like Sponsor, Start Date, Phase, Enrollment, and Condition, and predicts the trial status such as Completed, Recruiting, or Terminated.

---

## Features

* Real-time prediction using trained ML model
* Interactive Streamlit dashboard
* Visual graphs and analytics
* Phase-wise comparison
* Stores last 3 prediction history
* User-friendly output (no encoded values)

---

## Tech Stack

* Python
* Streamlit
* XGBoost
* Scikit-learn
* Pandas
* Plotly

---

## How to Run

Install dependencies:

```
pip install -r requirements.txt
```

Run app:

```
streamlit run app.py
```

---

## Output Example

Input: Phase 5, Enrollment 225
Output: Recruiting

---

## Project Info

Developed for Fusion Hackathon 24
