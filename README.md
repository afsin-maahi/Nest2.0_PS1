# 🏥 Clinical Trial Data Quality Intelligence Platform

**AI-Powered Multi-Study Analytics for Novartis NEST 2.0 Competition**

## 🎯 Overview

This platform analyzes 16 clinical trials (6,237 patients) using advanced AI/ML to:
- Predict future data quality issues
- Prioritize urgent interventions
- Detect anomalies automatically
- Transfer best practices across studies
- Calculate ROI of improvements

## 🚀 Key Features

### 1. Smart Context-Aware DQI
- Adapts weights based on trial phase, therapeutic area, and timeline
- Study 4: 74.1 DQI vs Study 14: 91.5 DQI

### 2. Multi-Study Intelligence (UNIQUE!)
- Compares all 16 studies to identify best practices
- Quantified ROI: $220K by adopting Study 14's SAE workflow
- Specific recommendations with timelines

### 3. AI/ML Models
- **Risk Prediction**: 85.5% accuracy predicting future deterioration
- **Priority Scoring**: ±1.2 points accuracy for urgency ranking
- **Anomaly Detection**: 4.8% flagged (301 patients)

### 4. Interactive ROI Calculator
- Model impact of hiring CRAs, automation, increased visits
- Real-time DQI improvement projections
- Cost vs benefit analysis

## 📊 Key Findings

- **Study 4 Critical**: 49.1% high-risk patients, 6.4 avg SAE pending
- **763 patients** need immediate attention
- **17.4 point DQI gap** between best and worst studies
- **$220K ROI** possible in 4 weeks with recommended changes

## 🛠️ Installation

```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly prophet joblib
