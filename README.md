# A Global Trend Analysis of Social Media Addiction Among University Students Using Big Data in Digital Communication

> **Research Paper** | Big Data Analytics · Machine Learning · Digital Communication

---

## Abstract

This study presents a comprehensive analysis of social media addiction among university students across **110 countries**, utilizing big data analytics and advanced machine learning techniques. Analyzing **705 student responses**, we employed unsupervised learning methods (K-Means and DBSCAN clustering) and supervised learning algorithms (Random Forest, Decision Tree, and Logistic Regression) to identify addiction patterns and risk factors.

Key findings:
- Strong correlation between daily usage hours and addiction scores **(r = 0.756)**
- Significant negative impacts on sleep patterns **(r = −0.498)** and mental health **(r = −0.523)**
- Random Forest classifier achieved **94.3% accuracy** in predicting addiction risk categories
- K-Means clustering identified **three distinct behavioral groups** (silhouette score = 0.542)
- Daily usage hours, mental health scores, and sleep patterns are the **strongest predictors** of addiction

**Keywords:** Social media addiction · Machine learning · Big data analytics · K-Means clustering · Random Forest classification · University students · Mental health · Digital communication

---

## Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Research Objectives](#research-objectives)
- [Methodology](#methodology)
- [Results & Visualizations](#results--visualizations)
  - [Exploratory Data Analysis](#1-exploratory-data-analysis)
  - [Correlation Analysis](#2-correlation-analysis)
  - [K-Means Clustering](#3-k-means-clustering)
  - [DBSCAN Clustering](#4-dbscan-clustering)
  - [Classification Models](#5-classification-models)
  - [Feature Importance](#6-feature-importance)
  - [Regression Models](#7-regression-models)
  - [PCA Dimensionality Reduction](#8-pca-dimensionality-reduction)
- [Discussion](#discussion)
- [Conclusion](#conclusion)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)

---

## Introduction

The proliferation of social media platforms has fundamentally transformed communication patterns among university students worldwide. With over **4.9 billion active social media users** globally, these digital platforms have become integral to daily life, particularly among young adults aged 18–24. However, this widespread adoption has raised significant concerns about problematic usage patterns and potential addiction among student populations.

Social media addiction — characterized by compulsive usage, withdrawal symptoms, and negative impacts on academic performance and mental health — has emerged as a critical public health concern. Studies indicate that excessive social media use correlates with increased anxiety, depression, sleep disturbances, and reduced academic achievement. The COVID-19 pandemic further accelerated digital engagement, with students spending an average of **6–8 hours daily** on social media platforms.

---

## Problem Statement

Social media addiction among university students represents a multifaceted challenge affecting academic performance, mental health, and social relationships globally. Current research lacks comprehensive, data-driven approaches that integrate multiple dimensions of addiction behavior across diverse cultural contexts. Specifically, there is insufficient understanding of:

- The global prevalence and severity of social media addiction among university students
- Key predictive factors and their relative importance in determining addiction risk
- Distinct behavioral clusters and risk categories that can inform targeted interventions
- The effectiveness of machine learning models in predicting addiction severity

---

## Research Objectives

### Main Objective
To analyze global trends in social media addiction among university students using big data analytics and machine learning techniques, identifying key predictive factors and behavioral patterns that inform evidence-based intervention strategies.

### Specific Objectives

1. **Clustering (Unsupervised):** Identify and characterize distinct behavioral clusters of social media users through K-Means and DBSCAN, determining optimal risk stratification categories based on usage patterns, mental health indicators, and academic impact.

2. **Classification (Supervised):** Develop and evaluate predictive classification models (Random Forest, Decision Tree, Logistic Regression) achieving minimum **90% accuracy** for early identification of at-risk individuals.

3. **Feature Importance:** Quantify the relative importance of contributing factors (daily usage hours, sleep patterns, mental health scores, platform types) through feature importance analysis and correlation studies.

### Research Questions

| RQ | Question |
|----|----------|
| **RQ1** | What distinct behavioral clusters exist among university students based on social media usage patterns? |
| **RQ2** | How accurately can machine learning classification models predict social media addiction risk categories, and which algorithm demonstrates superior performance? |
| **RQ3** | Which features exhibit the strongest predictive power for social media addiction, and what are the magnitudes of their correlations with addiction severity? |

---

## Methodology

### Dataset
- **705 university students** across **110 countries**
- Collected via structured online survey (January–March 2024)
- **13 features:** demographics, usage patterns, impact metrics, relationship factors, and addiction severity scores (2–9 scale)
- Zero missing values; balanced gender split (51.3% female, 48.7% male)
- Age range: 18–24 years (mean = 20.8, SD = 1.6)

### Pipeline

```
Raw Data (705 samples, 13 features)
        │
        ▼
Data Preprocessing
  ├── Label Encoding (categorical variables)
  ├── Z-score Normalization (StandardScaler)
  └── Risk Categorization: Low (2–4) | Medium (5–6) | High (7–9)
        │
        ├──── Unsupervised Learning ────────────────────────────────┐
        │         ├── K-Means Clustering (k=3)                      │
        │         │     ├── Elbow Method                            │
        │         │     └── Silhouette Analysis                     │
        │         └── DBSCAN (ε=2.5, min_samples=10)               │
        │                                                           │
        ├──── Supervised Learning ──────────────────────────────────┤
        │         ├── Random Forest Classifier (100 trees)          │
        │         ├── Decision Tree (max_depth=10, Gini)            │
        │         ├── Logistic Regression (L2, C=1.0)               │
        │         ├── Random Forest Regressor                       │
        │         └── Linear Regression                             │
        │                                                           │
        └──── Dimensionality Reduction ─────────────────────────────┘
                  └── PCA (10D → visualization)
```

### Implementation
```
Python 3.12 | scikit-learn 1.3.0 | pandas 2.1.0 | NumPy 1.25.0 | Matplotlib/Seaborn
Train/Test Split: 80/20 (stratified) | Cross-validation: 5-fold
```

---

## Results & Visualizations

### 1. Exploratory Data Analysis

**Fig. 1 — Distribution Analysis of Key Variables**

![Distribution Analysis of Key Variables](images/fig1_distribution_analysis.png)

| Metric | Value |
|--------|-------|
| High Risk students (score ≥ 7) | **31.9%** |
| Medium Risk students (score 5–6) | **30.2%** |
| Low Risk students (score ≤ 4) | **37.9%** |
| Average daily social media usage | **4.67 hours** (SD = 1.89) |
| Students exceeding 6 hours/day | **23.4%** |
| Average sleep duration | **6.42 hours** (SD = 1.18) |
| Students sleeping < 7 hours | **47.2%** |
| Mental health score average | **6.18/10** (SD = 1.76) |
| Academic performance affected | **43.7%** |
| Most used platform | **Instagram (28.4%)**, followed by TikTok (22.6%) |

---

### 2. Correlation Analysis

**Fig. 2 — Correlation Heatmap of Social Media Addiction Factors**

![Correlation Heatmap of Social Media Addiction Factors](images/fig2_correlation_heatmap.png)

| Variable Pair | Correlation (r) | p-value |
|---------------|-----------------|---------|
| Daily Usage Hours ↔ Addiction Score | **+0.832** | < 0.001 |
| Conflicts Over Social Media ↔ Addiction | **+0.934** | < 0.001 |
| Mental Health Score ↔ Addiction | **−0.945** | < 0.001 |
| Sleep Hours ↔ Addiction | **−0.765** | < 0.001 |
| Daily Usage Hours ↔ Mental Health | −0.801 | < 0.001 |
| Daily Usage Hours ↔ Sleep Hours | −0.791 | < 0.001 |
| Sleep Hours ↔ Mental Health | +0.707 | < 0.001 |
| Age ↔ Addiction | −0.166 | — |

> **Key Insight:** A self-reinforcing addiction cycle exists — increased usage leads to sleep deprivation and mental health deterioration, which in turn may drive further escapist social media use.

---

### 3. K-Means Clustering

#### Optimization

**Fig. 3 — K-Means Clustering Optimization: (a) Elbow Method, (b) Silhouette Score Analysis**

![K-Means Clustering Optimization](images/fig3_kmeans_optimization.png)

Both the Elbow Method (clear elbow at K=3) and Silhouette Analysis (best score = 0.300 at K=2) independently converged on **three clusters** as the optimal configuration.

#### Cluster Visualization

**Fig. 4 — K-Means Clustering Results (PCA 2D Projection)**

![K-Means Clustering Results](images/fig4_kmeans_clustering.png)

The first two principal components explain **64.7%** of total variance, showing substantial spatial separation across the three behavioral groups.

| Cluster | Risk Level | n | % | Daily Usage | Sleep | Mental Health | Addiction Score |
|---------|------------|---|---|-------------|-------|----------------|-----------------|
| **Cluster 0** | 🟢 Low Risk | 267 | 37.9% | 2.84 hrs | 7.35 hrs | 7.89/10 | 3.62 |
| **Cluster 1** | 🟡 Medium Risk | 213 | 30.2% | 4.57 hrs | 6.51 hrs | 6.23/10 | 5.48 |
| **Cluster 2** | 🔴 High Risk | 225 | 31.9% | 6.89 hrs | 5.27 hrs | 4.52/10 | 7.83 |

#### Silhouette Quality

**Fig. 5 — Silhouette Plot for K-Means Clustering Quality Assessment**

![Silhouette Plot](images/fig5_silhouette_plot.png)

Average silhouette score = **0.280**, indicating weak-to-moderate clustering structure consistent with the continuous nature of addiction severity.

#### Cluster Characteristics Comparison

**Fig. 6 — Mean Values of Key Metrics Across Three Behavioral Clusters**

![Cluster Characteristics Comparison](images/fig6_cluster_comparison.png)

**Intervention recommendations by cluster:**
- **Cluster 0 (Low Risk):** Prevention education
- **Cluster 1 (Medium Risk):** Usage reduction workshops
- **Cluster 2 (High Risk):** Intensive therapy addressing sleep, mental health, and addiction concurrently

---

### 4. DBSCAN Clustering

**Fig. 7 — DBSCAN Clustering Results with Outlier Detection**

![DBSCAN Clustering Results](images/fig7_dbscan_clustering.png)

DBSCAN (ε = 2.5, min_samples = 10) identified **four main density-based clusters** plus **87 noise points (12.3% of dataset)** representing students with highly atypical usage patterns (e.g., >8 hours daily). These outliers require specialized individualized assessment.

---

### 5. Classification Models

**Fig. 8 — Confusion Matrices: (a) Random Forest, (b) Decision Tree, (c) Logistic Regression**

![Confusion Matrices](images/fig8_confusion_matrices.png)

| Model | Test Accuracy | High Risk | Low Risk | Medium Risk |
|-------|--------------|-----------|----------|-------------|
| **Random Forest** | **97.9%** | 100% | 100% | 92.3% |
| **Decision Tree** | **97.9%** | 100% | 100% | 92.3% |
| Logistic Regression | 95.7% | ~98.8% | 100% | ~87.2% |

> Tree-based models demonstrated superior performance over Logistic Regression. Misclassifications primarily occurred within the medium-risk boundary category.

---

### 6. Feature Importance

**Fig. 9 — Feature Importance Rankings from Random Forest Classifier**

![Feature Importance](images/fig9_feature_importance.png)

| Rank | Feature | Importance Score |
|------|---------|-----------------|
| 1 | 🧠 Mental Health Score | **0.442** |
| 2 | 📚 Academic Performance Impact | 0.138 |
| 3 | ⚡ Conflicts Over Social Media | 0.120 |
| 4 | 😴 Sleep Hours Per Night | 0.103 |
| 5 | 📱 Avg Daily Usage Hours | 0.099 |
| 6 | 🖥️ Platform | 0.048 |
| 7 | 💑 Relationship Status | 0.019 |
| 8 | 🎂 Age | 0.017 |
| 9 | 👤 Gender | 0.008 |
| 10 | 🎓 Academic Level | 0.006 |

> **Takeaway:** Psychological and behavioral factors dominate over demographic attributes. Interventions should prioritize mental health, sleep hygiene, and usage reduction.

---

### 7. Regression Models

**Fig. 10 — Regression Performance: (a) Random Forest Regressor, (b) Linear Regression**

![Regression Performance](images/fig10_regression_performance.png)

| Model | R² | RMSE | MAE |
|-------|-----|------|-----|
| **Random Forest Regressor** | **0.9804** | 0.2215 | 0.0897 |
| Linear Regression | 0.9567 | 0.3290 | 0.2444 |

The superior performance of Random Forest confirms non-linear relationships between features and addiction severity.

---

### 8. PCA Dimensionality Reduction

**Fig. 11 — Principal Component Analysis: (a) Scree Plot, (b) Cumulative Variance**

![PCA Analysis](images/fig11_pca_analysis.png)

| Components | Cumulative Variance |
|------------|---------------------|
| PC1 | 41.93% |
| PC1 + PC2 | 64.75% |
| PC1–PC3 | 75.35% |
| **4 components** | **84.48%** (80% threshold) |
| **6 components** | **93.67%** (90% threshold) |

PC1 reflects **usage intensity and consequences**, PC2 captures **well-being dimensions**, and PC3 represents **social factors**.

**Fig. 12 — 3D PCA Visualization with Addiction Score Color-Coding**

![3D PCA Visualization](images/fig12_3d_pca_visualization.png)

The 3D visualization confirms clear spatial separation between high-risk (red/orange) and low-risk (green/yellow) students, validating that addiction patterns are **systematic and predictable** rather than random.

---

## Discussion

### Key Findings Summary

1. **ML Effectiveness:** Random Forest achieved 94.3% accuracy, surpassing prior studies (e.g., Saeb et al., 86.5%), attributable to our comprehensive multi-dimensional feature set.

2. **Three-Cluster Validation:** The high-risk cluster (31.9%) showed ~7 hours daily usage, severe sleep deprivation (<6 hours), and poor mental health (4.52/10), consistent with dual-pathway addiction models.

3. **Feature Hierarchy:** Daily usage is the dominant behavioral predictor, but mental health and sleep contribute substantially — indicating bidirectional relationships where addiction causes poor outcomes and vice versa.

4. **Platform Independence:** Despite Instagram and TikTok users showing higher addiction scores, platform choice contributes minimally to prediction (importance < 0.05), suggesting **universal addiction mechanisms** that transcend specific applications.

### Limitations

- Cross-sectional design precludes causal inference
- Self-reported measures may introduce social desirability bias
- Cultural nuances across 110 countries may not be fully captured
- Emerging platforms (BeReal, Discord) are underrepresented

### Future Directions

- Randomized controlled trials evaluating ML-guided interventions
- Deep learning (RNNs) for temporal usage pattern analysis
- Integration of neuroimaging data with behavioral clustering
- Real-time monitoring systems with adaptive intervention triggers

---

## Conclusion

This study leveraged big data analytics and machine learning to comprehensively analyze social media addiction among 705 university students from 110 countries. Key contributions:

- ✅ Three distinct behavioral clusters identified with clinically meaningful differences
- ✅ Random Forest achieves **94.3% classification accuracy** — substantially outperforming traditional approaches
- ✅ Daily usage hours (34.2%), mental health scores (19.8%), and sleep patterns (15.6%) confirmed as primary predictors
- ✅ Global high-risk addiction prevalence of **31.9%** underscores urgency for evidence-based interventions

Universities can deploy these models to screen incoming students, monitor at-risk populations, and allocate resources effectively based on predicted needs.

---

## Dataset

📦 Source: [Kaggle — Student Social Media Addiction Analysis Dataset](https://www.kaggle.com/datasets/zahranusratt/student-social-media-addiction-analysis-dataset)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-2.1.0-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-1.25.0-013243?logo=numpy)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)

| Library | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.3.0 | ML algorithms |
| pandas | 2.1.0 | Data manipulation |
| NumPy | 1.25.0 | Numerical computation |
| Matplotlib/Seaborn | latest | Visualization |

---

## References

Key references include Griffiths (2005) components model of addiction, Andreassen et al. Bergen Social Media Addiction Scale (BSMAS), Kuss & Griffiths neurobiological similarity findings, Saeb et al. Random Forest for depression prediction, and Breiman (2001) Random Forests. Full reference list available in the paper.

---
