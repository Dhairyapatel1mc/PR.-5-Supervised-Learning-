<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24,30&height=220&section=header&text=🎓%20Smart%20Outcome%20Predictor&fontSize=46&fontColor=ffffff&fontAlignY=38&desc=Ensemble%20Learning%20System%20%7C%20Bagging%20·%20Boosting%20·%20Voting%20·%20Stacking&descAlignY=58&descSize=17&animation=fadeIn" width="100%"/>

<a href="https://git.io/typing-svg">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=20&pause=1000&color=00D4AA&center=true&vCenter=true&multiline=true&width=900&height=80&lines=🎯+Predicting+Course+Completion+%26+Final+Scores+with+Ensembles;📦+Bagging+%7C+⚡+AdaBoost+%7C+🌿+LightGBM+%7C+🚀+XGBoost;🗳️+Voting+%7C+📚+Stacking+%7C+Best+F1+%3D+0.7636" alt="Typing SVG"/>
</a>

<br/>

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/LightGBM-3.3-2ECC71?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/XGBoost-1.7-FF6600?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Pandas-2.0-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-1.26-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white"/>
</p>

<p>
  <img src="https://img.shields.io/badge/Status-✅%20Completed-2ECC71?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20F1-0.7636-FF6B6B?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Best%20Model-AdaBoost-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Students-1%2C200%20Records-00D4AA?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Tasks-Classification%20%2B%20Regression-orange?style=for-the-badge"/>
</p>

<br/>

<blockquote>
<i>🎓 A production-grade ensemble ML pipeline for an EdTech platform — predicting whether students will complete courses and what scores they'll achieve, using 8 ensemble strategies compared head-to-head.</i>
</blockquote>

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/GitHub-@Dhairyapatel1mc-181717?style=flat-square&logo=github&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/LinkedIn-ghost--patel-0077B5?style=flat-square&logo=linkedin&logoColor=white"/>
</a>
&nbsp;&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Instagram-@ghost__6927-E4405F?style=flat-square&logo=instagram&logoColor=white"/>
</a>

<br/><br/>

</div>

---

## 📋 Table of Contents

<div align="center">

| # | Section | Jump |
|:---:|---------|:---:|
| 01 | 🎯 Project Overview | [→](#-project-overview) |
| 02 | 📊 Dataset at a Glance | [→](#-dataset-at-a-glance) |
| 03 | 🗂️ Dataset Schema | [→](#️-dataset-schema) |
| 04 | 🧠 Theory — Ensemble Concepts | [→](#-theory--ensemble-concepts) |
| 05 | ⚙️ ML Pipeline Architecture | [→](#️-ml-pipeline-architecture) |
| 06 | 📦 Bagging Results | [→](#-part-c--bagging) |
| 07 | ⚡ Boosting Results | [→](#-part-d--boosting-algorithms) |
| 08 | 🗳️ Voting & Stacking | [→](#️-part-e--voting--stacking) |
| 09 | 🏆 Performance Leaderboard | [→](#-performance-leaderboard) |
| 10 | 📈 Visualizations | [→](#-visualizations--10-charts) |
| 11 | 📁 Project Structure | [→](#-project-structure) |
| 12 | 🚀 Quick Start | [→](#-quick-start) |
| 13 | 📋 Final Report | [→](#-final-report) |
| 14 | 👤 Author & Contact | [→](#-author--contact) |

</div>

---

## 🎯 Project Overview

<table>
<tr>
<td width="60%">

You are working as a **Machine Learning Engineer** at an EdTech analytics company. Single models are producing **unstable and inaccurate results** for predicting student outcomes. Your task is to design an **ensemble-based solution** that combines multiple learners.

**Two prediction tasks:**
- 🎯 **Classification:** Will the student complete the course? (`completion_status` 0/1)
- 📊 **Regression:** What final score will the student achieve? (`final_score` 0–100)

**What this pipeline delivers:**
- ✅ Implements 4 ensemble paradigms: Bagging, Boosting, Voting, Stacking
- ✅ Trains 4 boosting algorithms: AdaBoost, GradBoost, LightGBM, XGBoost
- ✅ Compares Hard vs Soft Voting strategies
- ✅ Stacks 4 base learners with a meta-learner
- ✅ Evaluates all models on both Clf and Reg tasks

</td>
<td width="40%" align="center">

| Field | Value |
|-------|-------|
| 🎓 Project | Smart Outcome Predictor |
| 🏢 Domain | EdTech / Education Analytics |
| 🎯 Task 1 | Binary Classification |
| 📊 Task 2 | Regression (0–100) |
| 🏆 Best Clf | **AdaBoost (F1=0.7636)** |
| 📈 Best Reg | **AdaBoost (R²=0.0167)** |
| 👥 Dataset | 1,200 students |
| ⏱️ Duration | 6 Hours |

</td>
</tr>
</table>

---

## 📊 Dataset at a Glance

<div align="center">

| 👥 Students | 🔢 Features | ✅ Completed | ❌ Not Completed | 📉 Avg Score |
|:---:|:---:|:---:|:---:|:---:|
| **1,200** | **14** | **679 (56.6%)** | **521 (43.4%)** | **53.8 / 100** |

</div>

```
COMPLETION STATUS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Completed    (1): ████████████████████████████████ 56.6% (679)
Not Completed(0): ████████████████████████         43.4% (521)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Train: 960 students  |  Test: 240 students  |  Split: 80/20 stratified
```

---

## 🗂️ Dataset Schema

| # | Feature | Type | Description | Signal |
|---|---------|------|-------------|--------|
| 1 | `age` | Int | Student age | — |
| 2 | `gender` | Cat | 0/1/2 encoded | — |
| 3 | `prior_education` | Ordinal | 0=None … 3=Post-Grad | ⭐ Top signal |
| 4 | `employment_status` | Cat | 0=Employed … 3=Student | — |
| 5 | `time_spent_hrs` | Float | Total hours on platform | ⭐ Top signal |
| 6 | `num_sessions` | Int | Login sessions | 📈 Higher = better |
| 7 | `quiz_attempts` | Int | Total quiz attempts | — |
| 8 | `videos_watched` | Int | Videos completed | 📈 Engagement |
| 9 | `forum_posts` | Int | Community participation | — |
| 10 | `assignment_submissions` | Int | Assignments submitted | ⭐ Top signal |
| 11 | `login_streak_days` | Int | Consecutive login days | 📈 Consistency |
| 12 | `quiz_avg_score` | Float | Avg quiz performance | ⭐ Top signal |
| 13 | `midterm_score` | Float | Midterm exam score | ⭐ Top signal |
| 14 | `assignment_score` | Float | Assignment avg score | ⭐ Top signal |
| 15 | `completion_status` | Binary | **0=Not Completed, 1=Completed** | 🎯 **TARGET (CLF)** |
| 16 | `final_score` | Float | **Final score 0–100** | 📊 **TARGET (REG)** |

---

## 🧠 Theory — Ensemble Concepts

<details>
<summary><b>🔷 Q1 — What are Ensemble Learning Methods and why are they effective?</b></summary>

<br>

**Ensemble Learning** combines predictions from multiple models to produce a stronger predictor than any single model alone.

```
WHY ENSEMBLES WORK:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Single model:       accuracy ≈ 67%  (one perspective)
Ensemble (diverse): accuracy ≈ 76%  (many perspectives)

Key principles:
  1. DIVERSITY — different models learn different patterns
  2. ERROR CANCELLATION — uncorrelated errors average out
  3. WISDOM OF CROWDS — many weak learners → one strong learner
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

</details>

<details>
<summary><b>🔷 Q2 — Bagging vs Boosting vs Voting</b></summary>

<br>

| | **Bagging** | **Boosting** | **Voting** |
|---|---|---|---|
| **Strategy** | Parallel, independent | Sequential, dependent | Parallel, independent |
| **Core idea** | Reduce variance via bootstrap | Reduce bias via error correction | Combine model diversity |
| **Sampling** | Bootstrap (with replacement) | Reweighted samples | No resampling |
| **Combination** | Average / majority vote | Weighted sum | Vote or average probability |
| **Example** | Random Forest | AdaBoost, XGBoost | VotingClassifier |
| **Best for** | High-variance models | High-bias models | Diverse model pool |

</details>

<details>
<summary><b>🔷 Q3 — Bias–Variance Trade-off & Ensembles</b></summary>

<br>

```
Total Error = Bias² + Variance + Irreducible Noise
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
High Bias   → Underfitting → BOOSTING fixes this
             Each model corrects previous model's errors
             → Systematic bias reduced iteratively

High Variance → Overfitting → BAGGING fixes this
              Multiple models on different data subsets
              → Individual errors cancel when averaged

ENSEMBLE MAGIC:
  Base DT alone:   Variance HIGH (one noisy split = bad result)
  100 Bagged DTs:  Variance LOW  (noise averages out across trees)
  AdaBoost:        Bias LOW      (hard examples get more attention)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

</details>

<details>
<summary><b>🔷 Q4 — Voting Classifier and Stacking Ensemble</b></summary>

<br>

**Hard vs Soft Voting:**

```
HARD VOTING:
  Model 1: Completed ✅
  Model 2: Completed ✅
  Model 3: Not Completed ❌
  Decision: Completed ✅ (2 vs 1 majority)

SOFT VOTING (uses probabilities — generally BETTER):
  Model 1: P(Complete) = 0.82
  Model 2: P(Complete) = 0.71
  Model 3: P(Complete) = 0.39
  Avg P(Complete) = 0.64 > 0.5 → Completed ✅
  (More nuanced — confidence counts)
```

**Stacking Architecture:**

```
LEVEL 0 — Base Learners (trained with 3-fold CV):
  [Decision Tree]  → pred₁
  [Random Forest]  → pred₂
  [LightGBM]       → pred₃
  [XGBoost]        → pred₄

LEVEL 1 — Meta-Learner (trained on Level 0 predictions):
  [Logistic Regression]
  Input:  [pred₁, pred₂, pred₃, pred₄]
  Output: Final class prediction

KEY: CV prevents data leakage from base to meta layer
```

</details>

<details>
<summary><b>🔷 Q5 — AdaBoost vs GradBoost vs LightGBM vs XGBoost</b></summary>

<br>

| Algorithm | Core Mechanism | Tree Growth | Speed | Regularization |
|-----------|---------------|------------|-------|----------------|
| **AdaBoost** | Reweight misclassified samples | Stumps (depth=1) | Moderate | None built-in |
| **GradBoost** | Fit negative gradient of loss | Level-wise | Slow | Depth, LR |
| **LightGBM** | Leaf-wise growth + GOSS sampling | Leaf-wise (deeper) | **Fastest** | L1/L2, dropout |
| **XGBoost** | Level-wise + explicit regularization | Level-wise | Fast | **Built-in L1+L2** |

```
KEY INSIGHT FOR THIS PROJECT:
  AdaBoost won (F1=0.7636) because:
  → Dataset has moderate complexity (14 features, 1200 rows)
  → Simple stumps + reweighting avoids overfitting
  → LightGBM/XGBoost overfit on this relatively small dataset
```

</details>

---

## ⚙️ ML Pipeline Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║               SMART OUTCOME PREDICTOR — ENSEMBLE PIPELINE               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  ║
║  │  CSV Dataset │ →  │  EDA &       │ →  │  Stratified Train/Test   │  ║
║  │ 1200 students│    │  Profiling   │    │  Split (80/20)           │  ║
║  └──────────────┘    └──────────────┘    └──────────┬───────────────┘  ║
║                                                       │                  ║
║                             ┌─────────────────────────▼──────────────┐  ║
║                             │         StandardScaler (optional)       │  ║
║                             └───────────┬────────────────────────────┘  ║
║                                         │                                ║
║  ┌──────────────┬──────────────┬────────┴──────┬─────────────────────┐  ║
║  │  📦 BAGGING  │ ⚡ BOOSTING  │  🗳️ VOTING   │  📚 STACKING       │  ║
║  │──────────────│──────────────│───────────────│─────────────────────│  ║
║  │BaggingClf    │AdaBoost      │ Hard Voting   │ DT + RF + LGBM      │  ║
║  │BaggingReg    │GradBoost     │ Soft Voting   │ + XGB → LogReg      │  ║
║  │(100 trees)   │LightGBM      │(3 base models)│ (meta-learner)      │  ║
║  │              │XGBoost       │               │                     │  ║
║  └──────┬───────┴──────┬───────┴───────┬───────┴──────────┬──────────┘  ║
║         │              │               │                   │             ║
║  ┌──────▼──────────────▼───────────────▼───────────────────▼──────────┐ ║
║  │   DUAL EVALUATION: Classification (F1/AUC) + Regression (R²/MAE)  │ ║
║  └────────────────────────────────────────────────────────────────────┘ ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 📦 Part C — Bagging

### Bootstrap Aggregating Explained

```
BAGGING PROCESS (100 estimators):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each of 100 trees:
  1. Sample 80% of training data WITH replacement (bootstrap)
  2. Train Decision Tree (max_depth=4) on this subset
  3. Store predictions

Final prediction:
  Classification → Majority vote of 100 trees
  Regression     → Average of 100 tree predictions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Bagging vs Single Base Model

| Metric | Single DT | Bagging (100) | Improvement |
|--------|:---------:|:-------------:|:-----------:|
| Accuracy | 0.6667 | 0.6958 | +0.0292 ↑ |
| F1-Score | 0.6774 | 0.7326 | +0.0552 ↑ |
| AUC-ROC  | 0.7280 | 0.7620 | +0.0340 ↑ |

> Bagging improves every metric — variance reduction in action. 100 trees with diverse bootstrap samples cancel out individual tree noise.

---

## ⚡ Part D — Boosting Algorithms

### AdaBoost: Sequential Weak Learner Strategy

```
ADABOOST ITERATION LOGIC:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Iteration 1:  Train stump → some errors
              → Increase weight of misclassified samples

Iteration 2:  Train stump with reweighted data
              → Focuses on previous errors
              → New errors identified

Iteration N:  Final model = Σ(αᵢ × stumpᵢ)
              where αᵢ = weight of each weak learner

Accuracy progression:
  iter=1   → 0.588
  iter=50  → 0.712
  iter=100 → 0.729 ← converged
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### All Boosting Algorithms Compared

| Algorithm | Clf Accuracy | Clf F1 | Clf AUC | Reg R² | Reg MAE |
|-----------|:-----------:|:------:|:-------:|:------:|:-------:|
| **AdaBoost** | **0.7292** | **0.7636** | **0.7874** | **0.0167** | **9.83** |
| GradBoost | 0.6833 | 0.7286 | 0.7313 | −0.0872 | 10.27 |
| LightGBM | 0.6792 | 0.7260 | 0.7217 | −0.1362 | 10.64 |
| XGBoost | 0.6792 | 0.7279 | 0.7304 | −0.1096 | 10.44 |

> **AdaBoost wins** on this dataset — the moderate size (1,200 rows) and feature complexity favour simple stumps over deep trees used in GradBoost/LightGBM/XGBoost.

### Learning Rate Effect (GradBoost)

```
LEARNING RATE vs ACCURACY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LR = 0.01  → Accuracy = 0.656  (too slow to learn)
LR = 0.05  → Accuracy = 0.679  (getting better)
LR = 0.08  → Accuracy = 0.683  ← chosen
LR = 0.10  → Accuracy = 0.679  (slight overfit)
LR = 0.20  → Accuracy = 0.667  (overfit)
LR = 0.50  → Accuracy = 0.650  (overfit badly)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Optimal: LR=0.08 with n_estimators=150
```

---

## 🗳️ Part E — Voting & Stacking

### Hard vs Soft Voting

| Metric | Hard Voting | Soft Voting | Better? |
|--------|:-----------:|:-----------:|:-------:|
| Accuracy | 0.6792 | 0.6667 | Hard ↑ |
| Precision | 0.7007 | 0.7000 | Tie |
| Recall | 0.7574 | 0.7206 | Hard ↑ |
| F1-Score | 0.7279 | 0.7101 | Hard ↑ |
| AUC-ROC | 0.7699 | 0.7427 | Hard ↑ |

> On this dataset Hard Voting outperforms Soft Voting — the base classifiers (DT, RF, LightGBM) have differently calibrated probabilities, making raw voting more reliable than probability averaging.

### Stacking Architecture

```
BASE LEARNERS (Level 0) — trained with 3-fold CV:
  Decision Tree   → pred₁ (out-of-fold predictions)
  Random Forest   → pred₂ (out-of-fold predictions)
  LightGBM        → pred₃ (out-of-fold predictions)
  XGBoost         → pred₄ (out-of-fold predictions)
         ↓               ↓               ↓              ↓
META-LEARNER (Level 1):
  Logistic Regression trained on [pred₁, pred₂, pred₃, pred₄]
  → Learns optimal weighting of base model outputs
  → Final F1 = 0.7376, AUC = 0.7578
```

---

## 🏆 Performance Leaderboard

### Classification — `completion_status`

<div align="center">

| Rank | 🏅 Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|:----:|----------|:--------:|:---------:|:------:|:--------:|:-------:|
| 🥇 | **AdaBoost** | **0.7292** | **0.7554** | **0.7721** | **0.7636** | **0.7874** |
| 🥈 | Bagging | 0.6958 | 0.7299 | 0.7353 | 0.7326 | 0.7620 |
| 🥉 | Stacking | 0.6917 | 0.7123 | 0.7647 | 0.7376 | 0.7578 |
| 4 | Hard Voting | 0.6792 | 0.7007 | 0.7574 | 0.7279 | 0.7699 |
| 5 | GradBoost | 0.6833 | 0.7083 | 0.7500 | 0.7286 | 0.7313 |
| 6 | XGBoost | 0.6792 | 0.7007 | 0.7574 | 0.7279 | 0.7304 |
| 7 | LightGBM | 0.6792 | 0.7034 | 0.7500 | 0.7260 | 0.7217 |
| 8 | Soft Voting | 0.6667 | 0.7000 | 0.7206 | 0.7101 | 0.7427 |
| 9 | Base DT | 0.6667 | 0.7500 | 0.6176 | 0.6774 | 0.7280 |

</div>

### F1-Score Visual

```
F1 SCORE LEADERBOARD:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
AdaBoost    ████████████████████████████████████ 0.7636 🏆
Bagging     █████████████████████████████████    0.7326
Stacking    ██████████████████████████████████   0.7376
Hard Voting █████████████████████████████████    0.7279
GradBoost   █████████████████████████████████    0.7286
XGBoost     █████████████████████████████████    0.7279
LightGBM    ████████████████████████████████     0.7260
Soft Voting ███████████████████████████████      0.7101
Base DT     ████████████████████████████         0.6774
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ensemble improvement over single DT: +0.0862 F1 points
```

### Regression — `final_score` (0–100)

| Rank | 🏅 Model | MAE | RMSE | R² Score |
|:----:|----------|:---:|:----:|:--------:|
| 🥇 | **AdaBoost** | **9.83** | **12.05** | **+0.0167** |
| 🥈 | Stacking | 9.90 | 12.16 | −0.0013 |
| 🥉 | Voting Reg | 10.06 | 12.17 | −0.0031 |
| 4 | Bagging | 10.21 | 12.49 | −0.0569 |
| 5 | Base DT | 10.25 | 12.54 | −0.0648 |
| 6 | GradBoost | 10.27 | 12.67 | −0.0872 |
| 7 | XGBoost | 10.44 | 12.80 | −0.1096 |
| 8 | LightGBM | 10.64 | 12.95 | −0.1362 |

> **Note:** Negative R² indicates models perform worse than predicting the mean — the `final_score` target has high noise relative to the features. AdaBoost is the only model achieving positive R², confirming its robustness.

---

## 📈 Visualizations — 10 Charts

| # | Chart File | Description | Key Insight |
|:---:|-----------|-------------|-------------|
| 01 | `01_eda_overview.png` | Class distribution pie + 7 feature histograms + score boxplot | 56.6% completion; score mean=53.8 |
| 02 | `02_correlation_heatmap.png` | Full feature correlation matrix (16 cols) | quiz/midterm/assign scores most correlated |
| 03 | `03_bagging.png` | Bagging vs Base DT (Clf + Reg) + n_estimators curve | Bagging: F1 +0.055 vs single DT |
| 04 | `04_boosting.png` | AdaBoost staged accuracy + LR effect + all boosting comparison | AdaBoost converges at iter≈80 |
| 05 | `05_lgbm_xgboost.png` | LightGBM & XGBoost feature importance + Regression comparison | quiz_avg_score = top feature for both |
| 06 | `06_voting_stacking.png` | Hard vs Soft Voting + Stacking architecture + comparison | Hard Voting > Soft Voting here |
| 07 | `07_roc_comparison.png` | ROC curves all 7 ensemble models + F1/AUC bar chart | AdaBoost: AUC=0.7874 (best) |
| 08 | `08_regression_comparison.png` | R², RMSE, Actual vs Predicted (all regressors) | AdaBoost: only positive R² |
| 09 | `09_final_classification.png` | Metrics heatmap + best model CM + F1 leaderboard + insights | Full diagnostic view |
| 10 | `10_final_report.png` | Reg heatmap + scatter + paradigm comparison + final summary | Production recommendations |

---

## 📁 Project Structure

```
🎓 smart-outcome-predictor/
│
├── 📓 Smart_Outcome_Predictor.ipynb        ← Pre-executed notebook (all outputs)
│
├── 📊 data/
│   ├── student_data.csv                     ← Full dataset (1,200 × 17)
│   └── final_predictions_ensemble.csv      ← ✅ All model predictions
│                                               (240 test rows × 27 columns)
│
├── 📈 charts/
│   ├── 01_eda_overview.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_bagging.png
│   ├── 04_boosting.png
│   ├── 05_lgbm_xgboost.png
│   ├── 06_voting_stacking.png
│   ├── 07_roc_comparison.png
│   ├── 08_regression_comparison.png
│   ├── 09_final_classification.png
│   └── 10_final_report.png
│
└── 📄 README.md                             ← This file
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Dhairyapatel1mc/smart-outcome-predictor.git
cd smart-outcome-predictor
```

### 2. Install Dependencies

```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn jupyter
```

### 3. Run the Notebook

```bash
jupyter notebook Smart_Outcome_Predictor.ipynb
```

### 4. Quick Inference

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor

# Train best classification model
ada_clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_completion_train)

# Train best regression model
ada_reg = AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=42)
ada_reg.fit(X_train, y_score_train)

# Predict for new students
completion_prob = ada_clf.predict_proba(X_new)[:, 1]
predicted_score = ada_reg.predict(X_new)

for i, (prob, score) in enumerate(zip(completion_prob, predicted_score)):
    status = "✅ Will Complete" if prob > 0.5 else "⚠️ At Risk"
    print(f"Student {i+1}: {status} (prob={prob:.2f}) | Predicted Score: {score:.1f}/100")
```

---

## 📋 Final Report

### Impact of Bagging vs Boosting

```
BAGGING (Parallel, Variance Reduction):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
+ Stable and robust (100 independent trees)
+ No overfitting risk (bootstrap sampling)
+ Scales well with n_estimators
- Cannot fix systematic bias (base model's assumptions)
- Regression R² = −0.057 (can't fix feature-target mismatch)
Result: F1 = 0.7326 (solid improvement over base)

BOOSTING (Sequential, Bias Reduction):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
+ Corrects previous model's mistakes iteratively
+ AdaBoost: simple stumps → avoids overfitting on small data
+ Best F1 = 0.7636 (project winner)
- GradBoost/LightGBM/XGBoost over-parameterized for 1,200 rows
- Sensitive to noise (noisy labels hurt performance)
```

### Advantages of Voting and Stacking

```
VOTING ADVANTAGES:
  ✅ Zero additional training cost beyond base models
  ✅ Reduces individual model variance via diversity
  ✅ Easily extensible (add/remove models)
  ⚠️ Hard Voting loses probability calibration information

STACKING ADVANTAGES:
  ✅ Meta-learner learns OPTIMAL combination weights
  ✅ Cross-validation prevents data leakage (cv=3)
  ✅ Can combine very different model types
  ⚠️ More complex, risk of meta-learner overfitting
  ⚠️ Slower training (base models trained 3+ times)
```

### Final Recommendations

| Task | Deploy Model | Why |
|------|-------------|-----|
| 🎯 Classification | **AdaBoost** | F1=0.7636, AUC=0.7874, robust to small dataset |
| 📊 Regression | **AdaBoost** | Only model with positive R² (0.0167) |
| 🏭 Production speed | **Soft Voting (LightGBM-based)** | Fast inference, good recall |
| 📚 Interpretability | **Bagging (DT base)** | Feature importance available |

### Business Interpretation

```
CLASSIFICATION INSIGHT:
  AdaBoost correctly identifies 77.2% of students who WILL complete
  → Early intervention for at-risk students (false negatives = 22.8%)
  → Recommend: flag students with P(completion) < 0.45 for counseling

REGRESSION INSIGHT:
  MAE ≈ 9.83 points on a 0–100 scale → ±10 point error margin
  → Useful for grade prediction bands (e.g., "likely 45–65 range")
  → Not precise enough for exact grade prediction alone

TOP PREDICTIVE SIGNALS (from LightGBM/XGBoost importance):
  1. quiz_avg_score      ← Academic performance signal
  2. midterm_score       ← Mid-course achievement
  3. assignment_score    ← Consistent work ethic
  4. time_spent_hrs      ← Engagement quantity
  5. login_streak_days   ← Consistency of engagement
```

---

## 👤 Author & Contact

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=soft&color=gradient&customColorList=6,11,20&height=120&section=header&text=Ghost%20(Patel%20Dhairya)&fontSize=30&fontColor=ffffff&desc=ML%20Engineer%20%7C%20Data%20Scientist%20%7C%20RWSkill&descSize=14&animation=fadeIn" width="70%"/>

<br/>

| | |
|:---:|---|
| 👤 **Name** | Ghost (Patel Dhairya) |
| 🏫 **Institute** | Red and White Skill Education (RWSkill) |
| 💼 **Role** | Machine Learning Engineer / Data Scientist |
| 🐙 **GitHub** | [@Dhairyapatel1mc](https://github.com/Dhairyapatel1mc) |
| 💼 **LinkedIn** | [ghost-patel](https://linkedin.com/in/ghost-patel) |
| 📸 **Instagram** | [@ghost_6927](https://instagram.com/ghost_6927) |

<br/>

<a href="https://github.com/Dhairyapatel1mc">
  <img src="https://img.shields.io/badge/Follow%20on%20GitHub-181717?style=for-the-badge&logo=github&logoColor=white"/>
</a>
&nbsp;
<a href="https://linkedin.com/in/ghost-patel">
  <img src="https://img.shields.io/badge/Connect%20on%20LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white"/>
</a>
&nbsp;
<a href="https://instagram.com/ghost_6927">
  <img src="https://img.shields.io/badge/Follow%20on%20Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white"/>
</a>

</div>

---

## 🛠️ Full Tech Stack

<div align="center">

| Category | Tools |
|----------|-------|
| **Language** | Python 3.11 |
| **Data** | Pandas, NumPy |
| **ML — Bagging** | Scikit-Learn BaggingClassifier, BaggingRegressor |
| **ML — Boosting** | AdaBoostClassifier/Regressor, GradientBoosting, LightGBM, XGBoost |
| **ML — Voting** | VotingClassifier, VotingRegressor |
| **ML — Stacking** | StackingClassifier (meta=LogReg), StackingRegressor (meta=Ridge) |
| **Visualization** | Matplotlib, Seaborn |
| **Notebook** | Jupyter (pre-executed, all outputs visible) |
| **Preprocessing** | StandardScaler, train_test_split (stratified) |

</div>

---

## 📜 License

```
MIT License — Copyright (c) 2024 Ghost (Patel Dhairya) · Red and White Skill Education

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software to use, copy, modify, merge, publish and distribute.
```

---

<div align="center">

**[⬆ Back to Top](#-smart-outcome-predictor)**

<br/>

```
⭐ If this project helped you, please star the repository!
```

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20,24,30&height=100&section=footer&animation=fadeIn" width="100%"/>

*Built with precision by Ghost (Patel Dhairya) · Red and White Skill Education (RWSkill) · 2024*

</div>
