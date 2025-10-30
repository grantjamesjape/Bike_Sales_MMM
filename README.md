# Bike_Sales_MMM
Project 1 of my portfolio. 

# 📈 Advanced Marketing Mix Modeling (MMM) for Bike Sales

## Project Overview

This project utilizes advanced econometric techniques to quantify the sales contribution and sensitivity of various marketing channels by building a robust Marketing Mix Model. The primary objective is to move beyond simple linear regression by applying **Geometric Adstock** and **Log Saturation** to provide actionable, time-aware budget allocation insights.

The final model achieved a strong fit ($\mathbf{R^2: 0.731}$) and delivered clear, data-driven recommendations.

---

## 1. Methodology & Diagnostics

### 1.1. Initial Diagnostics: Multicollinearity
The initial Exploratory Data Analysis (EDA) flagged a critical issue: perfect correlation ($r=1.00$) between **Branded Search Spend** and **Facebook Spend**. This forced a feature engineering decision to **combine** these two channels into a single feature, **`Search & Facebook`**, to ensure model stability and prevent estimation failure.

**Marketing Spend Correlation Matrix**
![Correlation Matrix of Marketing Spend Channels](charts/spend_correlation_matrix.png)

### 1.2. Feature Engineering (Advanced MMM)
To capture real-world marketing effects and improve model fidelity, all spend variables were transformed:
* **Geometric Adstock (Decay $\lambda=0.5$):** Models the carryover effect (memory) of advertising.
* **Log Saturation ($\ln(X+1)$):** Models the law of diminishing returns.

---

## 2. Final Model Performance and Fit

The final OLS model, run on the log-adstock-saturated features, yielded strong performance metrics:

| Metric | Result | Interpretation |
| :--- | :--- | :--- |
| **R-squared** | **$0.731$** | $73.1\%$ of the variance in sales is explained by marketing spend features. |
| **Condition Number** | **$461$** | Confirms high model stability (after fixing multicollinearity). |
| **P-Value (F-stat)** | **$0.000$** | The model is statistically significant. |

### Model Fit Visualization

The predicted sales line closely tracks the actual sales, validating the model's ability to capture weekly trends over the full period.

**Actual vs. Predicted Sales**
![Actual vs. Predicted Sales Plot](charts/actual_vs_predicted_sales.png)

---

## 3. Key Findings and Actionable Recommendations

The most crucial output is the **Sales Sensitivity Coefficient**, which dictates which channels drive the highest incremental sales lift.

### Final Channel Sensitivity Ranking

| Channel | Coefficient (Sales Lift) | P-Value | Significance |
| :--- | :--- | :--- | :--- |
| **Nonbranded Search** | **$27,397.20$** | $0.0000$ | **Significant** |
| **Search & Facebook** | **$10,891.06$** | $0.0000$ | **Significant** |
| Radio | $1,415.73$ | $0.0000$ | **Significant** |
| TV | $750.99$ | $0.0052$ | **Significant** |
| OOH | $-749.44$ | $0.0030$ | **Significant** |
| Print | $-918.53$ | $0.0000$ | **Significant** |

### Channel Sensitivity Visualization

**Sales Sensitivity (Coefficient) by Marketing Channel**
![Channel Sensitivity Bar Chart](charts/channel_sensitivity_bar_chart.png)

### Actionable Conclusions

1.  **Prioritize Digital:** **Nonbranded Search** and the combined **Search & Facebook** are the dominant, most efficient drivers of sales. Budget allocation should maximize investment in these areas first.
2.  **Audit Traditional Media:** **OOH** and **Print** exhibit statistically significant **negative** sales coefficients. This is a critical business finding requiring immediate investigation into campaign effectiveness or budget allocation strategy.

---

## 4. Automation and Reproducibility

### Repository Structure
