import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.isotonic import IsotonicRegression

# ==========================================
# 1. GENERATE SYNTHETIC FRAUD DATA
# ==========================================
np.random.seed(42)
n_samples = 100000

# 1% true fraud baseline
y_true = np.random.choice([0, 1], size=n_samples, p=[0.99, 0.01]) 

# Generate raw uncalibrated scores
raw_scores = np.random.beta(1, 9, size=n_samples)
raw_scores[y_true == 1] = np.random.beta(6, 4, size=np.sum(y_true))

# Simulate dynamic transaction amounts ($10 to $5000)
tx_amounts = np.random.exponential(scale=150, size=n_samples) + 10

# ==========================================
# 2. SMOOTHING THE SCORE (Probability Calibration)
# ==========================================
# Isotonic regression smooths out structural jumps in imbalanced PR curves
iso_reg = IsotonicRegression(out_of_bounds='clip')
calibrated_probs = iso_reg.fit_transform(raw_scores, y_true)

# ==========================================
# 3. COMPUTE RECALL @ PRECISION (Target: 80%)
# ==========================================
precisions, recalls, thresholds = precision_recall_curve(y_true, calibrated_probs)

# Monotonic Envelope Smoothing (removes jagged saw-tooth drops)
smoothed_precisions = np.maximum.accumulate(precisions[::-1])[::-1]

TARGET_PRECISION = 0.80
# Find indices where our smoothed curve satisfies business requirements
valid_indices = np.where(smoothed_precisions >= TARGET_PRECISION)[0]

if len(valid_indices) > 0:
    # Maximize recall within our constrained precision budget
    best_idx = valid_indices[np.argmax(recalls[valid_indices])]
    op_threshold = thresholds[min(best_idx, len(thresholds)-1)]
    op_precision = smoothed_precisions[best_idx]
    op_recall = recalls[best_idx]
else:
    raise ValueError("Model cannot achieve target precision. Retrain or loosen constraint.")

# ==========================================
# 4. COST-WEIGHTED UTILITY MATRIX
# ==========================================
# Assign operational values to each outcome matrix quadrant
COST_MANUAL_REVIEW = 5.00      # Cost of analyst triage time per alert
COST_USER_FRICTION = 15.00     # Cost of customer support/churn for false positive
PENALTY_MISSED_FRAUD = 1.5     # Cost factor for missed fraud (chargeback fee + loss multiplier)

def calculate_utility(y_true, probs, threshold, amounts):
    # Apply threshold to determine final alert strategy
    y_pred = (probs >= threshold).astype(int)
    
    # Define matrix evaluation masks
    tp = (y_pred == 1) & (y_true == 1)
    fp = (y_pred == 1) & (y_true == 0)
    fn = (y_pred == 0) & (y_true == 1)
    tn = (y_pred == 0) & (y_true == 0)
    
    # Compute operational impacts
    saved_fraud = np.sum(amounts[tp])
    lost_fraud = np.sum(amounts[fn] * PENALTY_MISSED_FRAUD)
    review_costs = np.sum(y_pred) * COST_MANUAL_REVIEW
    friction_costs = np.sum(fp) * COST_USER_FRICTION
    
    # Net economic footprint of our classification policy
    net_utility = saved_fraud - (lost_fraud + review_costs + friction_costs)
    
    return {
        "Alert Yield Count": int(np.sum(y_pred)),
        "Alert Yield %": float(np.sum(y_pred) / len(y_true) * 100),
        "Saved Fraud ($)": float(saved_fraud),
        "Financial Damage ($)": float(lost_fraud),
        "Operational Overhead ($)": float(review_costs + friction_costs),
        "Net Utility ($)": float(net_utility)
    }

metrics = calculate_utility(y_true, calibrated_probs, op_threshold, tx_amounts)

# ==========================================
# 5. DISPLAY PIPELINE METRICS
# ==========================================
print(f"--- Operational Performance Metrics ---")
print(f"Operational Probability Cutoff : {op_threshold:.4f}")
print(f"Target Precision Baseline      : >= {TARGET_PRECISION*100}%")
print(f"Actual Operational Precision   : {op_precision*100:.2f}%")
print(f"Recall @ Precision Target     : {op_recall*100:.2f}%\n")

print(f"--- Downstream Business Impact ---")
print(f"Total Alert Yield Volume       : {metrics['Alert Yield Count']} cases ({metrics['Alert Yield %']:.2f}% of txs)")
print(f"Prevented Fraud Volume         : ${metrics['Saved Fraud ($)']:,.2f}")
print(f"Missed Fraud Penalties (FN)    : ${metrics['Financial Damage ($)']:,.2f}")
print(f"Operational Costs (Review+FP)  : ${metrics['Operational Overhead ($)']:,.2f}")
print(f"Net Economic Utility           : ${metrics['Net Utility ($)']:,.2f}")
