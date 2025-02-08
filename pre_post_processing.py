import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

def augmentQueries(common_queries):
    """
    Take a list of queries and augment them using a prespecified lexicon
    For example, if antineoplastic brand name (from GENIE BPC) this will add regimen names and brand names

    Parameters:
    - common_queries: list of strings

    Returns:
    - common_queries_full: list of strings (augmented to include other names)
    """
    chemotherapy_regimens = pd.read_csv('chemotherapy_regimens.csv')
    chemotherapy_brandnames = pd.read_csv('cancer_drug_mapping.csv')

    common_queries_full = []
    for c in common_queries:

        # regimens
        drugname_firstword = c.split(' ')[0].lower()
        alt_names = chemotherapy_regimens[chemotherapy_regimens['Components'].str.lower().str.contains(drugname_firstword)]['Name'].values
        if len(alt_names)>0:
            c_full = (c+' or '+' or '.join(alt_names))
        else:
            c_full = c

        # brand names
        alt_names = chemotherapy_brandnames[chemotherapy_brandnames['Generic Name'].str.lower().str.contains(c.lower())]['Brand Name'].values
        if len(alt_names)>0:
            c_full = (c_full+' or '+' or '.join(alt_names))
        else:
            c_full = c_full

        common_queries_full+=[c_full]

    return common_queries_full

def find_optimal_threshold_and_metrics(labels, probabilities):
    """
    Find the optimal threshold for a binary classification model based on the ROC curve.

    Parameters:
    - labels: List or array of true binary labels (0 or 1).
    - probabilities: List or array of predicted probabilities from the model.

    Returns:
    - optimal_threshold: The threshold that optimizes the balance between TPR and FPR.
    - metrics: A dictionary with counts of true positives, false positives, true negatives, and false negatives.
    """
    # Compute the ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)

    # Initialize variables to store the best threshold and F1 score
    max_f1 = 0
    optimal_threshold = 0

    # Iterate over thresholds to calculate precision, recall, and F1 score
    for i, threshold in enumerate(thresholds):
        # Convert probabilities to binary predictions using the threshold
        predictions = (np.array(probabilities) >= threshold).astype(int)

        # Calculate true positives, false positives, false negatives
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Update the optimal threshold if F1 score is higher
        if f1 > max_f1:
            max_f1 = f1
            optimal_threshold = threshold


    # Apply the threshold to calculate confusion matrix values
    predictions = (np.array(probabilities) >= optimal_threshold).astype(int)
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    true_negatives = np.sum((predictions == 0) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))

    metrics = {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives
    }

    return metrics
