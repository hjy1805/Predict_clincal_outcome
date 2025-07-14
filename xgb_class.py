import pandas as pd
import numpy as np
import optuna
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

df = pd.read_csv('/ibex/project/c2205/jiayi/Kp_ml/gffs/results/gene_presence_absence_roary.csv')
genome_data = df.iloc[:, 14:]  # All genome columns
genes = df['Gene']             # Gene names

#Convert to long format and then pivot to binary presence/absence
binary_matrix = genome_data.notnull().astype(int)
binary_matrix.insert(0, 'Gene', genes)
binary_matrix = binary_matrix.set_index('Gene').T

metadata = pd.read_csv('/ibex/project/c2205/jiayi/Kp_ml/Kp_clincal_metadata_18May.tsv', sep='\t')
meta_df = metadata[['strain', 'Age', 'Gender', 'LOS', 'Culture_type_Ass', 'Patient_Status']]
meta_df = meta_df.set_index('strain')

gender_map = {'F': 0, 'M': 1}
meta_df['Gender'] = meta_df['Gender'].map(gender_map)
status_map = {'Alive': 0, 'Died': 1}
meta_df['Death'] = meta_df['Patient_Status'].map(status_map)

meta_df = pd.get_dummies(meta_df, columns=['Culture_type_Ass'], prefix='', prefix_sep='')

meta_df[['Blood','Urine','Respiratory','Others']] = meta_df[['Blood','Urine','Respiratory','Others']].fillna(False).astype(int)
meta_df['Gender'] = meta_df['Gender'].fillna(False).astype(int)
meta_df = meta_df.dropna(subset=['Death'])
meta_df['Death'] = meta_df['Death'].astype(int)

merged_df = binary_matrix.merge(meta_df, left_index=True, right_index=True)
if 'Unnamed: 0' in merged_df.columns:
    merged_df = merged_df.drop(columns=['Unnamed: 0'])

#merged_df.to_csv("/ibex/project/c2205/jiayi/Kp_ml/ml_model/xgb_ready_dataset.csv")
X = merged_df.drop(columns=['Patient_Status','Death'])  # features
y = merged_df['Death']                 #label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
'''
def objective(trial):
    param = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'device': 'cuda',
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'subsample': trial.suggest_float('subsample', 0.4, 1.0),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'alpha': trial.suggest_float('alpha', 0.1, 1),         # L1 regularization (sparsity)
        'lambda': trial.suggest_float('lambda', 0, 10)        # L2 regularization (optional)
    }

    # Cross-validation on GPU
    cv_result = xgb.cv(
        param,
        dtrain,
        num_boost_round=100,
        nfold=3,
        early_stopping_rounds=10,
        seed=123,
        verbose_eval=False
    )

    return cv_result['test-logloss-mean'].min()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("Best trial:")
print(study.best_trial.params)

best_params = study.best_trial.params
best_params.update({
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'predictor': 'gpu_predictor',
    'device': 'cuda',
})

'''
best_params ={
    'colsample_bytree': 0.502876376299486, 
    'subsample': 0.41042032304496917, 
    'learning_rate': 0.09550020124819358, 
    'max_depth': 2, 
    'min_child_weight': 1, 
    'objective': 'binary:logistic',
    'tree_method': 'hist', 
    'device': 'cuda'
}

booster = xgb.train(
    best_params,
    dtrain,
    num_boost_round=100,
    evals=[(dtest, 'test')],
    early_stopping_rounds=10,
    verbose_eval=True
)

y_pred = booster.predict(dtest)
y_pred_binary = (y_pred > 0.5).astype(int)

#SHAP values calculation
explainer = shap.Explainer(booster, X_train)
shap_values = explainer(X_test)
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig("/ibex/project/c2205/jiayi/Kp_ml/ml_model/xgb_shap.svg", bbox_inches='tight')
plt.clf()
shap.plots.bar(shap_values, max_display=20,show=False)
plt.savefig("/ibex/project/c2205/jiayi/Kp_ml/ml_model/xgb_shap_bar.svg", bbox_inches='tight')
plt.clf()

# Bootstrapping for ROC curve with 95% CI
n_bootstraps = 1000
rng_seed = 42  # for reproducibility
rng = np.random.RandomState(rng_seed)

bootstrapped_tprs = []
bootstrapped_aucs = []
mean_fpr = np.linspace(0, 1, 100)

for i in range(n_bootstraps):
    # bootstrap by sampling with replacement
    indices = rng.randint(0, len(y_pred), len(y_pred))
    if len(np.unique(y_test.iloc[indices])) < 2:
        continue  # skip if only one class is present
    fpr, tpr, _ = roc_curve(y_test.iloc[indices], y_pred[indices])
    tpr_interp = np.interp(mean_fpr, fpr, tpr)
    tpr_interp[0] = 0.0
    bootstrapped_tprs.append(tpr_interp)
    score = roc_auc_score(y_test.iloc[indices], y_pred[indices])
    bootstrapped_aucs.append(score)

# Compute stats
bootstrapped_tprs = np.array(bootstrapped_tprs)
mean_tpr = np.mean(bootstrapped_tprs, axis=0)
std_tpr = np.std(bootstrapped_tprs, axis=0)
tpr_upper = np.minimum(mean_tpr + 1.96 * std_tpr, 1)
tpr_lower = np.maximum(mean_tpr - 1.96 * std_tpr, 0)

# Print AUC statistics
bootstrapped_aucs = np.array(bootstrapped_aucs)
auc_mean = np.mean(bootstrapped_aucs)
auc_ci_lower = np.percentile(bootstrapped_aucs, 2.5)
auc_ci_upper = np.percentile(bootstrapped_aucs, 97.5)

print(f"Test AUC = {auc_mean:.3f}")
print(f"95% CI = [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]")

# Plot
plt.figure(figsize=(8, 8))
plt.plot(mean_fpr, mean_tpr, label=f"AUC = {auc_mean:.3f} [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]", color='blue')
plt.fill_between(mean_fpr, tpr_lower, tpr_upper, color='blue', alpha=0.2, label="95% CI")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve with 95% CI (Bootstrapped)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("/ibex/project/c2205/jiayi/Kp_ml/ml_model/xgb_roc.svg", bbox_inches='tight')
plt.clf()

#save in csv
roc_ci_df = pd.DataFrame({
    'fpr': mean_fpr,
    'mean_tpr': mean_tpr,
    'tpr_lower': tpr_lower,
    'tpr_upper': tpr_upper
})
roc_ci_df.to_csv('/ibex/project/c2205/jiayi/Kp_ml/ml_model/xgb_roc_curve_data.csv', index=False)

# Save selected columns to TSV
with open("/ibex/project/c2205/jiayi/Kp_ml/ml_model/xgb_classifier_regularization_model_metrics.txt", "w") as f:
    f.write("Best parameters:" + str(best_params) + "\n")
    f.write("=== Model Evaluation Metrics ===\n")
    f.write("Accuracy: " + str(accuracy_score(y_test, y_pred_binary)) + "\n")
    f.write("F1 score: " + str(f1_score(y_test, y_pred_binary)) + "\n")
    f.write("AUC = {auc_mean:.3f} [{auc_ci_lower:.3f}, {auc_ci_upper:.3f}]" + "\n")
    f.write("=== Classification Report ===\n")
    f.write(str(classification_report(y_test, y_pred_binary)) + "\n")
    f.write("=== Confusion Matrix ===\n")
    f.write(str(confusion_matrix(y_test, y_pred_binary)) + "\n")
