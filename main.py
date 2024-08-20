def Results(x_test, y_test, tech):
    print("Combined Results")
    print()
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    classifier_DTC = load(f"models/{tech}/classifier_DTC.joblib")
    classifier_RF = load(f"models/{tech}/classifier_RF.joblib")
    classifier_MLP = load(f"models/{tech}/classifier_MLP.joblib")
    classifier_XGB = load(f"models/{tech}/classifier_XGB.joblib")
    classifier_ensemble = load(f"models/{tech}/classifier_ensemble.joblib")

    pred_prob = classifier_DTC.predict_proba(x_test)
    fpr_dtc_o, tpr_dtc_o, _ = roc_curve(y_test, pred_prob[:, 1], pos_label=1)

    fpr_rf_o, tpr_rf_o, _ = roc_curve(y_test, classifier_RF.predict_proba(x_test)[:, 1])

    pred_prob = classifier_MLP.predict_proba(x_test)
    fpr_mlp_o, tpr_mlp_o, _ = roc_curve(y_test, pred_prob[:, 1], pos_label=1)

    pred_prob = classifier_XGB.predict_proba(x_test)
    fpr_xgb_wc, tpr_xgb_wc, _ = roc_curve(y_test, pred_prob[:, 1], pos_label=1)
    
    ensemble_probs = classifier_ensemble.predict_proba(x_test)[:, 1]
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_probs)

    plt.figure(figsize=(8, 6))  # Adjust the figure size
    plt.plot(fpr_dtc_o, tpr_dtc_o, linestyle='-', color='green', label='Decision Tree')
    plt.plot(fpr_rf_o, tpr_rf_o, linestyle='-', color='blue', label='Random Forest')
    plt.plot(fpr_mlp_o, tpr_mlp_o, linestyle='-', color='red', label='Multi-Layer Perceptron')
    plt.plot(fpr_xgb_wc, tpr_xgb_wc, linestyle='-', color='orange', label='Gradient Boosting')
    plt.plot(fpr_ensemble, tpr_ensemble, linestyle='--', color='purple', label='Ensemble Learning')

    # Set scales and labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right',frameon=True, fancybox=True, edgecolor='black', facecolor='white')
    plt.grid(True)  # Add gridlines
    plt.xlim([0, 1])  # Set limit for x-axis
    plt.ylim([0, 1])  # Set limit for y-axis
    
        # Set border for the entire plot
    plt.gca().spines['top'].set_color('black')
    plt.gca().spines['right'].set_color('black')
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['left'].set_color('black')

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')

    # Set x and y axis scales to be the same
    plt.gca().set_aspect('equal', adjustable='box')

    # Save plot
    plt.savefig(f'plots/{tech}/roc_curve_combined.png',dpi=1200)
    plt.show()
import numpy as np 
import pandas as pd
from joblib import dump,load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import SMOTE
df = pd.read_csv("dataset/AID_1063_datatable_smiles_label.csv")
y=df["label"]

arr=np.load("dataset/avalon_fp.npy")
x=arr

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, shuffle=True)

tech = "avalon"

Results(x_test,y_test,tech)
import numpy as np 
import pandas as pd
from joblib import dump,load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import SMOTE
df = pd.read_csv('dataset/AID_1063_datatable_smiles_label.csv')
y=df["label"]

arr=np.load("dataset/updated_data_with_maccs.npy")
x=arr

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, shuffle=True)

tech = "maccs"

Results(x_test,y_test,tech)
import numpy as np 
import pandas as pd
from joblib import dump,load
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from imblearn.over_sampling import SMOTE
arr=np.load("dataset/target_pharmacore.npy",allow_pickle=True)
y=arr

arr=np.load("dataset/pharmacore.npy",allow_pickle=True)
x=arr

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, shuffle=True)

tech = "pharmacore"

Results(x_test,y_test,tech)
