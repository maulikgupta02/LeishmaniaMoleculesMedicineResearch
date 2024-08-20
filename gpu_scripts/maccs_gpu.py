import numpy as np 
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import warnings
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import std
warnings.filterwarnings("ignore")
from joblib import dump

def XGB( x_train, x_test, y_train, y_test,tech):
    
    print("XGBoost Classifier")
    print()
    
    warnings.filterwarnings("ignore")

    from xgboost import XGBClassifier

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    class_XGB = XGBClassifier(device="cuda", tree_method="hist")
    classifier_XGB = GridSearchCV(class_XGB, param_grid, cv=5,scoring='accuracy',refit=True)
    classifier_XGB.fit(x_train, y_train)
    
    dump(classifier_XGB, f"models/{tech}/classifier_XGB.joblib",compress=True)

    y_pred = classifier_XGB.predict(x_test)
    predictions = [round(value) for value in y_pred]
    from sklearn.metrics import confusion_matrix, accuracy_score
    # performace metric
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is :")
    print(cm)
    from sklearn.metrics import roc_curve
    
    #metrics
    pred_prob = classifier_XGB.predict_proba(x_test)
    xgb_acc_wo = accuracy_score(y_test, y_pred)
    xgb_prc_wo= precision_score(y_test, y_pred)
    xbg_f1_wo = f1_score(y_test, y_pred)
    xbg_auc_wo = roc_auc_score(y_test, y_pred)
    
    met_dic={"Accuracy":[xgb_acc_wo],'Precision': [xgb_prc_wo],'F1-Score': [xbg_f1_wo],'AUC': [xbg_auc_wo]}
    temp_df=pd.DataFrame(met_dic)
    temp_df.to_csv(f"metrics/{tech}/metrics_XGB.csv")
    
    pred_prob = classifier_XGB.predict_proba(x_test)
    fpr_xgb_wc, tpr_xgb_wc, thresh_xgb_wc = roc_curve(y_test, pred_prob[:,1], pos_label=1)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr_xgb_wc, tpr_xgb_wc, linestyle='--',color='orange', label='XGB')
    plt.savefig(f"plots/{tech}/roc_curve_xgb.png")
    plt.close()
    print("Accuracy_Score:",accuracy_score(y_test, y_pred))
    print("Roc_Score:",roc_auc_score(y_test,y_pred))
    print()

from imblearn.over_sampling import SMOTE
from joblib import dump

df = pd.read_csv('dataset/AID_1063_datatable_smiles_label.csv')
y=df["label"]

arr=np.load("dataset/updated_data_with_maccs.npy")
x=arr

smote = SMOTE(random_state=42)
x, y = smote.fit_resample(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1, shuffle=True)

tech = "maccs"

XGB( x_train, x_test, y_train, y_test,tech)