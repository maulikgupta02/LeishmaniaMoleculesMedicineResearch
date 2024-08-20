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
from imblearn.over_sampling import SMOTE


def DecisionTree( x_train, x_test, y_train, y_test,tech):
    
    print("Decision Tree Classifier")
    print()
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0,shuffle=True)

    from sklearn.tree import DecisionTreeClassifier

    param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
                  'ccp_alpha': [0.1, 0.01, 0.001, 0.0001, 0],
                  'max_depth' : [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, None],
                  'min_samples_split': [2, 5, 10, 15, 20],
                  'min_samples_leaf': [1, 2, 4, 6, 8]
                 }
    
    tree_clas = DecisionTreeClassifier()
    classifier_DTC = GridSearchCV(estimator=tree_clas, param_grid=param_grid, cv=5,scoring='accuracy',refit=True)
    classifier_DTC.fit(x_train, y_train)
    y_pred = classifier_DTC.predict(x_test)
    from sklearn.metrics import confusion_matrix, accuracy_score
    # performace metric
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is :")
    print(cm)
    print(accuracy_score(y_test, y_pred))
    from sklearn.metrics import roc_curve
    
    #metrics
    pred_prob = classifier_DTC.predict_proba(x_test)
    dtc_acc_o = accuracy_score(y_test, y_pred)
    dtc_prc_o= precision_score(y_test, y_pred)
    dtc_f1_o = f1_score(y_test, y_pred)
    dtc_auc_o = roc_auc_score(y_test, y_pred)

    from sklearn.metrics import roc_curve
    pred_prob = classifier_DTC.predict_proba(x_test)
    fpr_dtc_o, tpr_dtc_o, thresh_dtc_o = roc_curve(y_test, pred_prob[:,1], pos_label=1)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr_dtc_o, tpr_dtc_o, linestyle='--',color='orange', label='DTC')
    plt.savefig(f"plots/{tech}/roc_curve_dt.png")
    print("Accuracy_Score:",accuracy_score(y_test, y_pred))
    print("Roc_Score:",roc_auc_score(y_test,y_pred))
    print()


def RandomForest( x_train, x_test, y_train, y_test,tech):
    
    print("Random Forest Classifier")
    print()

    from sklearn.ensemble import RandomForestClassifier

    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],
        'criterion': ['gini', 'entropy']
    }

    rf_classifier = RandomForestClassifier()

    classifier_RF = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy', refit=True)
    classifier_RF.fit(x_train, y_train)

    y_pred_rf = classifier_RF.predict(x_test)

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    print("Confusion Matrix is:")
    print(cm_rf)
    
    #metrics
    rf_acc_o = accuracy_score(y_test, y_pred_rf)
    rf_prc_o = precision_score(y_test, y_pred_rf)
    rf_f1_o = f1_score(y_test, y_pred_rf)
    rf_auc_o = roc_auc_score(y_test, y_pred_rf)

    fpr_rf_o, tpr_rf_o, thresh_rf_o = roc_curve(y_test, classifier_RF.predict_proba(x_test)[:,1])

    # Plot ROC curve
    plt.plot(fpr_rf_o, tpr_rf_o, linestyle='--', color='blue', label='Random Forest')
    plt.savefig(f"plots/{tech}/roc_curve_rf.png")
    print("Accuracy:", accuracy_score(y_test, y_pred_rf))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_rf))
    print()


def XGB( x_train, x_test, y_train, y_test,tech):
    
    print("XGBoost Classifier")
    print()
    
    warnings.filterwarnings("ignore", message="Starting in XGBoost", category=UserWarning)

    from xgboost import XGBClassifier

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    class_XGB = XGBClassifier() 
    classifier_XGB = GridSearchCV(class_XGB, param_grid, cv=5,scoring='accuracy',refit=True)
    classifier_XGB.fit(x_train, y_train)

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
    
    pred_prob = classifier_XGB.predict_proba(x_test)
    fpr_xgb_wc, tpr_xgb_wc, thresh_xgb_wc = roc_curve(y_test, pred_prob[:,1], pos_label=1)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr_xgb_wc, tpr_xgb_wc, linestyle='--',color='orange', label='XGB')
    plt.savefig(f"plots/{tech}/roc_curve_xgb.png")
    print("Accuracy_Score:",accuracy_score(y_test, y_pred))
    print("Roc_Score:",roc_auc_score(y_test,y_pred))
    print()


def MLP( x_train, x_test, y_train, y_test,tech):

    print("MLP Classifier")
    print()
    
    from sklearn.neural_network import MLPClassifier

    param_grid = {
        'hidden_layer_sizes': [(50,),(100,),(50,50),(100,100)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [1000, 2000, 3000]
    }
    class_MLP = MLPClassifier()
    classifier_MLP = GridSearchCV(class_MLP, param_grid, cv=5,scoring='accuracy',refit=True)
    classifier_MLP.fit(x_train, y_train)
    y_pred = classifier_MLP.predict(x_test)
    from sklearn.metrics import confusion_matrix, accuracy_score
    # performace metric
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix is :")
    print(cm)
    from sklearn.metrics import roc_curve
    pred_prob = classifier_MLP.predict_proba(x_test)
    
    mlp_acc_o = accuracy_score(y_test, y_pred)
    mlp_prc_o= precision_score(y_test, y_pred)
    mlp_f1_o = f1_score(y_test, y_pred)
    mlp_auc_o = roc_auc_score(y_test, y_pred)
    
    from sklearn.metrics import roc_curve
    pred_prob = classifier_MLP.predict_proba(x_test)
    fpr_mlp_o, tpr_mlp_o, thresh_mlp_o = roc_curve(y_test, pred_prob[:,1], pos_label=1)

    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    # plot roc curves
    plt.plot(fpr_mlp_o, tpr_mlp_o, linestyle='--',color='orange', label='MLP')
    plt.savefig(f"plots/{tech}/roc_curve_mlp.png")
    print("Accuracy_Score:",accuracy_score(y_test, y_pred))
    print("Roc_Score:",roc_auc_score(y_test,y_pred))
    print()


def Ensemble( x_train, x_test, y_train, y_test,tech):
    print("Ensembled Learing")
    print()

    from sklearn.ensemble import VotingClassifier
    from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score


    ensemble_model = VotingClassifier(
        estimators=[
            ('dtc', classifier_DTC),
            ('rf', classifier_RF),
            ('mlp', classifier_MLP),
            ('xgb', classifier_XGB)
        ],
        voting='soft' 
    )

    ensemble_model.fit(x_train, y_train)

    ensemble_pred = ensemble_model.predict(x_test)

    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    ensemble_prc = precision_score(y_test, ensemble_pred)
    ensemble_f1 = f1_score(y_test, ensemble_pred)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    
    from sklearn.metrics import roc_curve, roc_auc_score
    import matplotlib.pyplot as plt
    ensemble_probs = ensemble_model.predict_proba(x_test)[:, 1]
    fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, ensemble_probs)
    roc_auc_ensemble = roc_auc_score(y_test, ensemble_probs)

                
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr_ensemble, tpr_ensemble, color='darkorange', lw=2, label=f'Ensemble (AUC = {roc_auc_ensemble:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Ensemble Model')
    plt.legend(loc='lower right')
    plt.savefig(f"plots/{tech}/roc_curve_ensemble.png")
    plt.show()

    print()



def Results(tech):

    print("Combined Results")
    print()
    import matplotlib.pyplot as plt
    plt.style.use('seaborn')

    plt.plot(fpr_dtc_o, tpr_dtc_o, linestyle='--',color='green', label='DecisionTreeClassifier')
    plt.plot(fpr_rf_o, tpr_rf_o, linestyle='--',color='blue', label='RandomForestClassifier')
    plt.plot(fpr_mlp_o, tpr_mlp_o, linestyle='--',color='red', label='MultiLayerPerceptron')
    plt.plot(fpr_xgb_wc, tpr_xgb_wc, linestyle='--',color='yellow', label='GradientBoostingClassifier')
    plt.plot(fpr_ensemble, tpr_ensemble, linestyle='--', color='purple', label='EnsembleLearning')
    plt.legend()
    plt.savefig(f'plots/{tech}/roc_curve_combined.png')


    models = pd.DataFrame({
        'Model': [ 'Decision Tree Classifier', 'Random Forest Classifier','Multilayer Perceptron Classifier','XgBoost Classifier','Ensemble Learning'],
        'Accuracy': [dtc_acc_o, rf_acc_o,mlp_acc_o , xgb_acc_wo, ensemble_acc],
        'Precision': [dtc_prc_o, rf_prc_o,mlp_prc_o , xgb_prc_wo, ensemble_prc],
        'F1-Score': [dtc_f1_o, rf_f1_o,mlp_f1_o , xgb_f1_wo, ensemble_f1],
        'AUC': [dtc_auc_o, rf_auc_o,mlp_auc_o , xgb_auc_wo, ensemble_auc]
    })
    print(models)
    models.to_csv(f"metrics/{tech}.csv")
