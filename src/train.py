#============================================================================================================
#================================ lib =======================================================================
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc ,confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
#============================================================================================================
#================================ preparing path to read data ===============================================
current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
dump_path = os.path.join(project_root, "data", "processed")
#============================================================================================================
#================================ read data =================================================================
xtrain = pd.read_csv(f"{dump_path}/xtrain.csv", usecols=lambda column: column != 'Unnamed: 0')
ytrain = pd.read_csv(f"{dump_path}/ytrain.csv", usecols=lambda column: column != 'Unnamed: 0')
xtest  = pd.read_csv(f"{dump_path}/xtest.csv" , usecols=lambda column: column != 'Unnamed: 0')
ytest  = pd.read_csv(f"{dump_path}/ytest.csv" , usecols=lambda column: column != 'Unnamed: 0')
#============================================================================================================
#================================ use Random Forest Model ===================================================
RF_model = RandomForestClassifier(n_estimators=100, random_state=42)
#============================================================================================================
#================================ training model ============================================================
RF_model.fit(xtrain, ytrain)
#============================================================================================================
#================================ Prediction ================================================================
y_train_prd=RF_model.predict(xtrain)
y_test_pred = RF_model.predict(xtest)
#============================================================================================================
#================================ get score =================================================================
train_accuracy=accuracy_score(y_train_prd,ytrain)
test_accuracy=accuracy_score(y_test_pred,ytest)
#============================================================================================================
#================================ classification Report =====================================================
classification_report = classification_report(ytest, y_test_pred)
#============================================================================================================
#================================ printing some info ========================================================
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:\n", classification_report)
#============================================================================================================
#================================ preparing path to save accuracy scores and plots ==========================
current_dir = os.getcwd()
project_root = os.path.dirname(current_dir)
out_path = os.path.join(project_root, "out")
#============================================================================================================
#================================ Write scores to a file ====================================================
with open(f'{out_path}/metrics.txt', 'w') as f:
        f.write(" Random Forest Model  \n")
        f.write('======' + '\n')        
        f.write(f"F1-score of Training is: {train_accuracy*100:.2f} %"+"\n")
        f.write("\n")
        f.write(f"F1-Score of Validation is: {test_accuracy*100:.2f} %"+"\n")
        f.write("\n")     
        f.write(f"classification_report is:\n")
        f.write(f"{classification_report} \n")
        f.write('======' + '\n')
#============================================================================================================
#================================ save confusion matrix as plot =============================================
CM=confusion_matrix(y_test_pred,ytest,)
sns.heatmap(CM, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.title("confusion_matrix")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f'{out_path}/confusion_matrix.png', bbox_inches='tight', dpi=300)
#============================================================================================================
#================================================= Done =====================================================
print("="*50)
print("-- Training Model Done -- ")
print("="*50)
#============================================================================================================
#============================================================================================================