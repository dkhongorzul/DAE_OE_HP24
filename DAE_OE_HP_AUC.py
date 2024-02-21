import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, mean_squared_error
#from sklearn.metrics import roc_auc_score, precision_score, plot_roc_curve, mean_squared_error
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
#import seaborn as sns
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm, datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import mean_squared_error
#sns.set()
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing, tree, svm, cluster
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

dataframe = pd.read_csv("D:/#2024/MBE/clean_old_0118_covid.csv")
cols = list(dataframe)
print(cols)
X = dataframe.values[0:, 0:-1]
print(X)
y = dataframe.values[0:, -1]
print(X.shape)

acc_class1_DAE_OE_NB=[]
pre_class1_DAE_OE_NB=[]
roc_auc_score_class1_DAE_OE_NB=[]
mean_squared_error_class1_DAE_OE_NB=[]
iou_class1_DAE_OE_NB=[]
dtprs_NB=[]

acc_class2_DAE_OE_RF=[]
pre_class2_DAE_OE_RF=[]
roc_auc_score_class2_DAE_OE_RF=[]
mean_squared_error_class2_DAE_OE_RF=[]
iou_class2_DAE_OE_RF=[]
dtprs_RF=[]

acc_class3_DAE_OE_DT=[]
pre_class3_DAE_OE_DT=[]
roc_auc_score_class3_DAE_OE_DT=[]
mean_squared_error_class3_DAE_OE_DT=[]
iou_class3_DAE_OE_DT=[]
dtprs_DT=[]

acc_class4_DAE_OE_KNN=[]
pre_class4_DAE_OE_KNN=[]
roc_auc_score_class4_DAE_OE_KNN=[]
mean_squared_error_class4_DAE_OE_KNN=[]
iou_class4_DAE_OE_KNN=[]
dtprs_KNN=[]

acc_class5_DAE_OE_XGB=[]
pre_class5_DAE_OE_XGB=[]
roc_auc_score_class5_DAE_OE_XGB=[]
mean_squared_error_class5_DAE_OE_XGB=[]
iou_class5_DAE_OE_XGB=[]
dtprs_XGB=[]

y = label_binarize(y, classes=[1, 2, 3])
n_classes = y.shape[1]

dmean_fpr = np.linspace( 0, 1, 100 )
count = 1
kf = KFold(n_splits=10)
#kf = KFold(n_splits=6)
for train_index,test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    classifiers=[]
    #classifiers.append(('LR', LogisticRegression(solver='liblinear')))
    classifiers.append(('DAE-OE-RF', OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=0, criterion='gini'))))
    classifiers.append(('DAE-OE-KNN', OneVsRestClassifier(KNeighborsClassifier(n_neighbors=3))))
    classifiers.append(('DAE-OE-XGB', OneVsRestClassifier(XGBClassifier(learning_rate= 0.01, max_depth= 4, gamma = 0, n_estimators= 500))))
    classifiers.append(('DAE-OE-DT', OneVsRestClassifier(DecisionTreeClassifier(criterion='gini'))))
    classifiers.append(('DAE-OE-NB', OneVsRestClassifier(GaussianNB())))

    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])
    colors = ['pink', 'purple', 'orange', 'red', 'blue']
    j = 0
    for name, cls in classifiers:
#        lw=3
        # classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True                                           ))
        model = cls.fit(X_train, y_train)
        yproba = model.predict(X_test)
        print('********************')
        jaccards = []
        accuracy = accuracy_score(y_test, yproba)
        pre = precision_score(y_test, yproba, average='weighted')
        mse = mean_squared_error(y_test, yproba)
        jaccards = jaccard_score(y_test, yproba, average='samples', pos_label=n_classes)
#        jaccards.append(jaccard)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], yproba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), yproba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        j = j + 1
        print('************fold-', count, '**********')
#        print(sk_report)
        if name == 'DAE-OE-NB':
            acc_class1_DAE_OE_NB.append((accuracy)*100.0)
            pre_class1_DAE_OE_NB.append((pre)*100.0)
            roc_auc_score_class1_DAE_OE_NB.append((roc_auc["micro"] * 100.0))
            mean_squared_error_class1_DAE_OE_NB.append(mse)
            iou_class1_DAE_OE_NB.append((jaccards)*100.0)
            dtprs_NB.append(np.interp(dmean_fpr, fpr["micro"], tpr["micro"]))
        elif name == 'DAE-OE-RF':
            acc_class2_DAE_OE_RF.append((accuracy)*100.0)
            pre_class2_DAE_OE_RF.append((pre)*100.0)
            roc_auc_score_class2_DAE_OE_RF.append((roc_auc["micro"] * 100.0))
            mean_squared_error_class2_DAE_OE_RF.append(mse)
            iou_class2_DAE_OE_RF.append((jaccards) * 100.0)
            dtprs_RF.append(np.interp(dmean_fpr, fpr["micro"], tpr["micro"]))
        elif name == 'DAE-OE-DT':
            acc_class3_DAE_OE_DT.append((accuracy)*100.0)
            pre_class3_DAE_OE_DT.append((pre)*100.0)
            roc_auc_score_class3_DAE_OE_DT.append((roc_auc["micro"] * 100.0))
            mean_squared_error_class3_DAE_OE_DT.append(mse)
            iou_class3_DAE_OE_DT.append((jaccards) *100.0)
            dtprs_DT.append(np.interp(dmean_fpr, fpr["micro"], tpr["micro"]))
        elif name == 'DAE-OE-KNN':
            acc_class4_DAE_OE_KNN.append((accuracy)*100.0)
            pre_class4_DAE_OE_KNN.append((pre)*100.0)
            roc_auc_score_class4_DAE_OE_KNN.append((roc_auc["micro"] * 100.0))
            mean_squared_error_class4_DAE_OE_KNN.append(mse)
            iou_class4_DAE_OE_KNN.append((jaccards) * 100.0)
            dtprs_KNN.append(np.interp(dmean_fpr, fpr["micro"], tpr["micro"]))
        elif name == 'DAE-OE-XGB':
            acc_class5_DAE_OE_XGB.append((accuracy)*100.0)
            pre_class5_DAE_OE_XGB.append((pre)*100.0)
            roc_auc_score_class5_DAE_OE_XGB.append((roc_auc["micro"] * 100.0))
            mean_squared_error_class5_DAE_OE_XGB.append(mse)
            iou_class5_DAE_OE_XGB.append((jaccards) * 100.0)
            dtprs_XGB.append(np.interp(dmean_fpr, fpr["micro"], tpr["micro"]))
        # geh met busdiigaa nemne
    count += 1
    if count ==11:
        break
plt.plot( [0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8 )
#plt.plot(fpr["micro"], tpr["micro"], label=name + '_ROC curve (area = {0:0.2f}) '
#                         ''.format(roc_auc["micro"]), color=colors[j], linewidth=2)
#print(name, " Roc auc score: %.2f%%" % (roc_auc["micro"] * 100.0))

mean_accuracy_class1 = np.mean(acc_class1_DAE_OE_NB)
mean_pre_class1 = np.mean(pre_class1_DAE_OE_NB)
mean_roc_class1 = np.mean(roc_auc_score_class1_DAE_OE_NB)
mean_mse_class1 = np.mean(mean_squared_error_class1_DAE_OE_NB)
mean_iou_class1 = np.mean(iou_class1_DAE_OE_NB)

mean_accuracy_class5 = np.mean(acc_class5_DAE_OE_XGB)
mean_pre_class5 = np.mean(pre_class5_DAE_OE_XGB)
mean_roc_class5 = np.mean(roc_auc_score_class5_DAE_OE_XGB)
mean_mse_class5 = np.mean(mean_squared_error_class5_DAE_OE_XGB)
mean_iou_class5 = np.mean(iou_class5_DAE_OE_XGB)

mean_accuracy_class2 = np.mean(acc_class2_DAE_OE_RF)
mean_pre_class2 = np.mean(pre_class2_DAE_OE_RF)
mean_roc_class2 = np.mean(roc_auc_score_class2_DAE_OE_RF)
mean_mse_class2 = np.mean(mean_squared_error_class2_DAE_OE_RF)
mean_iou_class2 = np.mean(iou_class2_DAE_OE_RF)

mean_accuracy_class3 = np.mean(acc_class3_DAE_OE_DT)
mean_pre_class3 = np.mean(pre_class3_DAE_OE_DT)
mean_roc_class3 = np.mean(roc_auc_score_class3_DAE_OE_DT)
mean_mse_class3 = np.mean(mean_squared_error_class3_DAE_OE_DT)
mean_iou_class3 = np.mean(iou_class3_DAE_OE_DT)

mean_accuracy_class4 = np.mean(acc_class4_DAE_OE_KNN)
mean_pre_class4 = np.mean(pre_class4_DAE_OE_KNN)
mean_roc_class4 = np.mean(roc_auc_score_class4_DAE_OE_KNN)
mean_mse_class4 = np.mean(mean_squared_error_class4_DAE_OE_KNN)
mean_iou_class4 = np.mean(iou_class4_DAE_OE_KNN)

print('************************Average*****************************')
print('mean_acc_XGB=%0.3f'% mean_accuracy_class5)
print('mean_ROC_XGB=%0.3f'% mean_roc_class5)
print('mean_F1-score_XGB=%0.3f'% mean_pre_class5)
print('mean_MSE_XGB=%0.3f'% mean_mse_class5)
print('mean_IoU_XGB=%0.3f'% mean_iou_class5)
mean_tpr_xgb = np.mean( dtprs_XGB, axis=0 )
mean_tpr_xgb[-1] = 1.0
plt.plot(dmean_fpr, mean_tpr_xgb, linestyle='-', color='blue', label=r'MACE_XGB (Mean AUC = %0.2f )' % (mean_roc_class5), lw=3, alpha=.8)


print('************************Average*****************************')
print('acc_KNN acc=%0.3f '% mean_accuracy_class4)
print('mean_ROC_KNN=%0.3f'% mean_roc_class4)
print('mean_F1-score_KNN=%0.3f'% mean_pre_class4)
print('mean_MSE_KNN=%0.3f'% mean_mse_class4)
print('mean_IoU_KNN=%0.3f'% mean_iou_class4)
mean_tpr_knn = np.mean( dtprs_KNN, axis=0 )
mean_tpr_knn[-1] = 1.0
plt.plot(dmean_fpr, mean_tpr_knn,linestyle='--', color='pink', label=r'MACE_KNN (Mean AUC = %0.2f )' % (mean_roc_class4), lw=3, alpha=.8)

print('************************Average*****************************')
print('acc_DT acc=%0.3f '% mean_accuracy_class3)
print('mean_ROC_DT=%0.3f'% mean_roc_class3)
print('mean_F1-score_DT=%0.3f'% mean_pre_class3)
print('mean_MSE_DT=%0.3f'% mean_mse_class3)
print('mean_IoU_DT=%0.3f'% mean_iou_class3)
mean_tpr_dt = np.mean( dtprs_DT, axis=0 )
mean_tpr_dt[-1] = 1.0
plt.plot(dmean_fpr, mean_tpr_dt, linestyle=':', color='red', label=r'MACE_DT (Mean AUC = %0.2f )' % (mean_roc_class3), lw=3, alpha=.8)

print('************************Average*****************************')
print('acc_RF acc=%0.3f'% mean_accuracy_class2)
print('mean_ROC_RF=%0.3f'% mean_roc_class2)
print('mean_F1-score_RF=%0.3f'% mean_pre_class2)
print('mean_MSE_RF=%0.3f'% mean_mse_class2)
print('mean_IoU_RF=%0.3f'% mean_iou_class2)
mean_tpr_rf = np.mean( dtprs_RF, axis=0 )
mean_tpr_rf[-1] = 1.0
plt.plot(dmean_fpr, mean_tpr_rf, linestyle='-.', color='green', label=r'MACE_RF (Mean AUC = %0.2f )' % (mean_roc_class2), lw=3, alpha=.8)


print('************************Average*****************************')
print('mean_acc_NB =%0.3f' % mean_accuracy_class1)
print('mean_ROC_NB=%0.3f' % mean_roc_class1)
print('mean_F1-score_NB=%0.3f' % mean_pre_class1)
print('mean_MSE_NB=%0.3f' % mean_mse_class1)
print('mean_IoU_NB=%0.3f' % mean_iou_class1)
mean_tpr_nb = np.mean( dtprs_NB, axis=0 )
mean_tpr_nb[-1] = 1.0
plt.plot(dmean_fpr, mean_tpr_nb, linestyle='--', color='purple', label=r'MACE_NB (Mean AUC = %0.2f )' % (mean_roc_class1), lw=3, alpha=.8)

plt.xlim( [-0.05, 1.05] )
plt.ylim( [-0.05, 1.05] )
plt.xlabel( 'False Positive Rate' )
plt.ylabel( 'True Positive Rate' )
plt.title( 'ROC curve to multi-class for Diabetes Risk Prediction' )
plt.legend( loc="lower right" )
plt.show()
