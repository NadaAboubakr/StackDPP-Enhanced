#!/usr/bin/env python
# coding: utf-8

# # Import

# In[1]:


import _pickle
import numpy as np
import pandas as pd
from mlxtend.evaluate import bias_variance_decomp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, RandomForestClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, make_scorer, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, LeaveOneOut,
                                     RandomizedSearchCV, StratifiedKFold,
                                     cross_validate)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier


# # GV and Func

# In[2]:


def specificity(y_true, y_pred):
    return (2 * accuracy_score(y_true, y_pred)) - recall_score(y_true, y_pred)
    # tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
    # return tn / (tn + fp)


scoring = {
    "accuracy": make_scorer(accuracy_score),
    "mcc": make_scorer(matthews_corrcoef),
    "precision": make_scorer(precision_score),
    "specificity": make_scorer(specificity, greater_is_better=True),
    "sensitivity": make_scorer(recall_score),
    "auroc": make_scorer(roc_auc_score),
    "ap": make_scorer(average_precision_score),
}


# In[3]:


clfs = {
    'DT':
    DecisionTreeClassifier(random_state=0),
    'LR':
    LogisticRegression(max_iter=10000, random_state=0),
    'RFC':
    RandomForestClassifier(random_state=0),
    'SVC(kernel = rbf)':
    SVC(kernel='rbf', random_state=0),
    'SVC(kernel = rbf, tuned)':
    SVC(kernel='rbf', C=5.44, gamma=.00237, random_state=0),
    'SVC(kernel = linear)':
    SVC(kernel='linear', random_state=0),
    'SVC(kernel = poly)':
    SVC(kernel='poly', random_state=0),
    'SVC(kernel = sigmoid)':
    SVC(kernel='sigmoid', random_state=0),
    'EXT':
    ExtraTreesClassifier(random_state=0),
    'GNB':
    GaussianNB(),
    "ADB":
    AdaBoostClassifier(random_state=0),
    "LDA":
    LinearDiscriminantAnalysis(),
    "DT":
    DecisionTreeClassifier(random_state=0),
    "KNN":
    KNeighborsClassifier(),
    'BG':
    BaggingClassifier(random_state=0),
    'BGsvmrbf':
    BaggingClassifier(estimator=SVC(kernel='rbf',
                                    random_state=0,
                                    probability=True),
                     random_state=0)
}


# In[4]:


skf = StratifiedKFold(
    n_splits=10,
    random_state=0,
    shuffle=True)


# In[5]:


def CV(X, y, model, cv=10):
    scores = cross_validate(model,
                            X,
                            y,
                            cv=cv,
                            scoring=scoring,
                            return_train_score=True,
                            return_estimator=True,
#                             verbose=1
                           )

    return {
        'ACC(%)': np.mean(scores["test_accuracy"]) * 100,
        'SN(%)': np.mean(scores["test_sensitivity"]) * 100,
        'SP(%)': np.mean(scores["test_specificity"]) * 100,
        'MCC': np.mean(scores["test_mcc"]),
    }


# In[6]:


def IND(X, y, test_X, test_y, model):
    model.fit(X, y)
    pred_y = model.predict(test_X)

    return {
        'ACC(%)': accuracy_score(test_y, pred_y) * 100,
        'SE(%)': recall_score(test_y, pred_y) * 100,
        'SP(%)': specificity(test_y, pred_y) * 100,
        'MCC': matthews_corrcoef(test_y, pred_y),
    }


# In[7]:


def JK(X, y, model, verbose = False):
    loo = LeaveOneOut()
    original = []
    pred = []
    i = 0
    for train_idx, test_idx in loo.split(X):
        print(i, end = '\t')
        i += 1
#         print(test_idx, end=',')
        model.fit(X[train_idx], y[train_idx])
        original.append(y[test_idx])
        pred.append(model.predict(X[test_idx]))
        
    return {
        'ACC(%)': accuracy_score(original, pred) * 100,
        'SE(%)': recall_score(original, pred) * 100,
        'SP(%)': specificity(original, pred) * 100,
        'MCC': matthews_corrcoef(original, pred),
    }


# # Load data

# In[8]:

import os

file_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Features', 'rf452.npz')


# In[9]:


data = np.load(file_name)

X = data['X']
y = data['y']
test_X = data['test_X']
test_y = data['test_y']

print(X.shape, test_X.shape)


# In[10]:


sc = StandardScaler()
X = sc.fit_transform(X)
test_X = sc.transform(test_X)


# In[11]:


np.max(test_X), np.max(X), np.min(test_X), np.min(X)


# # Test

# In[12]:


report_cv = []
report_ind = []

for name, clf in clfs.items():
    print(name)
    report_cv.append(CV(X, y, clf, cv= skf))
    report_cv[-1]['Classifier'] = name
    
    report_ind.append(IND(X, y, test_X, test_y, clf))
    report_ind[-1]['Classifier'] = name


# In[13]:


report_jk = []
name = 'SVC(kernel = rbf, tuned)'
report_jk.append(JK(X, y, clfs[name]))
report_jk[-1]['Classifier'] = name

print(clfs[name])

report_cv = pd.DataFrame(report_cv)
report_ind = pd.DataFrame(report_ind)
report_jk = pd.DataFrame(report_jk)

print(report_cv)

print(report_ind)

print(report_jk)


# In[14]:

# Extract just the filename without path for CSV naming
base_filename = os.path.splitext(os.path.basename(file_name))[0]

# ranked_features.to_csv('Feature_ranking_' + base_filename + '.csv', index = False)
report_cv.to_csv('Report_CV_' + base_filename + '.csv', index = False)
report_ind.to_csv('Report_IND_' + base_filename + '.csv', index=False)
report_jk.to_csv('Report_JK_' + base_filename + '.csv', index=False)


# # Tuning

# In[15]:


# parma_dist = dict(
#     C = np.logspace(-2, 4, 10),
#     gamma = np.logspace(-9, 3, 10)
# )

# rnd = RandomizedSearchCV(clf,
#                          param_distributions=parma_dist,
#                          scoring='accuracy',
#                          cv=skf,
#                          random_state=0,
#                          n_jobs=4,
#                          verbose=1)

# rnd.fit(X, y)


# In[16]:


parma_dist = dict(
    C = np.logspace(-2, 4, 10),
    gamma = np.logspace(-9, 3, 10)
)

grid = GridSearchCV(clfs['SVC(kernel = rbf)'],
                    param_grid=parma_dist,
                    cv=skf,
                    n_jobs=-1,
                    verbose=1,
                    scoring='accuracy')

grid.fit(X, y)

print(grid.best_params_, grid.best_score_, IND(X, y, test_X, test_y, grid.best_estimator_))
print(grid.param_grid)


# In[17]:


parma_dist = dict(
    C = np.linspace(1, 21, 10),
    gamma = np.linspace(0.0002, 0.01, 10)
)

grid = GridSearchCV(clfs['SVC(kernel = rbf)'],
                    param_grid=parma_dist,
                    cv=skf,
                    n_jobs=-1,
                    verbose=1,
                    scoring='accuracy')

grid.fit(X, y)

print(grid.best_params_, grid.best_score_, IND(X, y, test_X, test_y, grid.best_estimator_))
print(grid.param_grid)


# # Ensemble

# In[18]:


def JKSL(X, y, verbose=False):
    original = []
    pred = []

    loo = LeaveOneOut()
    # Define the stacking ensemble as in the main code
    level0 = [
        ('VC', VotingClassifier([
            ('DT', DecisionTreeClassifier(random_state=0)),
            ('ABC', AdaBoostClassifier(random_state=0)),
            ('LDA', LinearDiscriminantAnalysis()),
        ], voting='soft', n_jobs=-1)),
        ('svm(rbf, tuned)', SVC(kernel='rbf', C=5.44, gamma=.00237, random_state=0, probability=True))
    ]
    level1 = LogisticRegression(solver='liblinear', random_state=0)

    for train_idx, test_idx in loo.split(X):
        print(len(original), end='\t')
        model = StackingClassifier(
            estimators=level0,
            final_estimator=level1,
            stack_method='predict_proba',
            cv=10
        )
        model.fit(X[train_idx], y[train_idx])
        original.append(y[test_idx])
        pred.append(model.predict(X[test_idx])[-1])

    return {
        'ACC(%)': accuracy_score(original, pred) * 100,
        'SE(%)': recall_score(original, pred) * 100,
        'SP(%)': specificity(original, pred) * 100,
        'MCC': matthews_corrcoef(original, pred),
    }


# In[20]:


# make a prediction with a stacking ensemble
# define the base models
level0 = list()

level0.append((
    'VC',
    VotingClassifier(
        [
            ('DT', DecisionTreeClassifier(random_state=0)),
            ('ABC', AdaBoostClassifier(random_state=0)),
            ('LDA', LinearDiscriminantAnalysis()),
            #             ('SVCL', SVC(kernel='poly', random_state=0, probability=True))
        ],
        voting='soft',
        n_jobs=-1,
        # weights=[1, 1, 1.5]
    )))
# level0.append(('ext', ext(random_state=0)))
level0.append(('svm(rbf, tuned)',
               SVC(kernel='rbf',
                   C=5.44,
                   gamma=.00237,
                   random_state=0,
                   probability=True)))
# level0.append(('svc(poly)', SVC(kernel='poly',
#                                 random_state=0,
#                                 probability=True)))

# define meta learner model
level1 = LogisticRegression(solver='liblinear' ,random_state=0)
# level1 = SVC(kernel='linear', random_state=0)
# level1 = LinearDiscriminantAnalysis()
# define the stacking ensemble
model = StackingClassifier(estimators=level0,
                           final_estimator=level1,
                           stack_method='predict_proba',
                           cv=skf)
# fit the model on all available data
model.fit(X, y)
# make a prediction for one example
yhat = model.predict(test_X)
print('Predicted Accuracy: %f' % accuracy_score(test_y, yhat))
print('Predicted Sensitivity: %f' % recall_score(test_y, yhat))
print('Predicted Specificity: %f' % specificity(test_y, yhat))
print('Predicted MCC: %f' % matthews_corrcoef(test_y, yhat))


# In[21]:


print('10-Fold CV')
print(CV(X, y, model, StratifiedKFold(n_splits=10, shuffle=True, random_state=0)))

print(model)

print('JK')
print(JKSL(X, y, model))


# # Others

# In[22]:


vt = VotingClassifier(estimators=[
    ('gnb', GaussianNB()),
    ('adb', AdaBoostClassifier(random_state=0)),
    ('svmrbf', SVC(C=5.44, gamma=0.00273, probability=True, random_state=0)),
    ('ext', ExtraTreesClassifier(random_state=0))
],
                      voting='hard',
                      weights=[0.3, 0.3, 0.7, 0.3])

vt.fit(X, y)

p = vt.predict(test_X)

print(accuracy_score(test_y, p))


# In[ ]:


loss, bias, var = bias_variance_decomp(model, X, y, test_X, test_y, loss='0-1_loss', random_seed=0)
print('0-1 loss: ', loss)
print('Bias : ', bias)
print('Variance : ', var)


# In[26]:


model = clfs['SVC(kernel = rbf, tuned)']
mse, bias, var = bias_variance_decomp(model,
                                         X,
                                         y,
                                         test_X,
                                         test_y,
                                         loss='mse',
                                         random_seed=0)
print('Mean Square Error: ', mse)
print('Bias : ', bias)
print('Variance : ', var)


# In[ ]:




