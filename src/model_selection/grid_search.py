"""
http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html

"""
from sklearn import datasets, svm
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

svc = svm.SVC(C=1, kernel='linear')
mdl = svc.fit(X_digits[:-100], y_digits[:-100])
mdl.score(X_digits[-100:], y_digits[-100:])   # out of sample

# manually do kfold cross validation
X_folds = np.array_split(X_digits, 3)
y_folds = np.array_split(y_digits, 3)

X = ['a', 'a', 'b', 'c', 'c', 'c']
k_fold =KFold(n_splits=3)
scores = [svc.fit(X_digits[train], y_digits[train]).score(X_digits[test], y_digits[test])
          for train, test in k_fold.split(X_digits)]

# kfold by the helper
scs = cross_val_score(svc, X_digits, y_digits, cv=k_fold, n_jobs=-1)
scs = cross_val_score(svc, X_digits, y_digits, cv=k_fold, scoring='precision_macro')

# Grid search - you need to define your param_grid by arrays of values
# you want the search to go through
cs = np.logspace(-6, -1, 10)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=cs), n_jobs=-1)
clf.fit(X_digits[:1000], y_digits[:1000])   # the actual grid search happens here

clf.best_score_        # best score
clf.best_estimator_.C  # best estimator i.e. the one with the highest scores on the left-out sets - of the model class
clf.score(X_digits[1000:], y_digits[1000:])  # out of sample score using the best set of param


