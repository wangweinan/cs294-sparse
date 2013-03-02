# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import scipy.io
import pylab as pl
from sklearn.linear_model import SGDRegressor
from sklearn.datasets.samples_generator import make_regression

X_Y = scipy.io.loadmat('model-stemmed.mat')

#For efficiency concerns, we transformed the sparse matrix into Compressed Sparse Row Matrix.
X = X_Y['Xstemmed']
X = scipy.sparse.csr_matrix(X)
Y = X_Y['yuniq']
Y=Y[:,0]

# <codecell>

X_RowSum = X.sum(axis = 1)

#Get all the index of sorted smap in decreasing order
SmapSortedIndex_Dec= sorted(range(len(X_RowSum)),key=lambda x:X_RowSum[x])

# <codecell>

#High frequency words
def truncate(s,k):
    return (s[0:k])

# <codecell>

truncated_Row = truncate(SmapSortedIndex_Dec,5000)

# <codecell>

#Stochastic Gradient Descent

#X_Test = X[:,0:1000]
#Y_Test = Y[0:1000]

clf = SGDRegressor(alpha=0.0001, eta0=0.01, fit_intercept=True,
       learning_rate='invscaling', loss='squared_loss', n_iter=20, p=0.1,
       penalty='l2', power_t=0.25, rho=0.85, seed=0, shuffle=True,
       verbose=0, warm_start=False)

clf.fit(X.transpose(), Y)

# <codecell>

clf.decision_function(X.transpose())

# <codecell>


