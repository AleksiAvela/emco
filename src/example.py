import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import dok_matrix

from utils import initialize_data, test
from nltk.corpus import reuters
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from pydro.src.dro import DistributionalRandomOversampling
from DECOMlike import LDAOS
from edaos import EDAOversampling as EDAOS
from emco import ExtrapolatedMarkovChainOversampling as EMCO


'''
This code performs the same tests with the same methods as presented in the
results in the article, but only for one specified class of Reuters-21578
dataset (ModApte) and without averaging over multiple repetitions.

Note that
 a) SMOTE and ADASYN use 5 nearest neighbors, or the maximum number of available
    neighbors if the minority sample consists of only five or less observations
 b) if there is only one minority training example, SMOTE and ADASYN are replaced
    with ROS (as there are no neighbors to be used)
 c) if there are no majority observations in any of the minority neighborhoods,
    the sampling density function is not defined and ADASYN is replaced with SMOTE
'''

### Choose one label as the positive class in OvR classification:
category = 'ipi'

### SVC tolerance:
svc_tol = 1e-3

balance_ratio  = 0.1
only_headlines = True # True -> headlines, False -> full Reuters articles
train_min_df   = 2

### Preprocessing setup:
stopwords = nltk_stopwords.words('english')
tf     = False # default is False
idf    = True
smooth = True  # this should be True to prevent zero divisions
norm   = 'l2'

min_f = sum(['training' in index for index in reuters.fileids(category)]) / 7769
assert min_f < 0.75*balance_ratio, "Select a category whose relative frequency is lower than 0.75 x balance_ratio"


### Initialize the training and test sets:
tr_raw, tr_docs, tr_vocabulary, tr_labels, te_docs, te_labels = initialize_data(
	   stopwords=stopwords,
	   train_min_df=train_min_df,
	   only_headlines=only_headlines)

tr_corpus = [' '.join(doc) for doc in tr_docs]
te_corpus = [' '.join(doc) for doc in te_docs]

# tf-idf transformer with the default settings:
pipe = Pipeline([('count', CountVectorizer(vocabulary=tr_vocabulary)),
		 ('tfidf', TfidfTransformer(norm=norm,
					    use_idf=idf,
					    smooth_idf=smooth, 
					    sublinear_tf=tf))]).fit(tr_corpus)

X_tr = dok_matrix(pipe.transform(tr_corpus))
X_te = dok_matrix(pipe.transform(te_corpus))

### Create binary target vectors from the raw labels:
y_tr, y_te = [], []
for train_label_set in tr_labels:
	y_tr.append(int(category in train_label_set))
for test_label_set in te_labels:
	y_te.append(int(category in test_label_set))

sampling_strategy = balance_ratio/(1-balance_ratio)

### Initialize the oversampling methods: 
ros = ROS(sampling_strategy=sampling_strategy)
### Number of nearest neighbors for SMOTE and ADASYN:
n_neighbors = min(5, sum(y_tr)-1) # max neighbors = 5 (the default value)
### If there is only one positive training example, use ROS:
if n_neighbors == 0:
	smote = ROS(sampling_strategy=sampling_strategy)
	ada   = ROS(sampling_strategy=sampling_strategy)
else:
	smote = SMOTE(sampling_strategy=sampling_strategy,
	              k_neighbors=n_neighbors)
	ada   = ADASYN(sampling_strategy=sampling_strategy,
	               n_neighbors=n_neighbors)
dro = DistributionalRandomOversampling(rebalance_ratio=balance_ratio)
train_nwords = np.asarray(X_tr.sum(axis=1)).reshape(-1)
test_nwords  = np.asarray(X_te.sum(axis=1)).reshape(-1)
# These are the original DECOM hyperparameters:
decom = LDAOS(n_topics=30, alpha=50/30, beta=0.01, max_iter=300)
# EDA Oversampling:
edaos = EDAOS(s_ratio=balance_ratio)
# MCO with gamma = 0 and EMCO with gamma = 1 and gamma = 0.1:
mco   = EMCO(gamma=0)
emco  = EMCO(gamma=1)
emco2 = EMCO(gamma=0.1)

### ROS:
Xros, yros = ros.fit_resample(X_tr, y_tr)

### SMOTE:
Xsmote, ysmote = smote.fit_resample(X_tr, y_tr)

### ADASYN:
try:
	Xada, yada = ada.fit_resample(X_tr, y_tr)
except RuntimeError: # replace ADASYN with SMOTE
	Xada = Xsmote.copy()
	yada = ysmote.copy()

### DRO:
Xdro, ydro = dro.fit_transform(X_tr, np.asarray(y_tr), train_nwords)
Xdro_te    = dro.transform(X_te, test_nwords, samples=1)

### DECOM:
# DECOM is fitted with non-transformed data:
decom.fit(dok_matrix(pipe['count'].transform(tr_corpus)), y_tr)
Xdecom, ydecom = decom.sample(s_ratio=balance_ratio)
# Then the whole DECOM sample is transformed:
Xdecom = pipe['tfidf'].transform(dok_matrix(Xdecom))

### EDA Oversampling:
eda_docs, yeda = edaos.sample(tr_raw, y_tr)
Xeda = dok_matrix(pipe.transform(eda_docs))

### MCO and EMCO:
mco.fit(tr_docs, y_tr, pos_class=1)
emco.fit(tr_docs, y_tr, pos_class=1)
emco2.fit(tr_docs, y_tr, pos_class=1)
# First oversample the synthetic documents:
mcodocs, ymco   = mco.sample(s_ratio=balance_ratio, length='auto', complete=True)
emcodocs, yemco = emco.sample(s_ratio=balance_ratio, length='auto', complete=True)
emcodocs2, yemco2 = emco2.sample(s_ratio=balance_ratio, length='auto', complete=True)
# Then transform the oversampled data:
Xmco   = dok_matrix(pipe.transform([' '.join(doc) for doc in mcodocs]))
Xemco  = dok_matrix(pipe.transform([' '.join(doc) for doc in emcodocs]))
Xemco2 = dok_matrix(pipe.transform([' '.join(doc) for doc in emcodocs2]))

print("\nTesting with linear SVM ...")
print(datetime.now(), "\n")

method,bAcc,TPR,TNR,prec,f_1,f_2 = [],[],[],[],[],[],[]

for X, y, name in zip(
		[X_tr, Xros, Xsmote, Xada, Xdro,
		 Xdecom, Xeda, Xemco, Xemco2, Xmco],
		[y_tr, yros, ysmote, yada, ydro,
		 ydecom, yeda, yemco, yemco2, ymco],
		["Original", "ROS", "SMOTE", "ADASYN", "DRO",
		 "DECOM", "EDA", "EMCO(1.0)", "EMCO(0.1)", "MCO"]):
	if name == "DRO":
		### DRO also transforms the test data:
		res = test(X, y, Xdro_te, y_te, tol=svc_tol)
	else:
		res = test(X, y, X_te, y_te, tol=svc_tol)

	method.append(name)
	bAcc.append(res[0])
	TPR.append(res[1])
	TNR.append(res[2])
	prec.append(res[3])
	f_1.append(res[4])
	f_2.append(res[5])

results = pd.DataFrame({'Method'    : method,
			'Bal. Acc.' : bAcc,
			'TPR'       : TPR,
			'TNR'       : TNR,
			'Precision' : prec,
			'F1'        : f_1,
			'F2'        : f_2})

print(results)
