import time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import dok_matrix

from utils import preprocess, test
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from pydro.src.dro import DistributionalRandomOversampling
from DECOMlike import LDAOS
from emco import ExtrapolatedMarkovChainOversampling as EMCO


'''
Downloads and preprocesses 20Newsgroups dataset and performs oversampling and
classification experiments. The results are macro-averages of N ( = 5) repetitions
per category. Oversampling can be executed on different balance ratios.

Note that
 a) SMOTE and ADASYN use maximum number of available neighbors
	(or at maximum 5)
 b) if there is only one positive training example, SMOTE and
	ADASYN are replaced with ROS
 c) if there are no majority observations included in the neighbors,
	ADASYN is replaced with SMOTE
'''

### Number of times to repeat oversampling & testing for one class. The presented
### results are averages of these repetitions:
N = 5

### SVC tolerance:
svc_tol = 1e-3

balance_ratio  = 0.1 # Relative minority frequency after sampling
train_min_df   = 2   # Only words that appear more than "train_min_df" times in
		     # training documents are included in vocabulary
sampling_strategy = balance_ratio/(1-balance_ratio)

### Preprocessing setup:
stopwords = nltk_stopwords.words('english')
tf     = False # default is False
idf    = True
smooth = True  # this should be True to prevent zero divisions
norm   = 'l2'


data_train = fetch_20newsgroups(
    subset="train",
    categories=None,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

data_test = fetch_20newsgroups(
    subset="test",
    categories=None,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

train_data = data_train.data
test_data  = data_test.data

train_labels = np.array([data_train.target_names[j] for j in data_train.target])
test_labels  = np.array([data_test.target_names[j] for j in data_test.target])

categories, label_counts = np.unique(train_labels, return_counts=True)
category_frequencies = {u : c/len(train_labels) for u,c in zip(categories, label_counts)}

category_averages = {}

start_time = datetime.now()

### Initialize the training and test sets:
tr_docs, tr_vocabulary, _ = preprocess(list(train_data), min_df=train_min_df,
				       stopwords=stopwords, sep=None, stem=True)
te_docs, _, _ = preprocess(list(test_data), min_df=0, sep=None, stem=True)

### Free some memory:
data_train   = []
data_test    = []
train_data   = []
test_data    = []
label_counts = []

### Drop empty documents:
tr_idx = np.array([i for i,doc in enumerate(tr_docs) if doc != ['<empty>']])
te_idx = np.array([i for i,doc in enumerate(te_docs) if doc != ['<empty>']])
tr_docs = [tr_docs[i] for i in tr_idx]
te_docs = [te_docs[i] for i in te_idx]
train_labels = train_labels[tr_idx]
test_labels  = test_labels[te_idx]
try:
	del tr_vocabulary['<empty>']
	tr_vocabulary = list(tr_vocabulary.keys())
except KeyError:
	print("No empty documents")

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

### Free some memory:
tr_idx        = []
te_idx        = []
te_docs       = []
te_corpus     = []

for j, category in enumerate(categories):
	
	print()
	print(f"\t * Category {j+1} / {len(categories)} * ")
	print("\t *", datetime.now(), "* ")
	print()
		
	category_results = []
	
	y_tr = (train_labels == category)*1
	y_te = (test_labels == category)*1
	
	for k in range(N):
		
		print()
		print(f" * Iteration {k+1} / {N} * ")
		print(" *", datetime.now(), "* ")
		print()
		
		if k == 0:
			
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
			mco   = EMCO(gamma=0)
			emco  = EMCO(gamma=1)
			### Fitting EMCO doesn't include randomness so do it only
			### once for each category:
			mco.fit(tr_docs, y_tr, pos_class=1)
			emco.fit(tr_docs, y_tr, pos_class=1)
		
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
		
		### MCO and EMCO:
		# First oversample the synthetic documents:
		mcodocs, ymco   = mco.sample(s_ratio=balance_ratio, length='auto', complete=True)
		emcodocs, yemco = emco.sample(s_ratio=balance_ratio, length='auto', complete=True)
		# Then transform the oversampled data:
		Xmco  = dok_matrix(pipe.transform([' '.join(doc) for doc in mcodocs]))
		Xemco = dok_matrix(pipe.transform([' '.join(doc) for doc in emcodocs]))
		
		### Free some memory:
		emcodocs = []
		mcodocs  = []
			
		print("\nTesting with linear SVM ...")
		print(datetime.now())
		
		method,bAcc,TPR,TNR,prec,f_1,f_2 = [],[],[],[],[],[],[]
		for X, y, name in zip([X_tr, Xros, Xsmote, Xada, Xdro, Xdecom, Xemco, Xmco],
				      [y_tr, yros, ysmote, yada, ydro, ydecom, yemco, ymco],
				      ["Original","ROS","SMOTE","ADASYN","DRO","DECOM","EMCO","MCO"]):
			if name == "DRO":
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
		
		category_results.append(results)
		
		### Free some memory:
		Xdro_te = []
		Xros, Xsmote, Xada, Xdro, Xdecom, Xemco, Xmco = [], [], [], [], [], [], []
		yros, ysmote, yada, ydro, ydecom, yemco, ymco = [], [], [], [], [], [], []
		time.sleep(1)
		
	category_averages[category] = pd.concat(
		category_results).groupby(by='Method').mean()
	

runtime = str(datetime.now()-start_time)
print("\n\t *** COMPLETED *** ")
print(f"\t TOTAL RUNTIME: {runtime}")

### Collect the results
result_averages  = pd.concat(
	list(category_averages.values())).groupby(by='Method').mean()
