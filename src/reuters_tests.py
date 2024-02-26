import time
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
from emco import ExtrapolatedMarkovChainOversampling as EMCO


'''
Downloads and preprocesses Reuters-21578 dataset (ModApte) and performs oversampling
and classification experiments (for either headlines or full articles). The results are
macro-averages of N ( = 5) repetitions per category and are aggregated for low frequency
and very low frequency classes based on the given cutoff value ( = 0.015 ). Oversampling
can be executed on different balance ratios.

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
cutoff = 0.015 # cutoff between low and very low frequency categories

### SVC tolerance:
svc_tol = 1e-3

### Only classes with minority frequency lower than 0.75*balance_ratio are selected
balance_ratio  = 0.2   # Relative minority frequency after sampling
only_headlines = False # True -> only headlines, False -> full Reuters articles
train_min_df   = 2     # Only words that appear more than "train_min_df" times in
		       # training documents are included in vocabulary

### Preprocessing setup:
stopwords = nltk_stopwords.words('english')
tf     = False # default is False
idf    = True
smooth = True  # this should be True to prevent zero divisions
norm   = 'l2'

### Select only categories that have lower frequency than 0.75 times the balance ratio:
categories = []
for label in reuters.categories():
	minority_instances = sum(['training' in index for index in reuters.fileids(label)])
	if (minority_instances / 7769) < 0.75*balance_ratio:
		categories.append(label)

category_frequencies = {}
category_averages    = {}

start_time = datetime.now()

for j, category in enumerate(categories):
	
	print(f"\t * Category {j+1} / {len(categories)} * ")
	print("\t *", datetime.now(), "* ")
	print()
		
	category_results = []
	
	for k in range(N):
		
		if k == 0:
			
			### Initialize the training and test sets:
			tr_docs, tr_vocabulary, y_tr, te_docs, y_te = initialize_data(
				   pos_class=category,
				   stopwords=stopwords,
				   train_min_df=train_min_df,
				   only_headlines=only_headlines)
			
			### Save the category frequencies:
			category_frequencies[category] = sum(y_tr) / len(y_tr)
			
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
			mco   = EMCO(gamma=0)
			emco  = EMCO(gamma=1)
			### Fitting EMCO doesn't include randomness, so do it only
			### once for each category:
			mco.fit(tr_docs, y_tr, pos_class=1)
			emco.fit(tr_docs, y_tr, pos_class=1)
			### Free some memory:
			tr_docs   = []
			te_docs   = []
			te_corpus = []
		
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
		
		print("\nTesting with linear SVM ...")
		print(datetime.now(), "\n")
		
		method,bAcc,TPR,TNR,prec,f_1,f_2 = [],[],[],[],[],[],[]
		for X, y, name in zip([X_tr, Xros, Xsmote, Xada, Xdro, Xdecom, Xemco, Xmco],
				      [y_tr, yros, ysmote, yada, ydro, ydecom, yemco, ymco],
				      ["Original","ROS","SMOTE","ADASYN","DRO","DECOM","EMCO","MCO"]):
			if name == "DRO":
				### DRO transforms also the test data:
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
		time.sleep(1)
		
	category_averages[category] = pd.concat(
		category_results).groupby(by='Method').mean()
	

runtime = str(datetime.now()-start_time)
print("\n\t *** COMPLETED *** ")
print(f"\t TOTAL RUNTIME: {runtime}")

### Collect the results
VLF = [] # very low frequency categories
LF  = [] # low frequency categories

for category in categories:
    if category_frequencies[category] < cutoff:
        VLF.append(category_averages[category])
    else:
        LF.append(category_averages[category])

vlf_averages = pd.concat(VLF).groupby(by='Method').mean()
lf_averages  = pd.concat(LF).groupby(by='Method').mean()
