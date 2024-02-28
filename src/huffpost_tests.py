import time
import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt
from scipy.sparse import dok_matrix

from utils import preprocess, test
from nltk.corpus import stopwords as nltk_stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from imblearn.over_sampling import RandomOverSampler as ROS
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from pydro.src.dro import DistributionalRandomOversampling
from DECOMlike import LDAOS
from emco import ExtrapolatedMarkovChainOversampling as EMCO


'''
Downloads and preprocesses HuffPost News Category Dataset (should be included in the 
working directory, see, https://www.kaggle.com/datasets/rmisra/news-category-dataset)
and performs oversampling and classification experiments (for either headlines or
headlines + articles). The dataset can either be considered in its entirety or be split
into subsets (then category results are macro-averages over the subsets). The results
are aggregated for low frequency and very low frequency classes based on the given
cutoff value ( = 0.015 ). Oversampling can be executed on different balance ratios.

Note that
 a) SMOTE and ADASYN use 5 nearest neighbors, or the maximum number of available
    neighbors if the minority sample consists of only five or less observations
 b) if there is only one minority training example, SMOTE and ADASYN are replaced
    with ROS (as there are no neighbors to be used)
 c) if there are no majority observations in any of the minority neighborhoods,
    the sampling density function is not defined and ADASYN is replaced with SMOTE
'''

cutoff = 0.015 # cutoff between low and very low frequency categories

### SVC tolerance:
svc_tol = 1e-3

data_type  = 'headline' # 'headline' (-> title) or 'complete' (-> title & description)
data_split = 5 # number of subsets to split the data into

### The size of latent space in DRO equals the length of training data, so, in
### order to prevent memory issues, dropping DRO from experiments may be needed:
skip_dro = True

stopwords = nltk_stopwords.words('english')
tf     = False # default is False
idf    = True
smooth = True  # this should be True to prevent zero divisions
norm   = 'l2'

BALANCE_RATIO = 0.1 # Relative minority frequency after sampling
train_min_df  = 2   # Only words that appear more than "train_min_df" times in
		    # training documents are included in vocabulary


### Initialize the data:
print("\nInitialize data...")
print(datetime.now())
data = pd.read_json('News_Category_Dataset_v3.json', orient='records', lines=True)

headlines  = []
completes  = []
categories = []

print((int(len(data)/10000)+1)*'-')
for i in range(len(data)):
    if i%10000 == 0:
        print('*', end='')
    title = ','.join(RegexpTokenizer(r'[a-zA-Z]+').tokenize(data.iloc[i].headline))
    desc  = ','.join(RegexpTokenizer(r'[a-zA-Z]+').tokenize(data.iloc[i].short_description))
    label = data.iloc[i].category
    if len(title) > 0 and len(desc) > 0 and len(label) > 0:
        headlines.append(title)
        completes.append(','.join([title, desc]))
        categories.append(label)
print()

data = pd.DataFrame()
if data_type == 'headline':
	data['text'] = headlines
else:
	data['text'] = completes
data['category'] = categories
category_set = list(set(data.category))

### Category frequencies:
category_frequencies = {}
for category in category_set:
	category_frequencies[category] = np.array((data.category==category)*1).mean()

### Select only categories that have lower frequency than 0.75 times the balance ratio:
categories = []
for category in category_set:
	if category_frequencies[category] < 0.75*BALANCE_RATIO:
		categories.append(category)

split = int(round(len(data)/data_split, 0))
### Random shuffle before splitting the data into subsets:
data  = data.sample(frac=1, random_state=42).reset_index(drop=True)

### Split the data into subsets:
datasets = []
for i in range(data_split):
	if i+1 < data_split:
		datasets.append(data[i*split : (i+1)*split])
	else:
		datasets.append(data[i*split : ])
data = [] # Free memory

shares = []
for c in categories:
	set_shares = []
	for dataset in datasets:
		set_shares.append(np.array((dataset.category == c)*1).mean())
	shares.append(set_shares)

f, ax = plt.subplots(6, 7, sharey=True, figsize=(18,12))
i = 0
j = 0
for c, row in zip(categories, shares):
    ax[i,j].bar(np.arange(data_split), row)
    ax[i,j].set_xticks([])
    ax[i,j].set_xlabel(c)
    j += 1
    if j == 7:
        i += 1
        j = 0
plt.show()

### Go through the categories and sets:

category_averages = {}

start_time = datetime.now()

for j, category in enumerate(categories):
	
	print()
	print(f" * Category {j+1} / {len(categories)} * ")
	print(" *", datetime.now(), "*")
	print()
	
	category_results = []
	
	for k, dataset in enumerate(datasets):
		
		labels = np.array((dataset.category == category)*1)
		
		### Train-test split:
		train_data, test_data, y_tr, y_te = train_test_split(
			dataset.text, labels, train_size=0.5, random_state=42)
		
		tr_docs, tr_vocabulary, _ = preprocess(list(train_data), min_df=train_min_df,
						       stopwords=stopwords, sep=',', stem=True)
		te_docs, _, _ = preprocess(list(test_data), min_df=0, sep=',', stem=True)
		
		### Training (and test) data is transformed before oversampling:
		tr_corpus = [' '.join(doc) for doc in tr_docs]
		te_corpus = [' '.join(doc) for doc in te_docs]
		pipe = Pipeline([('count', CountVectorizer(vocabulary=tr_vocabulary)),
				 ('tfidf', TfidfTransformer(norm=norm,
							    use_idf=idf,
							    smooth_idf=smooth, 
							    sublinear_tf=tf))]).fit(tr_corpus)
		X_tr = dok_matrix(pipe.transform(tr_corpus))
		X_te = dok_matrix(pipe.transform(te_corpus))
		
		### Free some memory 1/4:
		labels     = []
		train_data = []
		test_data  = []
		te_docs    = []
		te_corpus  = []
		
		print()
		print(f"Set {k+1} / {data_split}")
		print("\tMinority class ({}) frequency in the training set:".format(category), 
			  str(round(100*(sum(y_tr)/len(y_tr)),1))+"%\n")
		print(datetime.now())
		print()
		
		sampling_strategy = BALANCE_RATIO/(1-BALANCE_RATIO)
		
		### Initialize the oversampling methods:
		ros   = ROS(sampling_strategy=sampling_strategy)
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
		if not skip_dro:
			dro = DistributionalRandomOversampling(rebalance_ratio=BALANCE_RATIO)
			train_nwords = np.asarray(X_tr.sum(axis=1)).reshape(-1)
			test_nwords  = np.asarray(X_te.sum(axis=1)).reshape(-1)
		emco  = EMCO(gamma=1)
		mco   = EMCO(gamma=0)
		# These are the original DECOM hyperparameters:
		decom = LDAOS(n_topics=30, alpha=50/30, beta=0.01, max_iter=300)
		
		### General oversampling:
		Xros,   yros   = ros.fit_resample(X_tr, y_tr)
		Xsmote, ysmote = smote.fit_resample(X_tr, y_tr)
		try:
			Xada, yada = ada.fit_resample(X_tr, y_tr)
		except RuntimeError: # replace ADASYN with SMOTE
			Xada = Xsmote.copy()
			yada = ysmote.copy()
			
		### DRO:
		if not skip_dro:
			Xdro, ydro = dro.fit_transform(X_tr, np.asarray(y_tr), train_nwords)
			Xdro_te    = dro.transform(X_te, test_nwords, samples=1)
		else:
			Xdro, ydro = [], []
			Xdro_te    = []
		
		### DECOM:
		# decom is fitted with non-transformed data
		decom.fit(dok_matrix(pipe['count'].transform(tr_corpus)), y_tr)
		Xdecom, ydecom = decom.sample(s_ratio=BALANCE_RATIO)
		# then the whole decom sample is transformed
		Xdecom = pipe['tfidf'].transform(dok_matrix(Xdecom))
		
		### EMCO:
		emco.fit(tr_docs, y_tr, pos_class=1)
		mco.fit(tr_docs, y_tr, pos_class=1)
		
		### Free some memory 2/4:
		tr_corpus = []
		tr_docs   = []
		
		# First oversample the synthetic documents:
		emco_docs, yemco = emco.sample(s_ratio=BALANCE_RATIO, length='auto', complete=True)
		mco_docs, ymco   = mco.sample(s_ratio=BALANCE_RATIO, length='auto', complete=True)
		# Then transform the oversampled data:
		Xemco = dok_matrix(pipe.transform([' '.join(doc) for doc in emco_docs]))
		Xmco  = dok_matrix(pipe.transform([' '.join(doc) for doc in mco_docs]))
		
		### Free some memory 3/4:
		emco_docs = []
		mco_docs  = []
		ros   = ROS(sampling_strategy=sampling_strategy)
		ada   = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=min(5,sum(y_tr)-1))
		if not skip_dro:
			dro = DistributionalRandomOversampling(rebalance_ratio=BALANCE_RATIO)
		emco  = EMCO(gamma=1)
		mco   = EMCO(gamma=0)
		decom = LDAOS(n_topics=30, alpha=50/30, beta=0.01, max_iter=300)
		
		print("\nTesting with linear SVM ...")
		print(datetime.now())
		
		method,bAcc,TPR,TNR,prec,f_1,f_2 = [],[],[],[],[],[],[]
		for X, y, name in zip([X_tr, Xros, Xsmote, Xada, Xdro, Xdecom, Xemco, Xmco],
				      [y_tr, yros, ysmote, yada, ydro, ydecom, yemco, ymco],
				      ["Original","ROS","SMOTE","ADASYN","DRO","DECOM","EMCO","MCO"]):
			if name == "DRO":
				if not skip_dro:
					res = test(X, y, Xdro_te, y_te, tol=svc_tol)
				else:
					continue
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
		
		### Free memory 4/4:
		X_tr,Xros,Xsmote,Xada,Xdro,Xdecom,Xemco,Xmco = [], [], [], [], [], [], [], []
		y_tr,yros,ysmote,yada,ydro,ydecom,yemco,ymco = [], [], [], [], [], [], [], []
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
