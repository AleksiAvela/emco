import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf
from nltk.corpus import reuters

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import balanced_accuracy_score, fbeta_score
from sklearn.metrics import recall_score, precision_score

from utils import initialize_data
from emco import ExtrapolatedMarkovChainOversampling as EMCO
from edaos import EDAOversampling as EDAOS


'''
Downloads and preprocesses Reuters-21578 dataset (ModApte) and performs oversampling
and classification experiments using an LSTM neural network with pre-trained fastText
word vectors (available at https://fasttext.cc/docs/en/english-vectors.html).
All documents are truncated and padded to be the same length. The results are
aggregated for low frequency and very low frequency classes based on the given cutoff
value ( = 0.015 ). Oversampling can be executed on different balance ratios.
'''


def load_vectors(fname, N=1000000):

    embeddings = {}
    i = -1
    with open(fname, 'r', encoding='utf-8') as f:
        for line in f:
            i += 1
            if i == 0: # The first row is just the dimensions
                continue
            values = line.rstrip().rsplit(' ')
            word   = values[0]
            coefs  = np.asarray(values[1:], dtype='float32')
            embeddings[word] = coefs
            if i == N:
                return embeddings		
    return embeddings


def test(X_tr, y_tr, X_te, y_te, embedding_matrix, truncate_n,
		 X_insample, y_insample):
	
	### X_tr and y_tr may include oversampled training instances, and therefore
	### original X_insample and y_insample are passed for in-sample evaluation
	
	### Create the embedding layer:
	embedding = tf.keras.layers.Embedding(embedding_matrix.shape[0],
                                          embedding_matrix.shape[1],
                                          input_length=truncate_n, 
                                          name='embedding')
	embedding.build(input_shape=(1,))
	embedding.set_weights([embedding_matrix])
	embedding.trainable = False
	
	### Initialize the model architecture:
	model = tf.keras.models.Sequential()
	model.add(embedding)
	model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
	model.add(tf.keras.layers.Dense(32, activation="relu")) 
	model.add(tf.keras.layers.Dropout(0.4)) 
	model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
	model.compile("rmsprop", 
			      "binary_crossentropy", 
				  metrics=["Recall"]) 
	
	### Fit the model:
	print()
	_ = model.fit(X_tr, y_tr, epochs=3)
	
	### Evaluate in-sample and out-of-sample:
	y_pred_is = 1*(model.predict(X_insample) > 0.5)
	y_pred_os = 1*(model.predict(X_te) > 0.5)
	
	### Clear memory:
	tf.keras.backend.clear_session()
	
	return {'in-sample' :  {'Bacc' : balanced_accuracy_score(y_insample, y_pred_is),
						    'F1'   : fbeta_score(y_insample, y_pred_is, beta=1),
							'F2'   : fbeta_score(y_insample, y_pred_is, beta=2),
							'TPR'  : recall_score(y_insample, y_pred_is),
							'TNR'  : recall_score(y_insample, y_pred_is, pos_label=0),
							'Prec' : precision_score(y_insample, y_pred_is)},
		    'out-sample' : {'Bacc' : balanced_accuracy_score(y_te, y_pred_os),
						    'F1'   : fbeta_score(y_te, y_pred_os, beta=1),
							'F2'   : fbeta_score(y_te, y_pred_os, beta=2),
							'TPR'  : recall_score(y_te, y_pred_os),
							'TNR'  : recall_score(y_te, y_pred_os, pos_label=0),
							'Prec' : precision_score(y_te, y_pred_os)}}


### Oversampling ratio and EMCO's gamma:
BALANCE_RATIO = 0.2
emco_gamma    = 0.1
cutoff = 0.015

### Load fastText word embeddings:
fastvecs = load_vectors('../wiki-news-300d-1M/wiki-news-300d-1M.vec', N=1000000)

### Select only categories that have lower frequency than 0.75 times the balance ratio:
categories = []
for label in reuters.categories():
	minority_instances = sum(['training' in index for index in reuters.fileids(label)])
	if (minority_instances / 7769) < 0.75*BALANCE_RATIO:
		categories.append(label)

### Initialize the training and test sets:
_, tr_docs, _, train_labels, te_docs, test_labels = initialize_data(
	   stopwords=[],
	   train_min_df=0,
	   only_headlines=True)

train_data = [' '.join(doc) for doc in tr_docs]
test_data  = [' '.join(doc) for doc in te_docs]

### Initialize the vocabulary and embedding matrix with lowercase fastText dictionary:
vocabulary_index = {}
embedding_index  = {}
vocabulary_index['<pad>'] = 0
embedding_index[0] = np.zeros(300)
idx = 1
for token in fastvecs:
	if token.islower():
		vocabulary_index[token] = idx
		embedding_index[idx]    = fastvecs[token]
		idx += 1
embedding_matrix = np.zeros((idx, 300))
for i in embedding_index:
	embedding_matrix[i] = embedding_index[i]

### Fit a vectorizer pipeline (be defaul converts tokens to lowercase):
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary_index))]).fit(train_data)

### Save some memory:
fastvecs        = {}
embedding_index = {}
data_train      = []
data_test       = []

### Iterate over the categories and save results:
category_results_insample  = {}
category_results_outsample = {}
category_frequencies       = {}
for cat_n, category in enumerate(categories):
	
	category_results_insample[category]  = {}
	category_results_outsample[category] = {}
	
	print()
	print(f'Category: {cat_n+1}/{len(categories)} {category}, {datetime.now()}')
	print()
	
	### Tokenizer that converts tokens to lowercase:
	tokenizer = pipe['count'].build_analyzer()
	
	### Initialize training set:
	X_train = []
	y_train = []
	for doc, y in zip(train_data, train_labels):
	    x = []
	    for token in tokenizer(doc):
	        if token in vocabulary_index:
	            x.append(vocabulary_index[token])
	    if len(x) > 0:
	        X_train.append(x)
	        y_train.append(1*(category in y))
	y_train = np.array(y_train)
	train_lengths = np.array([len(x) for x in X_train])
	category_frequencies[category] = y_train.mean()
	
	### Initialize test set:
	X_test = []
	y_test = []
	for doc, y in zip(test_data, test_labels):
	    x = []
	    for token in tokenizer(doc):
	        if token in vocabulary_index:
	            x.append(vocabulary_index[token])
	    if len(x) > 0:
	        X_test.append(x)
	        y_test.append(1*(category in y))
	y_test = np.array(y_test)
	
	### Truncate and pad the sequences:
	truncate_n = int(np.quantile(train_lengths, 0.75))
	X_pad = np.zeros((len(X_train), truncate_n), dtype=int)
	for i in range(len(X_train)):
	    padded_x = np.zeros(truncate_n)
	    for j in range(min(truncate_n, len(X_train[i]))):
	        padded_x[j] = X_train[i][j]
	    X_pad[i] = padded_x	    
	X_pad_test = np.zeros((len(X_test), truncate_n), dtype=int)
	for i in range(len(X_test)):
	    padded_x = np.zeros(truncate_n)
	    for j in range(min(truncate_n, len(X_test[i]))):
	        padded_x[j] = X_test[i][j]
	    X_pad_test[i] = padded_x
		
	### EMCO:
	emco = EMCO(gamma=emco_gamma)
	# Note that X_train is in a token-index form but that doesn't affect EMCO:
	emco.fit(X_train, y_train, pos_class=1)
	emco_docs, y_emco = emco.sample(s_ratio=BALANCE_RATIO, length='auto', complete=True)
	X_pad_emco = np.zeros((len(emco_docs), truncate_n), dtype=int)
	for i in range(len(emco_docs)):
	    padded_x = np.zeros(truncate_n)
	    for j in range(min(truncate_n, len(emco_docs[i]))):
	        padded_x[j] = emco_docs[i][j]
	    X_pad_emco[i] = padded_x
		
	### EDA:
	words = list(vocabulary_index.keys())
	edaos = EDAOS(s_ratio=BALANCE_RATIO)
	edadocs, y_eda = edaos.sample([' '.join([words[t] for t in doc]) for doc in X_train],
							      y_train, stem=False)
	X_pad_eda = np.zeros((len(edadocs), truncate_n), dtype=int)
	for i in range(len(edadocs)):
		padded_x = np.zeros(truncate_n)
		### Skip tokens that are out of fastText vocaulary:
		doc = []
		for t in tokenizer(edadocs[i]):
			if t in vocabulary_index:
				doc.append(vocabulary_index[t])
		for j in range(min(truncate_n, len(doc))):
			padded_x[j] = doc[j]
		X_pad_eda[i] = padded_x
		
	### Evaluate:
	for method, X_tr, y_tr, in zip(['RNN', 'EDA', 'EMCO'],
								   [X_pad, X_pad_eda, X_pad_emco],
								   [y_train, y_eda, y_emco]):
		result = test(X_tr, y_tr, X_pad_test, y_test,
				embedding_matrix, truncate_n, X_pad, y_train)
		category_results_insample[category][method]  = result['in-sample']
		category_results_outsample[category][method] = result['out-sample']
		
### Collect the results:
vlf_averages = []
lf_averages  = []
for category_results in [category_results_insample, category_results_outsample]:
	results_vlf = {'Bacc' : {},
				   'F1'   : {},
				   'F2'   : {},
				   'TPR'  : {},
				   'TNR'  : {},
				   'Prec' : {}}
	results_lf  = {'Bacc' : {},
				   'F1'   : {},
				   'F2'   : {},
				   'TPR'  : {},
				   'TNR'  : {},
				   'Prec' : {}}
	for metric in results_vlf:
		results_vlf[metric]['RNN']  = []
		results_vlf[metric]['EDA']  = []
		results_vlf[metric]['EMCO'] = []
		results_lf[metric]['RNN']   = []
		results_lf[metric]['EDA']   = []
		results_lf[metric]['EMCO']  = []
		
	for cat in category_results:
		for method in category_results[cat]:
			for metric in category_results[cat][method]:
				if category_frequencies[cat] < cutoff:
					results_vlf[metric][method].append(category_results[cat][method][metric])
				else:
					results_lf[metric][method].append(category_results[cat][method][metric])
		
	average_results_vlf = pd.DataFrame()
	average_results_lf  = pd.DataFrame()
	for metric in results_vlf:
		average_results_vlf[metric] = [
			np.array(results_vlf[metric][method]).mean() for method in results_vlf[metric]]
	for metric in results_lf:
		average_results_lf[metric] = [
			np.array(results_lf[metric][method]).mean() for method in results_lf[metric]]
	average_results_vlf.index = [method for method in results_vlf['Bacc']]
	average_results_lf.index  = [method for method in results_lf['Bacc']]
	vlf_averages.append(average_results_vlf)
	lf_averages.append(average_results_lf)

print()
print('Out-of-sample average results, VLF categories:')
print(vlf_averages[1])
print()
print('Out-of-sample average results, LF categories:')
print(lf_averages[1])
