from nltk.corpus import reuters
from nltk.tokenize import RegexpTokenizer
import snowballstemmer
from sklearn.svm import LinearSVC
from sklearn import metrics


def preprocess(data, min_df=1, min_len=1, stopwords=[], sep=None, stem=True):
	
	'''
	data            : array or list of documents
	min_df, min_len : word is included in the vocabulary only if its frequency in data
			  is strictly greater than min_df and its length is strictly greater 
			  than min_len
	stopwords       : list of stopwords to be removed from the vocabulary
	sep             : symbol separating words; if None is given, the documents are
			  tokenized with RegexpTokenizer(r'[a-zA-Z]+')
	stem            : whether to (snowball) stem the words
	---
	returns         : list of preprocessed documents as lists of lower case tokens
			  and dictionaries of {word : index} and {word : frequency}
	'''
		
	stemmer = snowballstemmer.stemmer('english')
	
	docs       = [] # Preprocessed documents
	infrequent = {} # Encountered words whose frequencies are not yet higher than min_df
	vocabulary = {} # Dictionary of words and their indices
	word_count = {} # Dictionary of words and their frequencies
	
	for i in range(len(data)):
		
		if sep:
			text = data[i].split(sep)
		else:
			text = RegexpTokenizer(r'[a-zA-Z]+').tokenize(data[i])
		
		clean_text = []
		if stem:
			for w in text:
				token = stemmer.stemWord(w.lower())
				if len(token) > min_len and token not in stopwords:
					clean_text.append(token)
		else:
			for w in text:
				token = w.lower()
				if len(token) > min_len and token not in stopwords:
					clean_text.append(token)
		
		if len(clean_text) == 0:
			clean_text = ['<empty>']
		
		for w in clean_text:
			if w in vocabulary.keys():
				word_count[w] += 1
			elif min_df == 0:
				vocabulary[w] = len(vocabulary)
				word_count[w] = 1
			elif w in infrequent.keys():
				if infrequent[w] == min_df:
					vocabulary[w] = len(vocabulary)
					word_count[w] = min_df + 1
				else:
					infrequent[w] += 1
			else:
				infrequent[w] = 1
		
		docs.append(clean_text)
	
	return [[w for w in d if w in vocabulary] for d in docs], vocabulary, word_count

	
def initialize_data(pos_class, stopwords=[], train_min_df=1, only_headlines=True):
	
	'''
	Initialize Reuters-21578 dataset with ModApte test-training split for binary
	classification based on the given the positive class. By default, considers
 	only headlines, and if only_headlines=False, considers full articles.
	---
	returns:
		train_docs       : preprocessed training documents
		train_vocabulary : training vocabulary
		y_train          : binary training labels
		test_docs        : preprocessed test documents (where min_df=0)
		y_test           : binary test labels
	'''
	
	train   = [] # unpreprocessed training documents
	test    = [] # unpreprocessed test documents
	y_train = [] # binary training labels given the category pos_class
	y_test  = [] # binary test labels given the category pos_class
	
	for file in reuters.fileids():
		if file.split('/')[0] == 'test':
			if only_headlines:
				test.append(reuters.raw(file).split('\n')[0])
			else:
				test.append(reuters.raw(file))
			y_test.append(int(pos_class in reuters.categories(file)))
		else:
			if only_headlines:
				train.append(reuters.raw(file).split('\n')[0])
			else:
				train.append(reuters.raw(file))
			y_train.append(int(pos_class in reuters.categories(file)))
	
	train_docs, train_vocabulary, _ = preprocess(
		train, min_df=train_min_df, stopwords=stopwords, stem=True)
	test_docs, _, _ = preprocess(test, min_df=0, stem=True)
	
	return train_docs, train_vocabulary, y_train, test_docs, y_test
		

def test(X, y, X_test, y_test, tol=1e-3):
	
	'''
	X      : (oversampled) training data
	y      : (oversampled) binary training labels
	X_test : test data
	y_test : binary test labels
	tol    : tolerance for stopping criteria when fitting SVM
	---
	Performs classification tests with the given data. SVM with a linear kernel is used
	in classification. Returns balanced accuracy, TPR, TNR, precision, F1 and F2 -scores
	'''
	
	### Default settings:
	### penalty = l2, dual = True, C = 1.0
	clf = LinearSVC(loss='hinge', max_iter=100000, tol=tol)
	clf.fit(X, y)
	predicted = clf.predict(X_test)
	
	bAcc = metrics.balanced_accuracy_score(y_test, predicted)
	TPR  = metrics.recall_score(y_test, predicted, pos_label=1)
	TNR  = metrics.recall_score(y_test, predicted, pos_label=0)
	prec = metrics.precision_score(y_test, predicted, pos_label=1, zero_division=0.0)
	f_1  = metrics.f1_score(y_test, predicted, pos_label=1)
	f_2  = metrics.fbeta_score(y_test, predicted, beta=2, pos_label=1)
	
	return bAcc, TPR, TNR, prec, f_1, f_2
