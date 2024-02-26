import numpy as np
import random
from scipy.sparse import dok_matrix


'''
 ****************************************************************
 * Implementation of the Extrapolated Markov Chain Oversampling *
 * (EMCO) method for imbalanced text classification developed   *
 * by Avela, A. and Ilmonen, P. (2024)                          *
 *                                                              *
 * EMCO is based on an assumption that the sequential structure *
 * of text can be partly learned from the majority class. Thus, *
 * oversampling with EMCO will allow the minority feature space *
 * to expand, which helps generalizing the minority class.      *
 ****************************************************************
'''


class ExtrapolatedMarkovChainOversampling:
	
	
	def __init__(self, gamma=1, random_state=None):
		
		### gamma (float) : Weight parameter for transitions in majority documents
		self.gamma = gamma
		### Flag for whether the transition matrix has been fitted:
		self.fitted = False
		### If given, set the random state:
		self.rnd = random.Random(random_state)
		
		
	def __preprocess_data(self):
		
		### Collect vocabularies:
		### Minority vocabulary distribution:
		min_counts = {}
		for doc in self.min_data:
			for w in doc:
				if w in min_counts.keys():
					min_counts[w] += 1
				else:
					min_counts[w] = 1
		### List of majority-only vocabulary:
		maj_words = []
		for doc in self.maj_data:
			for w in doc:
				if w not in min_counts:
					if w not in maj_words:
						maj_words.append(w)
		
		### Arrange vocabulary as [ min_vocabulary, maj-only_vocabulary, <STOP> ]
		### Vocabulary is a dict in a form of {word : index}
		words = list(min_counts.keys()).copy()
		for t in maj_words:
			words.append(t)	
		vocabulary = dict([[word,i] for i,word in enumerate(words)])
		vocabulary['<STOP>'] = len(vocabulary)
		
		self.min_vocabulary = list(min_counts.keys()) # Minority vocabulary
		self.vocabulary		= vocabulary              # Total vocabulary
		self.distinct_words = list(vocabulary.keys()) # List of words in vocabulary
		
		### Minority document length distribution
		self.length_distribution = [len(doc) for doc in self.min_data]
				
		### Word distribution in the minority documents:
		self.min_dist = np.array(list(min_counts.values()))
		
		
	def __add(self, first, second, value=1):
		
		### Add given value to the transition count from word 'first' to word
		### 'second' as well as to the row sum corresponding to the first word:
		self.P[self.vocabulary[first], self.vocabulary[second]] += value
		self.row_sums[self.vocabulary[first]] += value
		

	def __fit_transition_probabilities(self):
		
		'''
		vocabulary = ( {min_voc}, {maj-only_voc}, <STOP> )
		P : Unnormalized Markov probability matrix, dim : [ vocabulary x vocabulary ]
		row_sums : The row sums are used when sampling words, see __draw_next()
		'''
		
		### Initialize the P matrix and row sum vector:
		self.P = dok_matrix(
			(len(self.distinct_words), len(self.distinct_words)), dtype=np.float32)
		self.row_sums = np.zeros(len(self.distinct_words))
				
		### Minority transitions:
		for doc in self.min_data:
			self.__add('<STOP>', doc[0])
			for i in range(len(doc)-1):
				if doc[i] != doc[i+1]:
					self.__add(doc[i], doc[i+1])
			self.__add(doc[-1], '<STOP>')
		
		### gamma-weighted majority transitions (from common words to any words):
		for doc in self.maj_data:
			for i in range(len(doc)-1):
				if doc[i] in self.min_vocabulary:
					if doc[i] != doc[i+1]:
						self.__add(doc[i], doc[i+1], value=self.gamma)
		
		### Save items {row -> (id,weight)} for efficient sampling:
		self.items = {}
		for i in range(len(self.min_vocabulary)):
			self.items[i] = self.P[i,:].items()
		self.items[self.vocabulary['<STOP>']] = self.P[self.vocabulary['<STOP>'],:].items()
		### Save some memory:
		self.P = []
		self.min_data  = []
		self.maj_data  = []
		
		
	def __draw_next(self, current):
		
		if current in self.min_vocabulary or current == '<STOP>':
			### Draw a uniformly distributed random number from [0, row_sum[current]]
			u = self.rnd.uniform(0, self.row_sums[self.vocabulary[current]])
			s = 0
			for idx, weight in self.items[self.vocabulary[current]]:
				s += weight
				if s >= u:
					break
			return self.distinct_words[idx[1]]
		else:
			### Transitions from majority-only are based on the minority distribution:
			u = self.rnd.uniform(0, self.min_dist.sum())
			s = 0
			for idx, weight in enumerate(self.min_dist):
				### Vocabulary is arranged s.t. minority words are the first indices
				s += weight
				if s >= u:
					break
			return self.distinct_words[idx]
		
		
	def __chain(self, length):
		
		'''
		Generate a synthetic document as a Markov chain based on the estimated
		transition probability matrix P.
		---
		length : length of the generated chain; if none is given, the length will
			 be drawn from the minority document length distribution
		'''
		
		chain   = []
		current = '<STOP>'
		
		if type(length) == int:
			L = length
		else:
			L = self.length_distribution[int(self.rnd.uniform(
				0, len(self.length_distribution)))]
		
		i = 0
		while i < L:
			next_word = self.__draw_next(current)
			if next_word != '<STOP>':
				i += 1
				chain.append(next_word)
			current = next_word
		
		return chain
		
		
	def fit(self, data, y, pos_class=1):
		
		'''
		Divides the data into minority and majority documents, discards empty rows
		from estimation, and estimates the transition probability matrix. Multiclass
		data sets are modified to binary, where observations in the positive class
		represent minority class, and all the rest of the observations majority class
		---
		data      : list of documents (where each document is a list of tokens)
		y         : list of (binary or multiclass) labels
		pos_class : positive (minority) class in data
		'''
		
		assert len(data) == len(y), "Data and y must have the same length"
		assert pos_class in y, "There must at least one observation in the positive class"
		
		### Separate minority and majority data and discard empty documents:
		min_data = np.array([doc for doc,label in zip(data,y) if
					   label==pos_class and len(doc) > 0])
		maj_data = np.array([doc for doc,label in zip(data,y) if
					   label!=pos_class and len(doc) > 0])
		
		self.data      = data
		self.y         = np.array(np.array(y) == pos_class, dtype=int)
		self.min_data  = min_data
		self.maj_data  = maj_data
		
		self.__preprocess_data()
		self.__fit_transition_probabilities()
		self.fitted = True
					
			
	def sample(self, s_ratio=0.5, length='auto', complete=False):
		
		'''
		Generates a synthetic sample using the estimated transition probabilities. Note
		that the labels are set to binary (1 and 0) based on the positive class even if
		the labels given for fit() would have included multiple classes.
		---
		s_ratio  : sampling ratio, i.e., the relative frequency of minority class after
			   oversampling. Must be greater than minority class frequency and
		           smaller than or equal to 0.5.
		length   : length of the synthetic documents; if 'auto' is given draws lengths
			   of the documents from the minority document length distribution
	    	complete : if True, returns the original data appended with the synthetic
			   observations, otherwise returns only the synthetic sample. Note that,
			   even though empty documents are discarded in estimation, they are
			   still included in the complete sample (and may have an effect on the
			   number of generated synthetic documents)
	    	---
		Returns  : an array of documents (each document is a list of tokens), and an array
			   of of binary labels, where minority label => 1 and majority label => 0
		'''
		
		assert self.fitted, "Run .fit() before sampling!"
		assert s_ratio>self.y.mean(), "Sampling ratio must greater than minority frequency"
		assert s_ratio<=0.5, "Sampling ratio must smaller than or equal to 0.5"
		
		### There will always be at least one synthetic document generated:
		n = max(1, int((s_ratio*len(self.y) - sum(self.y)) / (1 - s_ratio)))
		
		if complete:
			documents = list(self.data.copy())
			labels    = list(self.y.copy())
		else:
			documents = []
			labels    = []
		for i in range(n):
			documents.append(self.__chain(length))
			labels.append(1)
		return np.array(documents), np.array(labels)
