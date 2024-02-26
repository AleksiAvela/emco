import numpy as np
from scipy.sparse import dok_matrix, vstack
from sklearn.decomposition import LatentDirichletAllocation as LDA

'''
Our implementation of DECOM oversampling method by Chen et al. (2011)
based on a Latent Dirichlet Allocation (LDA) topic-model 
'''

class LDAOS:
	
	def __init__(self, n_topics=30, alpha=50/30, beta=0.01, max_iter=300):
		
		self.n_topics = n_topics
		self.alpha = alpha
		self.beta = beta
		self.max_iter = max_iter
		
		
	def fit(self, X, y, pos_class=1):
		
		lda = LDA(n_components=self.n_topics,
			      doc_topic_prior=self.alpha,
				  topic_word_prior=self.beta,
				  max_iter=self.max_iter)
		
		data = X[np.array(y)==pos_class]
		lda.fit(data)
		
		# Inferred topic and word distributions:
		topics = lda.transform(data).sum(axis=0)
		words  = lda.components_
		
		# Normalize the topic and word distributions:
		topics = topics/topics.sum()
		for i in range(self.n_topics):
			words[i,:] /= words[i,:].sum()
		
		self.topics = topics
		self.words  = words
		self.lengths = (data > 0).toarray().sum(axis=1)
		self.pos_class = pos_class
		self.X = X
		self.y = y
		
		
	def sample(self, s_ratio=0.5):
		
		n = max(1, int((s_ratio*len(self.y) - sum(self.y)) / (1 - s_ratio)))
		m = self.X.shape[1]
		
		X_os = dok_matrix((n,m))
		y_os = list(self.y)
		
		for i in range(n):
			l = np.random.choice(self.lengths)
			doc = np.zeros(m)
			for j in range(l):
				topic = np.random.choice(self.n_topics, 1, p=self.topics)[0]
				word  = np.random.choice(self.words.shape[1], 1, p=self.words[topic])[0]
				doc[word] += 1
			X_os[i] = doc
			y_os.append(self.pos_class)
		
		return vstack([self.X, X_os]), np.array(y_os)
		