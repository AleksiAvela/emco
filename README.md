# EMCO
This repository includes a Python implementation of Extrapolated Markov Chain Oversampling (EMCO) method introduced in the submitted manuscript *Extrapolated Markov Chain Oversampling Method for Imbalanced Text Classification* and the source code of the presented experiments on Reuters-21578 corpus, HuffPost News Category Dataset, and 20 Newsgroups dataset.

Running the full experiments can take from a few hours up to tens of hours depending on the experiment set up and available computational power. For testing the method, we suggest running example.py which includes the same tests but only for one specified category of Reuters-21578 and without averaging over multiple repetitions.

For the source code and documentations of the other tested methods, please refer to: https://imbalanced-learn.org/stable/index.html and https://github.com/AlexMoreo/pydro. This repository also includes our implementation of DECOM oversampling method by Chen et al. (2011). The HuffPost News Category Dataset (Misra, 2022) is available at https://www.kaggle.com/datasets/rmisra/news-category-dataset.
