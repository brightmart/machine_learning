# machine_learning
Machine learning applied to NLP without deep learning

The purpose of this resposity is to use machine learning to solve NLP problem without using deep learning. So only traditional machine learning methods will be used here. It will include Naive Bayes, Random Forest,GBDT and so on.

We will first use Naive Bayes to do binary classification, that is to classify a sentence to be a theft or not.
check naive_bayesian_binary_classification.py for more detail.

Then we use Naive Bayes to do multiple label classification, which we use it to detect intent of user. it has more than 100 different intents, our job is to correct classify the sentence from user to a specific label. traditional, only the current sentence will be used. however, we also try to model this task with context. It means not only current sentence will be used, previous interactive will be also used.
check naive_bayesian_multi_labels.py
