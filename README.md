Machine learning applied to NLP without deep learning

The purpose of this respository is use machine learning to solve NLP problem without involving deep learning releted technology. So only traditional machine learning methods will be used here. It will include Naive Bayes, Random Forest,GBDT and so on.

We will first use Naive Bayes to do binary classification, which is to classify a sentence releted to be a 'theft' or not.

Check naive_bayesian_binary_classification.py for more detail.

Then we use Naive Bayes to do multiple label classification, which we use it to detect intent of user from spoken dialougue. In this task there are more than 100 different intents, our job is to classify the sentence to a specific label correctly. Traditionally, only the current sentence will be used. However, we also try to model this task with context. It means not only current sentence will be used, previous interactive will be also used.

Check naive_bayesian_multi_labels.py
