# Thesis
Digital sovereign communities are facing a fundamental conflict between their desire to include all members in the decision making processes and the limited time and attention that are at the disposal of the community members; i.e., there is a need for so-called Decentralized Governance at Scale (DEGAS). Here we investigate a combination of two techniques for DEGAS, namely Natural Language Processing (NLP) and sampling. 
The focus of this research is on the evaluation of a combination of machine learning techniques and sampling-based solutions. Our approach is essentially as follows: we use supervised learning in machine learning with NLP (The first step was to implement a simple NLP model which would convert raw data into a matrix of TF-IDF features and then improve the simple model using a BERT model) to develop a trained model that is able to take a textual governance proposal to change the status quo and estimate the probability that the community would accept the proposal if directly voted on it; then, based on the estimation of such a trained model, we select a numerical value representing the fraction of the community that we then ask to actively vote on the proposal. We then take the sampled votes and use majority voting to decide on the proposal. At a glimpse, our solution is composed of an prediction module -- that predicts an acceptance probability for a given proposal; a sampling module -- that, based on the prediction of the prediction module, selects a vote sample; and a decision module that decides on the fate of the proposal based on the vote sample. 
Our aim is to evaluate the above-mentioned architecture as a solution approach to DEGAS. To this end, we develop several variants of the architecture and evaluate their performance with respect to three data sources  :data obtained from Kaggle (a popular venue for ML-related data); data gathered from Snapshot (a popular voting application for digital communities); and data sourced from DeepDAO (an  analytics and information gathering platform for the DAO ecosystem).


In this github folder, there are the central code files:

1. Data files that organize relevant data as data frames:
- KaggleData.py
- SnapshotData.py
- DeepDAOData.py
2. Creating training and test sets based on the data:
- DataExploration.py
3. Data as features (TF-IDF or BERT):
- NLP_TFIDF.py
- BERTmodel.py
4. Classifiers using the TF-IDF features:
- Models.py
5. Simulating voting by creating votes:
- GenerateVotes.py
6. Sampling:
- Sampling.py
7. System evaluation:
- SystemEvaluation.py

Notes:
1. Running the main file (main.py) is all that is required.
