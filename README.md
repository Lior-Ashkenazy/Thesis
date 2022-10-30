# Thesis
Digital sovereign communities are facing a fundamental conflict
between their desire to include all members in the decision making
processes and the limited time and attention that are at the disposal of
the community members; i.e., there is a need for so-called Decentralized
Governance at Scale (DEGAS). Here we investigate a combination of two
techniques for DEGAS, namely Natural Language Processing (NLP) and
sampling. Essentially, we propose a system in which each governance proposal
to change the status quo is first sent to a trained NLP model, which
estimates the probability that the proposal would pass if all community
members directly vote on it; then, based on such an estimation, a population
sample of a certain size is being selected and the proposal is decided
upon by taking the sample majority. To this end, we develop several variants of the architecture and evaluate
their performance with respect to three data sources: data obtained from Kaggle
(a popular venue for ML-related data); data gathered from Snapshot (a popular
voting application for digital communities); and data sourced from Deep DAO (an
analytics and information gathering platform for the DAO ecosystem).

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
4. Classifiers:
- Models.py
5. Simulating voting by creating votes:
- GenerateVotes.py
6. Sampling:
- Sampling.py
7. System evaluation:
- SystemEvaluation.py

Notes:
1. Running the main file (main.py) is all that is required
