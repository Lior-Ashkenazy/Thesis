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
upon by taking the sample majority. We develop several concrete algorithms
following this scheme and evaluate them using data from several
Decentralized Autonomous Organizations (DAOs).

In this github folder, there are the central code files. Files order:

Split samples ID to train/test: split_train_test_IDs_for_autoencoder.py
Due to data size, split the data into samples chuncks: Split_each_chr_to_chunks.py
Train dimensionality reduction: 3.1. For PCA: PCA_dimension_reduction.py 3.2. For Autoencoder: Autoencoder_for_each_chr.py
Preprocessing on the covariate matrix: covariate_arrangement.py
Predict the variables after dimensionality reduction (just for autoencoder): snp_prediction_dimension_reduction.py
Scale the variables after dimensionality reduction: scale_genes.py
Join variables from all chromosomes after dimensionality reduction + covarivate matrix: join_all_chr_after_dr.py
Match the dataframe (the result of section 7) to specific phenotype: match_pheno_IDs.py
Train perdiction model: 9.1. For height phenotype: Height_prediction.py 9.2. For hypertension phenotype: hypertension_prediction.py
