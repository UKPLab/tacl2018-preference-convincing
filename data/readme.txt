This directory will contain one folder for each of the datasets mentioned in table 1 in the document. 
Each directory contains the labels and text needed for these datasets.

The embeddings are stored in vocabulary.embeddings.all.pkl.bz2. 

The linguistic features (as used for the SVM experiments in Habernal & Gurevych 2016) are saved
inside the child directory 'lingdata'. Since the complete set of pairs is provided by UKPConvArg1All
and the other datasets use subsets of this with the same argument IDs, we run the pipeline once 
and look up the linguistic features given the argument text ID. When running the experiments, we
load the SVM lib format text features and identify the argument pairs as a string beginning with
"_argxxxxxx_argxxxxxx_ax"; since we only use this for text features, we just ignore duplicate pairs
due to different crowdworkers labelling the same two arguments.
