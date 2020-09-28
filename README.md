# Gaussian process preference learning (GPPL) and CrowdGPPL

The repository provides an implementation of two Bayesian pairwise preference learning
methods, GPPL and crowdGPPL, for ranking, scoring items and predicting pairwise labels.
GPPL does not distinguish between different sources of labels, while
crowdGPPL models each individual labeller in a crowd of labellers. 
CrowdGPPL can make 
predictions of individual preferences as well as infer 
the consensus of a crowd,  
by combining GPPL with Bayesian matrix
factorisation.
GPPL and crowdGPPL are both implemented using using stochastic variational inference.

The master branch is intended to house the latest version of the GPPL implementation. 
There are two associated papers, which each use code in a separate branch:
1. Finding convincing arguments using scalable bayesian preference learning, Simpson and Gurevych, TACL (2018). Please see [tacl2018](https://github.com/UKPLab/tacl2018-preference-convincing/tree/tacl2018) branch.
1. Scalable Bayesian preference learning, Simpson and Gurevych, Machine Learning (2020). Please see [crowdGPPL](https://github.com/UKPLab/tacl2018-preference-convincing/tree/crowdGPPL) branch. This paper contains the details of the machine learning approaches.
Please cite one of these two papers when using these methods.

**Contact person:** Edwin Simpson, simpson@ukp.informatik.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be)
or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background
details on the respective publication.

## Project Structure

* data -- a folder containing small data files + default place to generate dataset files for the experiments
* documents -- sources for the paper
* error_analysis -- working data files for error analysis
* python/analysis -- experiment code
* python/analysis/habernal_comparison -- experiment code for use with the datasets discussed in paper, originally obtained from
https://github.com/UKPLab/acl2016-convincing-arguments
* python/models -- the implementation of the GPPL method
* python/test -- some simple test scripts for the GPPL methods
* results -- an output directory for storing results

## Requirements

Dependencies for just using the gp_pref_learning model:

   * scikit-learn>=0.18.1
   * scipy>=0.19.0
   * numpy>=1.12.1

For running the experiments, please see the requirements.txt for further dependencies. 

* Python 3
* virtualenv (to install requirements) 
* The required packages are listed in requirements.txt. 
You can install them using pip install -r requirements.txt.
* Maven -- check if you have the command line program 'mvn' -- required to extract the linguistic features from our 
experimental argument convincingness datasets. You can skip this if you are not re-running our experiments or training a model on 
UKPConvArg*** datasets.
* Java JDK 1.8. Newer and older versions may cause problems with preprocessing. If you
have other versions of JDK installed, you need to set the environment 
variable JAVA_HOME to point to the 1.8 version, 
e.g. export JAVA_HOME=/Library/Java/JavaVirtualMachines/jdk1.8.0_261.jdk/Contents/Home/jre/ .

Set up your virtual environment:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

## How to run the experiments

Please see either the [crowdGPPL](https://github.com/UKPLab/tacl2018-preference-convincing/tree/crowdGPPL)
 or [tacl2018](https://github.com/UKPLab/tacl2018-preference-convincing/tree/tacl2018) branch.

## Template for running on a new dataset with Ling+Glove feature sets

You can use the following script as a template for running GPPL on new datasets 
using the same feature sets as in our paper. If you have another method for
extracting features from your datasets, you may with to skip this example
and look at 'how to use the GPPL implementation'.


```
python ./python/example_use.py
```

The script will train a convincingness model on the UKPConvArgStrict data, then
run it to score arguments in a new dataset. 

Pre-requisite: this script assumes you have carried out step 0 above and 
run "python/analysis/habernal_comparison/run_preprocessing.py" to extract the linguistic features.

## How to use the GPPL implementation

The preference learning method is implemented by the gp_pref_learning class in
python/models/gp_pref_learning.py. 
The template for training and prediction described above contains an example of how
to use the class, but also contains code for extracting our linguistic features and Glove embeddings,
which you may not need.
You can run a simpler example that generates
synthetic data by running python/test/gp_pref_learning_test.py.

The preference learning model and algorithm are implemented by the class 
GPPrefLearning inside python/test/gp_pref_learning_test.py. 
The important methods in this class are listed below; please look at the 
docstrings for these methods for more details:
   * The constructor: set the model hyperparameters
   * fit(): train the model
   * predict(): predict pairwise labels for new pairs
   * predict_f(): predict scores for a set of items given their features, which can be used to rank the items.

## Example usage of GPPL

In this example, we assume that you have a file, 'items.csv', that contains the feature data for some
documents or other items that you wish to model.

Start by loading in some data from a CSV file using numpy:
~~~
item_data = np.genfromtxt('./data/items.csv', dtype=float, delimiter=',', skip_header=1)
item_ids = item_data[:, 0].astype(int) # the first column contains item IDs
item_feats = item_data[:, 1:] # the remaining columns contain item features
~~~

Now, load the pairwise preference data:
~~~
pair_data = np.genfromtxt('./data/pairwise_prefs.csv', dtype=float, delimiter=',', skip_header=1)
# the first column contains IDs of the first items in the pairs. Map these to indices into the item_ids matrix.
items_1_idxs = np.array([np.argwhere(item_ids==iid)[0][0] for iid in pair_data[:, 1].astype(int)])
# the second column contains IDs of the second items in the pairs. Map these to indices into the item_ids matrix.
items_2_idxs = np.array([np.argwhere(item_ids==iid)[0][0] for iid in pair_data[:, 2].astype(int)])
# third column contains preference labels in binary format (1 indicates the first item is preferred, 0 indicates the second item is preferred)
prefs = pair_data[:, 2] 
~~~

Construct a GPPrefLearning object. The following values are reasonable defaults
for the applications we have tried so far:
~~~
from gp_pref_learning import *
from gp_classifier_vb import compute_median_lengthscales # use this function to set sensible values for the lengthscale hyperparameters
model = GPPrefLearning(item_feats.shape[1], shape_s0=2, rate_s0=200, 
                        ls_initial=compute_median_lengthscales(summary_matrix) )
~~~

Now train the object given the data:
~~~
model.fit(items_1_idxs, items_2_idxs, item_feats, prefs, optimize=False)
~~~

Given the fitted model, we can now make predictions about any items given their 
features. These may be new, previously unseen items, or items that were used in 
training. To obtain a score for each item, e.g. to be used for ranking items,
call the following:
~~~
model.predict_f(test_item_feats)
~~~

You can also predict pairwise labels for any items given their features:
~~~
model.predict(test_item_feats, test_items_1_idxs, test_items_2_idxs)
~~~
Here, the test_item_feats object is a matrix where each row is a feature vector
of an item. The test_items_1_idxs and test_items_2_idxs objects are vectors 
(lists or 1-dimensional numpy arrays) containing indices into test_item_feats
of items you wish to compare.

## Setting priors

For setting the prior mean function:
   * when calling fit(), pass in a vector mu0 that is the same size as item_features. Each entry of mu0 is the prior
preference function mean for the corresponding item in item_features
   * when calling predict_f() to predict the score for test items, or 
when calling predict() to predict pairwise labels, 
the argument mu0_output should be used to provide the preference function mean 
for the corresponding items in items_coords, i.e. each item indexed in 
items_coords should have a prior mean value in mu0_output

For the prior precision:
   * The prior precision controls how much the preference learning function can move away from your prior, mu0
   * High precision means that your prior mu0 is very strong, so that when you train the model using fit(), the values
 will not move far from mu0
   * Low precision means that the prior mu0 is very weak and the preference function values estimated using fit() then
 predict_f() will be larger
   * The prior precision itself has a prior distribution, a Gamma distribution with parameters shape_s0 and rate_s0
   * These hyperparameters are equivalent to pre-training the model on n data points with variance v, so you can set them
 as follows...  
   * shape_s0 = n / 2.0
   * rate_s0 = v * n / 2.0
   
## How to use the crowdGPPL implementation

These are guidelines for using crowdGPPL outside of our experimental setup. 
The method is implemented by the CollabPrefLearningSVI class in
python/models/collab_pref_learning_svi.py. 
You can run a simple example using synthetic data by running python/test/collab_pref_learning_test.py.

The important methods in the CollabPrefLearningSVI class are:
   * The constructor: set the model hyperparameters
   * fit(): train the model
   * predict(): predict pairwise labels for new pairs
   * predict_f(): predict scores for a set of items given their features, which can be used to rank the items.
The example below shows how to use these methods.

## Example usage of GPPL

In this example, we assume that you have a file, 'items.csv', that contains the feature data for some
items that you wish to model. In this file, each row corresponds to an item, and each column to a feature.

Start by loading in some data from a CSV file using numpy:
~~~
item_data = np.genfromtxt('./data/items.csv', dtype=float, delimiter=',', skip_header=1)
item_ids = item_data[:, 0].astype(int) # the first column contains item IDs
item_feats = item_data[:, 1:] # the remaining columns contain item features
~~~

You may also have similar data for a set of users -- this is optional. If available,
we load the user features in the same manner:
~~~
user_data = np.genfromtxt('./data/users.csv', dtype=float, delimiter=',', skip_header=1)
user_ids = item_data[:, 0].astype(int) # the first column contains item IDs
user_feats = item_data[:, 1:] # the remaining columns contain item features
~~~

Now, load some pairwise preference data:
~~~
pair_data = np.genfromtxt('./data/pairwise_prefs.csv', dtype=float, delimiter=',', skip_header=1)
# the first column contains IDs of the first items in the pairs. Map these to indices into the item_ids matrix.
items_1_idxs = np.array([np.argwhere(item_ids==iid)[0][0] for iid in pair_data[:, 1].astype(int)])
# the second column contains IDs of the second items in the pairs. Map these to indices into the item_ids matrix.
items_2_idxs = np.array([np.argwhere(item_ids==iid)[0][0] for iid in pair_data[:, 2].astype(int)])
# third column contains the IDs of the users who provided each preference label
users = pair_data[:, 2]
# fourth column contains preference labels in binary format (1 indicates the first item is preferred, 0 indicates the second item is preferred)
prefs = pair_data[:, 3] 
~~~

Construct a CollabPrefLearningSVI object. The following values are reasonable defaults
for the applications we have tried so far:
~~~
from collab_pref_learning_svi import CollabPrefLearningSVI
model = CollabPrefLearningSVI(item_feats.shape[1], user_feats.shape[1], shape_s0=2, rate_s0=200, 
                              use_lb=True, use_common_mean_t=True, ls=None)
~~~

Now train the object given the data:
~~~
model.fit(users, items_1_idxs, items_2_idxs, item_feats, prefs, user_feats, optimize=False, use_median_ls=True)
~~~

Given the fitted model, we can now make predictions about any combination of
users and items given their features. 
These may be new, previously unseen items, or items that were used in 
training. To obtain a score for each item for a set of test users, e.g. for ranking items,
call the following:
~~~
model.predict_f(test_item_feats, test_user_feats)
~~~
If the users wer already seen during training, you can instead pass in the list of userids you want to make predictions for:
~~~
model.predict_f(test_item_feats, userids)
~~~

To get the consensus prediction for the whole crowd, rather
than a specific user:
~~~
model.predict_t(test_item_feats)
~~~
Or, if you want to predict the consensus scores for the items seen during training:
~~~
model.predict_t()
~~~

You can also predict pairwise labels for a specific user for any items given their features:
~~~
model.predict(test_users, test_items_1_idxs, test_items_2_idxs, test_item_feats, test_user_feats)
~~~
Here, the test_item_feats object is a matrix where each row is a feature vector
of an item. The test_items_1_idxs and test_items_2_idxs objects are vectors 
(lists or 1-dimensional numpy arrays) containing indices into test_item_feats
of items you wish to compare.
If you wish to make predictions on the items seen during training, set the item_feats object to None:
~~~
model.predict(test_users, test_items_1_idxs, test_items_2_idxs, None, test_user_feats)
~~~

For the consensus pairwise labels (i.e. the crowd's decision) for a set of test items:
~~~
model.predict_common(test_item_feats, test_items_1_idxs, test_items_2_idxs)
~~~
For consensus pairwise labels on the training items:
~~~
model.predict_common(None, test_items_1_idxs, test_items_2_idxs)
~~~

## Setting priors

For the inverse scales of the latent preference functions, s and $\sigma$:
   * The inverse scale of the latent preference functions reflects the noise at each observation
   * High values correspond to noisy observations or inactive latent components and causes the latent preference functions to be close to 0
   * Low precision means that the scale of the latent prefernece function will be larger
   * In our implementation, s and $\sigma$ are learned from the data but share a common prior distribution: a Gamma distribution with parameters shape_s0 and rate_s0
   * The mean of the inverse scale is shape_s0 / rate_s0, so for noisy observations, it makes sense to increase decrease rate_s0,
and for stronger observations, to increase rate_s0
   * These hyperparameters are equivalent to pre-training the model on n data points with variance v:
      * shape_s0 = n / 2.0
      * rate_s0 = v * n / 2.0
      * Hence shape_s0 reflects the confidence of the prior
      * Setting shape_s0=2.0 means a prior that is very weakly informative, so may be a good choice if you are uncertain what value to use

The values of shape_s0 and rate_s0 could be tuned against a validation set, or tuned by selecting the values that maximise the value of model.lowerbound().
