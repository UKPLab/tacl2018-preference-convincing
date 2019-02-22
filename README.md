# CrowdGPPL

This branch contains an implementation of crowdGPPL, a Bayesian pairwise preference learning
method for ranking, scoring items and predicting pairwise labels, 
given preference data from a crowd of individuals. CrowdGPPL can make 
predictions of individual preferences as well as infer the consensus of a crowd.
The model combines Gaussian process preference learning and Bayesian matrix
factorisation and our implementation uses stochastic variational inference.

The crowdGPPL paper has not yet been published -- please contact the authors below for 
more information. 
For single-user GPPL, please cite:
```
@article{simpson2018finding,
  title={Finding convincing arguments using scalable bayesian preference learning},
  author={Simpson, Edwin and Gurevych, Iryna},
  journal={Transactions of the Association of Computational Linguistics},
  volume={6},
  pages={357--371},
  year={2018},
  publisher={MIT Press}
}
```

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

   * scikit-learn==0.18.1
   * scipy==0.19.0
   * numpy==1.12.1

For running the experiments, please see the requirements.txt for further dependencies. 

* Python 3
* virtualenv
* The required packages are listed in requirements.txt. You can install them using pip install -r requirements.txt
* Maven -- check if you have the command line program 'mvn' -- required to extract the linguistic features 
from the argument convincingness dataset. You can skip 
this if you are not re-running these experiments or training a model on UKPConvArg*** datasets.

## How to run the experiments

Checkout the code:
```
git clone --single-branch --branch crowdGPPL https://github.com/UKPLab/tacl2018-preference-convincing.git
```

Set up your virtual environment:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Synthetic data

The plot shown in Figure 1 (a) is generated from this script:
```
python3 ./python/analysis/simulations/synth_tests.py 0
```
This will save plots to results/multi_user_consensus. 
To run a subset of methods, comment or uncomment them from the script. 
The plots shown in the paper are tau_test.pdf (the Kendall's tau rank correlation
for test data points).

For Figure 1 (b), run:
```
python3 ./python/analysis/simulations/synth_tests.py 1
```
Plots will be saved to results/multi_user_personal.

For Figure 1(c), run:
```
python3 ./python/analysis/simulations/synth_latent_factor_tests.py
```
This will save plots to results/multi_factor_correlations_P.


### Argument convincingness

1. Extract the linguistic features from the data by running:

```
python ./python/analysis/habernal_comparison/run_preprocessing.py. 
```
The data resides in directory ./data and this path is set in ./python/analysis/data_loading.py, line 12.
The data is originally provided by https://github.com/UKPLab/acl2016-convincing-arguments, the copy is
provided here for convenience.

2. Run the consensus test (Table 2) with single-user GPPL:

```
python ./python/analysis/habernal_comparison/personalised_tests.py 7
```

This script sets some parameters for the test:
   * the choice of method
   * dataset
   * features to use with each method
Given these settings, the experiments are then implemented by ./python/analysis/habernal_comparison/tests.py.

Compute the performance metrics:
```
python ./python/analysis/habernal_comparison/personalised_metrics.py 7
```
This script also just sets some parameters and then calls ./python/analysis/habernal_comparison/compute_metrics.py.

3. Run the consensus test (Table 2) with crowdGPPL and compute performance metrics:

```
python ./python/analysis/habernal_comparison/personalised_tests.py 1
python ./python/analysis/habernal_comparison/personalised_metrics.py 1
```

4. Run the personalised test (Table 2) with single-user GPPL and compute performance metrics:

```
python ./python/analysis/habernal_comparison/personalised_tests.py 6
python ./python/analysis/habernal_comparison/personalised_metrics.py 6
```

5. Run the consensus test (Table 2) with single-user GPPL and compute performance metrics:

```
python ./python/analysis/habernal_comparison/personalised_tests.py 0
python ./python/analysis/habernal_comparison/personalised_metrics.py 0
```

6. Run the scalability tests in Figure 2(a):

```
python ./python/analysis/habernal_comparison/scalability_tests_personalised.py 0
python ./python/analysis/habernal_comparison/scalability_plots_personalised.py 0
```
The plots will be saved to './results/scalability/num_inducing_300_features.pdf'.

7. Run the scalability tests in Figure 2(b):

```
python ./python/analysis/habernal_comparison/scalability_tests_personalised.py 3
python ./python/analysis/habernal_comparison/scalability_plots_personalised.py 3
```
The plots will be saved to './results/scalability/num_arguments.pdf'.

8. Run the scalability tests in Figure 2(c):

```
python ./python/analysis/habernal_comparison/scalability_tests_personalised.py 2
python ./python/analysis/habernal_comparison/scalability_plots_personalised.py 2
```
The plots will be saved to './results/scalability/num_pairs.pdf'.

  
### Sushi preferences

These experiments make use of the data in './data/sushi3-2016/', which
were obtained from http://www.kamishima.net/sushi/. 

The experiments are divided into three parts, each on a different subset of the data. 
To run Sushi-A-small:
```
python ./python/analysis/sushi_10_tests.py 0
```
To run Sushi-A:
```
python ./python/analysis/sushi_10_tests.py 2
```
To run Sushi-B:
```
python ./python/analysis/sushi_10_tests.py 4
```

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
