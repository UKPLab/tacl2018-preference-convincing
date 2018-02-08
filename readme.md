---+ Dependencies

Dependencies for running gp_pref_learning model:

   * scikit-learn==0.18.1
   * scipy==0.19.0
   * numpy==1.12.1

For running the experiments, please see the requirements.txt for further dependencies. 

---+ How to run

To run preference learning, see gp_pref_learning class in 
python/models/gp_pref_learning.py. You can run a simple example that generates
synthetic data by running python/test/gp_pref_learning_test.py.

The preference learning model and algorithm are implemented by the class 
GPPrefLearning inside python/test/gp_pref_learning_test.py. 
The important methods in this class are listed below; please look at the 
docstrings for these methods for more details:
   * The constructor: set the model hyperparameters
   * fit(): train the model
   * predict(): predict pairwise labels for new pairs 
   * predict_f(): predict rankings for a set of items given their features.

---+ Setting priors

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