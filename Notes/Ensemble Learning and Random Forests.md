# Ensemble Learning and Random Forests

Owner: Piyush Mittal

Sometimes when the crowd is large enough and we combine their answers, we end up with an answer better than an experts answer. (***Wisdom of the Crowd***)

Similarly in Machine Learning, if we combine multiple classifiers or regressors and get a cumulative answer, we will get a better answer than the best individual classifier or regressor. When the predictors are combined, we call it an ensemble, thus the world ensemble learning.

## *Voting Classifiers*

In voting classifiers, we build an ensemble of multiple predictors, aggregate their predictions and predict the class with the most votes. This is called a ***hard voting classifier***.

Voting classifiers achieve higher accuracy than the best classifier in the ensemble. Even if all the classifiers in the ensemble are weak learners (slightly better than random guessing), then also we can obtain an ensemble that is a strong learner (high accuracy) provided we have a good amount of weak learners that are diverse (i.e. they are independent and make uncorrelated errors, which is not the case every time as they are trained on the same data) .

***Soft Voting Classifiers*** make use of Probability. If the underlying predictors in the ensemble can predict probability that an instance belongs to a particular class, then we can predict the class with the highest probability, averaged over all the predictors.

- **It works better than hard voting in most cases as it gives more weight to highly confident votes.**
- **To use soft voting, change voting = “hard” to “soft” and ensure the underlying classifiers can predict probabilities(predict_proba()).**
- **For *SVC we need to set probability hyper parameter to True*, but it will slow down training.**

## Bagging and Pasting

These are two methods used in Ensemble training to introduce some diversity in the Models. This is done to expect less correlated models and hopefully reduced bias and variance.

In bagging we train each model in the ensemble on different subsets that are randomly picked/generated with replacement. Since there is replacement we say its bagging. When sampling is done without replacement, we called it pasting.

- Bagging and pasting both allow training instances to be sampled multiple times across multiple models but only bagging allows the instances to be sampled several times for different predictors.
- Because in pasting once an instance in picked in one subset, it cannot be picked for the other subset which is not the case for bagging.

Generally the resulting ensemble has similar bias but a noticeable lower variance than a single predictor trained on the original training set.

### Bagging and Pasting in Scikit-Learn

To use bagging and pasting in Scikit-Learn use the BaggingClassifier or BaggingRegressor API.

- To use bagging set bootstrap = True , else False for pasting.
- n_jobs hyper parameter is used to tell Scikit-Learn how many cores of our processor to use for training and predictions.
- BaggingClassifier automatically uses soft voting instead of hard , if the all the models in the ensemble allow predict_proba().

### *Out of Bag Evaluation*

OOB Evaluation is a good practice of evaluating your models’ accuracy without needing a separate evaluation/test set. When we bootstrap an estimator, there are few instances that are left out, meaning, they are not used for training that particular tree. 

This means that, if we identify the trees that weren’t trained on that instance, we can use that instance on those trees, aggregate their results and make a prediction.

Usually the estimate is quite accurate and is unbiased, because the predictions are only made using the trees which weren’t trained on that sample.

OOB evaluation is particularly useful when there is data scarcity, meaning splitting the data might be expensive and lead to model overfitting and high variance or even high bias. 

We can only OOB evaluation on ensemble techniques like bagging, because we use bootstrapping. If we aren’t bootstrapping OOB evaluation wouldn’t be applicable. 

### *Random Patches and Random Subspaces*

To achieve even more diversity in the training samples, ***BaggingClassifier*** allows us to samples features as well. 
It is controlled by two hyper-parameters, max_features and bootstrap_features, and they work the same way as max_samples and bootstrap. When we sample both features and instances, it is called *Random Patches*, but when we only sample features, it is called *Random Subspaces*.

## *Random Forests*

Instead of using BaggingClassifier and passing DecisionTreeClassifier we can use Random Forest Classifier(more convenient and optimised for Decision Trees), similarly for regressor.

RandomForestClassifier has all the hyper parameters of a decision tree classifier, plus all the hyper parameters of a Bagging Classifier to control the ensemble itself.

### Extra Trees

- When growing a tree in a Random Forest, we consider a random set of features to introduce more randomness, and find the best feature within that set.
- To make the trees even more random, instead of finding the best thresholds, we can use random thresholds for each feature.

A forest of such extreme randomness is called an ***Extremely Randomised Tree Ensemble*** or ***Extra Trees*** for short.

This makes training the tree much faster, because finding the best threshold value for split is one of the most time consuming tasks of training a tree.

### Feature Importance

Random Forests allow us to measure the relative importance of the features. Scikit-Learn calculates this by measuring the amount of impurity is reduced in a tree nodes that use that feature. 

Thus, this can help us eliminate features that do no have significant or meaningful contributions in the predictions.

We can print the values for all the features by using *feature_importances_* variable.

### Boosting

It is a method used in ensemble training where we combine multiple weak learners into a strong learner. We train the models sequentially on the samples that were misclassified on the preceding model.

### Ada Boost

- In Ada Boost short for adaptive boost, we assign higher weights to the samples that were misclassified, so that in the next iteration more focus is on the misclassified sample(s).
- In ada boost the trees are usually stumps( one node and two children).
- The weight of the misclassified example is increased, and in the subsequent iteration the subsequent model is trained with updated weights.
- Ada Boost, instead of tweaking a single predictors’ parameters, it adds predictors to the ensemble.
- We use the weighted average of the predictions. The weight to each predictor is given based on the amount of error, less weight if high error and vice versa, meaning less weight is given to weaker models.
- Since the learning is sequential, it can’t be parallelised, meaning it doesn’t scale well for large data. If the predictor is mostly wrong, the weight given to is negative.
- Scikit-Learn uses a multi class version of ada boost called SAMME which stands for Stagewise additive Modelling using a Multi class Exponential loss function.
    - When there are only two classes SAMME acts like normal ada boost.
    - If our predictor is able to predict probabilities, using predict_proba(), then we can use SAMME.R which relies on class probabilities and generally performs better.
- Its working:
    - First sample weights are calculated for each sample, which initially is equal 1/n, n being number of samples.
    - Total error is calculated for the stump for each misclassified sample by adding their sample weights(only for misclassified samples).
    - The amount of say is then calculated for the predictor using a formula
        
        alpha_t = eta * ln((1 - e_t) / e_t), where eta is the learning rate, and e_t is the total error for the node.
        
    - We then increase the weight for the misclassified sample and decrease the weight for the correctly classified samples using the formula : w(i) * e^amt_of_say for misclassified and w(i) * e^-(amt_of_say) for correctly classified.
    - Then normalise the weights.

### Gradient Boosting

- Gradient Boosting, like Ada Boost, builds predictors sequentially, but adjusts to the residual errors of the previous predictor rather than instance weights.
- A model is trained, predicts for each instance, and passes the residuals to the next model. This process is repeated with each model learning from the previous residuals.
- The models are combined via simple addition, with each model correcting its predecessor's mistakes. The addition rate can be adjusted with the learning rate or "shrinkage", which affects performance and computation cost.
- Gradient Boosting is versatile with different loss functions but is sensitive to noisy data and outliers and requires parameter tuning.
- Its’ working:
    - Just like in AdaBoost, initially all the samples are given equal weight, which is 1/n, where n is the number of samples.
    - Next, a model is fit on the data and predictions are made. The residuals (or differences between actual and predicted values) are calculated as r = y - y_hat, where y is the actual value and y_hat is the predicted value.
    - A new model is then fit on the residuals, not on the original data. This new model is trying to correct the errors made by the previous model.
    - The predictions from this new model are calculated and added to the previous predictions to get updated predictions. This is done using the formula y_hat_new = y_hat_old + learning_rate * prediction_residuals, where learning_rate is a parameter that decides how much of the residuals from the current model will be used to update the predictions.
    - Steps 3 and 4 are repeated until a stopping condition is met, such as if the number of models reaches a set number, or if the residuals are no longer decreasing.
    - The final prediction is then the sum of the predictions from all the models.

## *XGBoost*

XGBoost or Extreme Gradient Boosting is a highly efficient implementation of gradient boosting, designed for speed and performance. It stands out for its capacity to perform the parallel computations. XGBoost also includes a variety of regularisation which reduces overfitting and improves overall performance. You can use XGBoost for both regression and classification tasks.

# Sayonara…