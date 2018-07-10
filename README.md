# activity_recognition
Classify human actions using data from sensors on a waist-mounted smartphone. Data can be found on Kaggle as [UCI Machine Learning's Human Activity Recognition with Smartphones](https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones/home)

## nFoldCrossValidation.m
A simple first approach to this supervised learning problem. Solve for the optimal weights needed to identify the human activity by treating the dataset and its labels as a linear least squares problem.

The code uses n-fold cross validation in order to determine the "best" lambda to use (read: prevent overfitting with training data). The weights corresponding to the "best" lambda are the weights used for the test set.

Cross validation accuracy is ~98%, while test accuracy is ~96%. This discrepancy is likely due to the data set NOT being IID. The cross validation's training and validation sets are all data generated from the same 21 volunteers, while the test set is data generated from a different group of 9 volunteers.

Plots of the results to come!
