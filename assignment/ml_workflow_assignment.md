# Task - 1

# Label
repeat_purchase_flag: This is the target variable (label) we are trying to predict. It represents a binary outcome: 1 if the customer made a repeat purchase within 30 days, 0 otherwise.

# Features
order_count_last_90d: we want this is as our features for if this particular days this customer has order or not.

days_since_last_order: we need this also its based on we predict if the customer last order is within 30 days or not.

avg_order_value: this is not impacting much in model prediction if we use the preditction not create big impact.

# Data Leakage
discount_used_on_repeat_order: This column introduces data leakage because it contains information about the discount applied on the repeat purchase itself. You cannot know this value until after the customer has already made the repeat purchase, making it impossible to use for prediction—it is using knowledge of the outcome to predict the outcome.


# Task 2

1. If this ML useful: first you have to make decision for this case ML model needed or not for our case it's needed because based on customer purchase we have plan futher improvements.

2. Audit: Collect the data and handle the missing values, outliers and duplication and others.

3. features selection(features engineering): we have to choose model prediction needed features and lable if we choose more features it's create imbalance model because of too many features or uncessary features create data leakage.

4. traing, testing, validation: Have to split data set to traing, testing and validation data for model traing and evaluvating. Traing set used to train a model and testing set used to evaluvate the model and check the accurecy using and validation set used to tune a model and catch a overffiting. 

5. Baseline: this is very important one first we have to understand the features and lable to predict base line of our model if we do not know this then we can't confirm our model is working good or not. EDA to understand distributions, correlations, and potential issues before modeling

6. Feature Engineering: This is the important concept here we are tune the our features based on column values if the column has categorical values we can used encodeing technique to tune the features to chage ML understanding level and same as follow scalling an nomalization for the features tuning if the data has outlier then used IQR and robust scalling and no outliers used standard scalling and we want values 0 to 1 using MinMaxScaling likewise using technique to tune the features.

7. Evaluvating Model: using testing data to test the model to find the prediction accuracy(mean squre error, r2 sequered values) to know the model performance. 