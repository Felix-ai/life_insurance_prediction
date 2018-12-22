## Life Insurance Prediction

In the data folder, you will find data on mortality in Kenya.
The obituaries dataset was garthered from the Daily Nation.
You will also find a Word Document that explains the variables.

The task at hand is to use this data to inform products for life insurance as there are
several analysis that can be conducted to enable make better decisions for creating and
marketing of life insurance products.

## Kaplan-mier Survival Curve
![alt text](images/Kaplan-meier_Survival_Curve.png "Kaplan-meier Survival Curve")


## Prediction of deaths that are likely to need fundraising
The task was carried out in Python.

After loading the data set, simple EDA was carried to familiarize ones self with  the data set.
Then identification of features and targets followed.
Categorical variables in the dataset were then encoded and all cases with missing values for
Fundraiser variable(the target variable) were dropped. These case can be used as test cases.

The objective was to determine whether an obituary will explicitly request for fund raising
(yes/no). This translates to a binary classification task.

Different classification algotithms could be employed. In this case, a Random Forest Classifier
was used to predict deaths that are likely to need fundraising. The main advantage been that it,
is an ensembe algorithmn.
