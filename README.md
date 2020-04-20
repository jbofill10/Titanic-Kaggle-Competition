# Titanic-Kaggle-Competition
A competition that I took part in that evaluates whether your model of choice was correct in picking the survival of 418 passengers.

I scored in the top 15% of the entire competition with an accuracy score of: 79.425%

Competition: https://www.kaggle.com/c/titanic

## Feature Selection
The most important features were:
- Age
- Sex
- Pclass
- Cabin
- Fare
- Title
- Embark Location

Cabin had a significant amount of missing values, so I filled those as Unknown instead of dropping.

The other missing values weren't missing nearly as many values as Cabin, so I filled them using the median based on a subset grouping of the passengers Pclass, Title, and Sex

I figured that Family Size would be important since if you have a larger family, you can help each other to escape the boat better than someone by himself. So I created a feature that added the siblings and parch (parent/children) aboard into one column called Family_Size  
  
I wanted to generalize some of the features I had because specific values would not mean much here.
- Age
- Title
- Fare

For Age and Fare, I split them up into 4 divisions using the quartiles of the data set.  
This resulted in Age having 4 ordinal variables: Child, Teen, Adult, Elder
Fare had: Low_Fare, Median_Fare, Average_Fare, and High_Fare

For Title, I took the royalty status titles and grouped them all under a Royalty feature. Since there were so few of these titles, I feared overfitting and generalized it.

There were also some titles like: Mlle, Ms, and Mme -- These 3 titles also map like this: Mlle -> Miss, Ms -> Miss, Mme -> Mrs. So I replaced the original 3 titles with their mappings

## Encoding
I only used one hot encoding since I only had binary or categorical values left.

## Model Selection
I used a GridSearch with Cross Validation to gauge which model would perform better. The only two model's I tested with hyperparameter tuning was Logistic Regression and Random Forest.

## Model Testing
The RF model had a CV score of 83% and Logistic Regression had 82%. So I decided to continue with RF

The params the GridSearch claimed did the best were:  
```json
{
  "max_depth": 70, 
  "max_features": "sqrt", 
  "min_samples_leaf": 1, 
  "min_samples_split": 5, 
  "n_estimators": 100, 
  "oob_score": "False"
}
```
I will do a train test split to see how well the model does against itself:
```
[[120  10]
 [ 29  64]]
Accuracy: 82.51121076233184
```

This was expected.  
  
I would like to test more models in the future, but this is all I have for now.
