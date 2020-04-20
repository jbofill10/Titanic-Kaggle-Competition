from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, accuracy_score


import pickle
import pandas as pd
import numpy as np
import os


def begin(train_df, test_df):

    dfs = [train_df, test_df]

    target = train_df['Survived']
    train_df.drop('Survived', axis=1, inplace=True)
    test_predictions = test_df[['PassengerId']].copy()

    # Feature Engineering

    for df in dfs:
        df.drop(['Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

        df['Fare'].replace(0, np.median(df['Fare']), inplace=True)

        df['Cabin'] = df['Cabin'].apply(lambda x: x[:1])

    for df in dfs:
        df['Family_Size'] = df['SibSp'] + df['Parch'] + 1

        df['Age_Category'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 200], labels=['Child', 'Teen', 'Adult', 'Elder'])

        df['Fare_Category'] = pd.cut(df['Fare'], bins=[0, 7.91, 14.45, 31.0, 600],
                                     labels=['Low_Fare', 'Median_Fare', 'Average_Fare', 'High_Fare'])

        df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                           'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'the Countess'], 'High_Rank')

        df['Title'] = df['Title'].replace('Mlle', 'Miss')
        df['Title'] = df['Title'].replace('Ms', 'Miss')
        df['Title'] = df['Title'].replace('Mme', 'Mrs')

    for df in dfs:
        df.drop(['Age', 'Fare'], axis=1, inplace=True)

    test_df['T'] = 0


    one_hot = make_column_transformer(
        (OneHotEncoder(), ['Pclass', 'Sex', 'Embarked', 'Age_Category', 'Fare_Category', 'Title', 'Cabin']),
        remainder='passthrough')

    params = {
        'log_params': {
            'penalty': ['l2'],
            'C': range(1, 11),
        },
        'random_params': {
            'n_estimators': [100, 200, 300, 400, 500, 800, 1000, 1500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, None],
            'min_samples_split': [2,5,10],
            'min_samples_leaf':[1, 2, 4],
            'oob_score': [False,True]
        }
    }

    logistic = LogisticRegression(max_iter=1000)
    log_grid = GridSearchCV(logistic, param_grid=params['log_params'], cv=5, refit=True, verbose=100)

    random_forests = RandomForestClassifier(n_jobs=-1)
    rf_grid = GridSearchCV(random_forests, param_grid=params['random_params'], cv=5, refit=True, verbose=100)

    preprocess = Pipeline(steps=[
        ('one_hot', one_hot)
    ])

    grids = {'log': log_grid, 'rf': rf_grid}

    print(train_df.shape)
    print(test_df.shape)

    train_df = preprocess.fit_transform(train_df)
    test_df = preprocess.fit_transform(test_df)

    if not os.path.isfile('pickles/models'):

        print(test_df)
        results = run_grids(grids, train_df, target)

        for result in results:
            print(result)

        with open('pickles/models', 'wb') as file:
            pickle.dump(results, file)

    else:
        with open('pickles/models', 'rb') as file:
            results = pickle.load(file)

    for i in results:
        print(results[i]['results'].best_score_)
        print(results[i]['results'].best_params_)

    best_params = results[i]['results'].best_params_

    testing_model = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
                                           max_features=best_params['max_features'], min_samples_leaf=best_params['min_samples_leaf'],
                                           min_samples_split=best_params['min_samples_split'], oob_score=best_params['oob_score'])

    x_train, x_test, y_train, y_test = train_test_split(train_df, target)

    testing_model.fit(x_train, y_train)

    y_pred_test = testing_model.predict(x_test)

    print(confusion_matrix(y_test, y_pred_test))
    print(accuracy_score(y_test, y_pred_test))

    y_pred = results['rf']['model'].predict(test_df)

    test_predictions['Survived'] = y_pred

    test_predictions.to_csv('submission/submission.csv', index=False)


def run_grids(grids, x, y):
    estimators = {}
    for name, grid in grids.items():
        search = grid.fit(x, y)
        estimators[name] = {
            'model': grid,
            'results': search
        }

    return estimators
