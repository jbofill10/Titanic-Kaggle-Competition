import numpy as np

def clean(train_df, test_df):

    dfs = [train_df, test_df]

    train_titles = list()
    test_titles = list()
    for name in train_df['Name']:

        title = name[name.index(',')+1:name.index('.')]

        train_titles.append(title.strip())

    for name in test_df['Name']:
        title = name[name.index(',')+1:name.index('.')]

        test_titles.append(title.strip())

    train_df['Title'] = train_titles
    test_df['Title'] = test_titles

    train_age_group = train_df.groupby(['Sex', 'Pclass', 'Title'])
    test_age_group = test_df.groupby(['Sex', 'Pclass', 'Title'])
    
    train_df['Age'] = train_age_group['Age'].apply(
        lambda x: x.fillna(x.median())
    )
    test_df['Age'] = test_age_group['Age'].apply(
        lambda x: x.fillna(x.median())
    )

    # Mrs median was NaN, so I used Train df grouping to fill in that value

    test_df['Age'] = train_age_group['Age'].apply(
        lambda x: x.fillna(x.median())
    )

    # So many NaNs in Cabin and I can't drop because I'll drop more than 75% of the DF. I will just leave it as unknown
    # Embarked has >5 missing values. Therefore I can just add the most popular one (safe to assume)

    for df in dfs:

        df['Cabin'].fillna('Unknown', inplace=True)
        temp_embark_grouping = df['Embarked'].value_counts().sort_values(ascending=False)
        df['Embarked'].fillna(temp_embark_grouping.index[0], inplace=True)
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    return train_df, test_df