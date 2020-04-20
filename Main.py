import pandas as pd
import os
import DataCleaning
import Modeling

def main():
    dirs = []
    for dirname, _, filenames in os.walk('Data'):
            dirs = [dirname + "/" + i for i in filenames]

    test_df = pd.read_csv(dirs[0])
    train_df = pd.read_csv(dirs[1])

    train_df, test_df = DataCleaning.clean(train_df, test_df)

    Modeling.begin(train_df,test_df)


if __name__ == '__main__':
    main()