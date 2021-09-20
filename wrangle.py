import pandas as pd
import numpy as np
from acquire import get_zillow_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PATH = "/content/drive/MyDrive/Colab Notebooks/clustering/data/"


def wrangle_zillow(test_size=.12, k=2, thresh=.4, random_state=0, path=PATH):
    
    #df = get_zillow_data()
    #df.to_csv(path+"sup_2017.csv")
    df = pd.read_csv(path+"sup_2017.csv", index_col='parcelid')
    df = df.rename(columns={'calculatedfinishedsquarefeet':'finishedsqft',
                            'lotsizesquarefeet':'lotsqft',
                            'structuretaxvaluedollarcnt':'structuretaxvalue',
                            'yearbuilt':'year', 'taxvaluedollarcnt':'taxvalue',
                            'landtaxvaluedollarcnt':'landtaxvalue'})
    
    #zips = pd.get_dummies(df.regionidzip, drop_first=True)
    #df = pd.concat([df, zips], axis=1)
    
    df.latitude, df.longitude = df.latitude/1e6, df.longitude/1e6
    df['livingarearatio'] = df.finishedsqft/df.lotsqft
    df['buildinglandvalueratio'] = df.structuretaxvalue/df.landtaxvalue 
    df['taxrate'] = df.taxamount/df.taxvalue
    df['age'] = 2017-df.year
    df['acres'] = df.lotsqft/43560
    df['sqftvalue'] = df.taxvalue/df.finishedsqft
    df['county'] = df.fips.map({6111:"Ventura", 6037:"Los Angeles", 6059:"Orange"})
    
    df = df.drop(columns=['county', 'id', 'regionidcounty', 'Unnamed: 0', 'roomcnt',
                            'maxdate', 'id.1', 'unitcnt', 'parcelid.1', 'fullbathcnt',
                            'finishedsquarefeet12', 'propertycountylandusecode',
                            'propertyzoningdesc', 'year', 'calculatedbathnbr',
                            'assessmentyear', 'lotsqft', 'rawcensustractandblock',
                            'transactiondate'])
                            
    df = drop_columns(df, thresh)
    df = iqr_method(df, k, ['taxvalue', 'taxamount', 'taxrate'])
    df = backfill(df)
    
    y = df.pop('logerror')
    
    return split_data(df, y, test_size, random_state)


def iqr_method(df, k, cols):
    
    was = len(df)
    #drop row if column outside fences, k is usu. between 1.5 and 3
    for col in cols:
        
        q1, q3 = df[col].quantile([.25,.75])
        iqr = q3-q1
        upperbound, lowerbound = q3 + k*iqr, q1 - k*iqr
        df = df[(df[col] > lowerbound) & (df[col] < upperbound)]
    
    print(f"{was-len(df)} outliers dropped.")
    return df


def split_data(X, y, test_size, random_state):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    test_size2 = test_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size2, random_state=random_state)
    print(f"X_train {X_train.shape}, X_test {X_test.shape}, X_val {X_val.shape}")
    print(f"y_train {y_train.shape}, y_test {y_test.shape}, y_val {y_val.shape}")
    
    return X_train, X_test, X_val, y_train, y_test, y_val


def drop_columns(df, thresh):
    #drop columns missing more than threshold percentage
    missing = df.isna().sum()
    out = pd.DataFrame(pd.concat([missing, missing/len(df)], axis=1))
    out = out.rename(columns={0:'missing count', 1:'missing %'})
    out = out[out['missing %'] > thresh]
    print("These columns were dropped.")
    print(out)
    return df.drop(columns=list(out.index))


def backfill(df):
    
    print("\nThese are the counts of missing data. \nMissing data is replaced with the median.")
    df = df.fillna(0)
    print(np.sum(df==0))
    
    for col in list(df.columns):
        df[col].replace(0, np.nanmedian(df[col]), inplace=True)
        
    return df