
import numpy as np 
import pandas as pd 
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import  OneHotEncoder, StandardScaler

def load_benchmark(name ='Churn', for_model = 'deep', data_path = 'Datasets/', k= 5, for_classif=True):
    if name=='Churn':
        data = pd.read_csv(data_path+'Churn_Modelling.csv')
        data.set_index('RowNumber', inplace = True)
        data.drop(['CustomerId','Surname'], axis=1, inplace=True)
        cat_cols = ['Geography','Gender']
        num_cols = ['CreditScore', 'Age', 'Tenure','Balance', 'NumOfProducts', 'EstimatedSalary']
        feature_names = list(data.columns)
        _ = feature_names.pop()
        
        y = data['Exited'].values
        
        data = data[feature_names]
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        if for_model =='deep':
            scaler = StandardScaler( )
            data[num_cols] = scaler.fit_transform(data[num_cols])
        
        data = pd.get_dummies(data, columns=cat_cols)
        feature_names = list(data.columns)
        X = data.values
        n = len(data)
    elif name=='CreditDefault':
        data = pd.read_csv(data_path+'defaultofcreditcardclients.csv')
        data.columns = data.iloc[0,:].values
        data.drop([0], inplace = True)
        data.set_index('ID', inplace = True)
        cat_cols = ['SEX','EDUCATION','MARRIAGE']
        num_cols = [c for c in data.columns if c not in cat_cols]
        _ = num_cols.pop()
        feature_names = list(data.columns)
        _ = feature_names.pop()
    
        y = data['default payment next month'].values
        data = data[feature_names]
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        
        if for_model =='deep':
            scaler = StandardScaler( )
            data[num_cols] = scaler.fit_transform(data[num_cols])
        
    
        data = pd.get_dummies(data, columns=cat_cols)
        feature_names = list(data.columns)
        X = data.values 
        n = len(data)
    elif name=='BankMarketing':
        data = pd.read_csv(data_path+'bank-full.csv',sep=';')
 
        
        map_month = {'jan':1,'feb':2,'mar' :3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12 }
        data['month'] = data['month'].apply(lambda x: map_month.get(x))
        cat_cols = ['job','marital','education','default','housing','loan','contact','poutcome', 'month']
        num_cols = [c for c in data.columns if c not in cat_cols]

        _ = num_cols.pop()

        feature_names = list(data.columns)
        feature_names.pop()

        y = data['y'].values

        data = data[feature_names]
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        if for_model =='deep':
            scaler = StandardScaler( )
            data[num_cols] = scaler.fit_transform(data[num_cols])
    
        data = pd.get_dummies(data, columns=cat_cols)
        feature_names = list(data.columns)
        X = data.values
        n = len(data)
    elif name=='AdultIncome':
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num','marital-status',
           'occupation','relationship', 'race', 'sex','capital-gain','capital-loss',
           'hours-per-week','native-country','income'
           
          ]
        data = pd.read_csv(data_path+'adult.data', header=None,na_values='?' , skipinitialspace=True).dropna()
        data.columns = columns
        cat_cols = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
        num_cols = [c for c in data.columns if c not in cat_cols]
        _ = num_cols.pop()
        
        feature_names = list(data.columns)
        _ = feature_names.pop()
        y = np.where(data['income']=='<=50K',0,1)
        data = data[feature_names]
    
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        
        if for_model =='deep':
            scaler = StandardScaler( )
            data[num_cols] = scaler.fit_transform(data[num_cols])
        
        data = pd.get_dummies(data, columns=cat_cols)
        
        feature_names = list(data.columns)
        X = data.values
        n = len(data)
    elif name=='TelChurnBig':
        data = pd.read_excel(data_path+'mobile-churn-data.xlsx')
        data = data.drop(["year", "user_account_id"],axis=1)
        feature_names = list(data.columns)
        cat_cols = []

        _= feature_names.pop()
        
        y = data['churn'].values
        
        data = data[feature_names]
        
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        
        if for_model =='deep':
            scaler = StandardScaler( )
            data = scaler.fit_transform(data)
            X = data
        else:
            X = data.values
        n = len(data)
    elif name=="HelocFico":
        data = pd.read_csv(data_path+'heloc_dataset_v1.csv')
        feature_names = list(data.columns)
        feature_names.remove('RiskPerformance')
        cat_cols = []

        y = data['RiskPerformance'].values
        
        data = data[feature_names]
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        
        if for_model =='deep':
            scaler = StandardScaler( )
            data = scaler.fit_transform(data)
            X = data
        else:
            X = data.values
        n = len(data)
    elif name=='Blastchar':
        data = pd.read_csv(data_path+ 'Telco-Customer-Churn.csv')
        data.drop(['customerID'], axis=1, inplace=True)
        data.Churn = np.where(data.Churn=='No',0,1)
        cat_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'Contract', 'PaperlessBilling','PaymentMethod' ,'StreamingMovies','OnlineSecurity'
           ]
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        ## fill NAs with the mean
        data['TotalCharges'] = np.where(data['TotalCharges']==' ',np.nan, data['TotalCharges'])  
        data[num_cols] = data[num_cols].astype(float) 
        data['TotalCharges'] = data['TotalCharges'].fillna(data['TotalCharges'].mean())
        feature_names = list(data.columns)
        _ = feature_names.pop()
        y = data['Churn'].values
        
        data = data[feature_names]
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        if for_model =='deep':
            scaler = StandardScaler()
            data[num_cols] = scaler.fit_transform(data[num_cols])
        
        data = pd.get_dummies(data, columns=cat_cols)
        feature_names = list(data.columns)
        X = data.values
        n = len(data)
    elif  name=='CreditCardFraud':
        print('Please first make sure that you have download the dataset if not dowload it at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download')
        data = pd.read_csv(data_path+'creditcard.csv')
        feature_names = list(data.columns)
        cat_cols = []

        _ = feature_names.pop()
        _ = feature_names.remove('Time')

        y = data['Class'].values
        enc = OneHotEncoder()
        Y = enc.fit_transform(y[:, np.newaxis]).toarray()
        data = data[feature_names]
        X = data.values
        n = len(data)

    else:
        print('Unknown dataset')
        return None
    
    n_features =  X.shape[1]
    
    
    ## cross validation split 
    if for_classif:
        Sf= StratifiedKFold(n_splits=k, shuffle=True, random_state= 42)
        datasets = [(X[train_index], X[test_index],Y[train_index], Y[test_index]) for train_index,test_index in Sf.split(X, Y[:,1]) ]
        n_classes = 2
    else:
        Sf= KFold(n_splits=k, shuffle=True, random_state= 42)
        datasets = [(X[train_index], X[test_index],Y[train_index], Y[test_index]) for train_index,test_index in Sf.split(X, Y) ]
        n_classes = 1
    
    return len(cat_cols),n, feature_names, n_features,n_classes, datasets
