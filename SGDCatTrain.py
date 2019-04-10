from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import dump, load


column_names = ['label','interger1','interger2','interger3','interger4','interger5','interger6',
                'interger7','interger8','interger9','interger10','interger11','interger12','interger13',
                'categorical1','categorical2','categorical3','categorical4','categorical5','categorical6',
                'categorical7','categorical8','categorical9','categorical10','categorical11','categorical12',
                'categorical13','categorical14','categorical15','categorical16','categorical17','categorical18',
                'categorical19','categorical20','categorical21','categorical22','categorical23','categorical24',
                'categorical25','categorical26']


df = pd.read_csv('/home/bahbbc/workspace/display-ads-challenge/dac/train.txt', sep='\t',
            names=column_names, chunksize=327433, engine='python')


logistic = SGDClassifier(loss='log',
                         penalty='l2',
                         alpha=0.1,
                         fit_intercept=False,
                         max_iter=1000,
                         shuffle=False,
                         verbose=0,
                         n_jobs=-1,
                         random_state=42,
                         learning_rate='optimal',
                         tol=0.01)

scaler = StandardScaler()
feature_hasher = FeatureHasher(n_features=1000)


i = 0
weights = []
for chunk in tqdm(df):
    integer_cols = ['interger2', 'interger8', 'interger5','interger11', 'interger9', 'interger7']
    cat_vars = ['categorical8', 'categorical14', 'categorical5', 'categorical12', 'categorical11', 'categorical10', 'categorical7', 'categorical1', 'categorical15', 'categorical18', 'categorical13', 'categorical16']

    #merge cols
    X = chunk[integer_cols + cat_vars]
    y = chunk['label']

    # replace nulls
    X.update(X[integer_cols].fillna(0))
    X.update(X[cat_vars].fillna('NULL'))

    X_cat = feature_hasher.transform(X[cat_vars].to_dict('records')).toarray()

    if i == 0:
        scaler.fit(X[integer_cols])
        X_integer = scaler.transform(X[integer_cols])
        X = np.hstack((X_cat, X_integer))

        logistic.fit(X, y)
        dump(logistic, 'first_logit.pkl')
    else:
        X_integer = scaler.transform(X[integer_cols])
        X = np.hstack((X_cat, X_integer))

        logistic.partial_fit(X, y, classes=[0,1])
    i += 1

print(logistic)
print(logistic.predict_proba(X).mean())
dump(logistic, 'regularized-cat-logistic.pkl')
dump(scaler, 'integer-scaler.pkl')
