from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
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
            names=column_names, chunksize=1309732, engine='python')


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
fh = FeatureHasher(n_features=6, input_type='string')


i = 0
weights = []
for chunk in tqdm(df):
    integer_cols = ['interger1', 'interger2', 'interger3', 'interger4',
    'interger5', 'interger6', 'interger7', 'interger8', 'interger9',
    'interger10', 'interger11', 'interger12', 'interger13']
    X = chunk[integer_cols]
    y = chunk['label']

    X = X.fillna(0)

    if i == 0:
        scaler.fit(X)
        X = scaler.transform(X)
        logistic.fit(X, y)
        dump(logistic, 'first_logit.pkl')
    else:
        X = scaler.transform(X)
        logistic.partial_fit(X, y, classes=[0,1])
    i += 1

print(logistic)
print(logistic.predict_proba(X).mean())
dump(logistic, 'regularized-logistic.pkl')
