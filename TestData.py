from sklearn.linear_model import SGDClassifier
import pandas as pd
from tqdm import tqdm
from joblib import dump, load

column_names = ['interger1','interger2','interger3','interger4','interger5','interger6',
                'interger7','interger8','interger9','interger10','interger11','interger12','interger13',
                'categorical1','categorical2','categorical3','categorical4','categorical5','categorical6',
                'categorical7','categorical8','categorical9','categorical10','categorical11','categorical12',
                'categorical13','categorical14','categorical15','categorical16','categorical17','categorical18',
                'categorical19','categorical20','categorical21','categorical22','categorical23','categorical24',
                'categorical25','categorical26']

answers = pd.read_csv('/home/bahbbc/workspace/display-ads-challenge/dac/test.txt', sep='\t',
            names=column_names, engine='python')

logistic = load('logistic.pkl')

integer_cols = ['interger1', 'interger2', 'interger3', 'interger4',
'interger5', 'interger6', 'interger7', 'interger8', 'interger9',
'interger10', 'interger11', 'interger12', 'interger13']

X = answers[integer_cols]

print(logistic.predict_proba(X).mean())
