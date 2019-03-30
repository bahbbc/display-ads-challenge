import dask.dataframe as dd
import pandas as pd

column_names = ['label','interger1','interger2','interger3','interger4','interger5','interger6',
                'interger7','interger8','interger9','interger10','interger11','interger12','interger13',
                'categorical1','categorical2','categorical3','categorical4','categorical5','categorical6',
                'categorical7','categorical8','categorical9','categorical10','categorical11','categorical12',
                'categorical13','categorical14','categorical15','categorical16','categorical17','categorical18',
                'categorical19','categorical20','categorical21','categorical22','categorical23','categorical24',
                'categorical25','categorical26']

df = dd.read_csv('/home/bahbbc/workspace/display-ads-challenge/dac/train.txt', sep='\t', names = column_names)

answer = dd.read_csv('/home/bahbbc/workspace/display-ads-challenge/dac/test.txt', sep='\t', names=column_names[1:])

df['interger1'] = df.interger1.mask(df.interger1.isnull(), df.interger1.mean())
df['interger2'] = df.interger2.mask(df.interger2.isnull(), df.interger2.mean())
df['interger3'] = df.interger3.mask(df.interger3.isnull(), df.interger3.mean())
df['interger4'] = df.interger4.mask(df.interger4.isnull(), df.interger4.mean())
df['interger5'] = df.interger5.mask(df.interger5.isnull(), df.interger5.mean())
df['interger6'] = df.interger6.mask(df.interger6.isnull(), df.interger6.mean())
df['interger7'] = df.interger7.mask(df.interger7.isnull(), df.interger7.mean())
df['interger8'] = df.interger8.mask(df.interger8.isnull(), df.interger8.mean())
df['interger9'] = df.interger9.mask(df.interger9.isnull(), df.interger9.mean())
df['interger10'] = df.interger10.mask(df.interger10.isnull(), df.interger10.mean())
df['interger11'] = df.interger11.mask(df.interger11.isnull(), df.interger11.mean())
df['interger12'] = df.interger12.mask(df.interger12.isnull(), df.interger12.mean())
df['interger13'] = df.interger13.mask(df.interger13.isnull(), df.interger13.mean())


integer_data = df[['interger1', 'interger2', 'interger3', 'interger4',
       'interger5', 'interger6', 'interger7', 'interger8', 'interger9',
       'interger10', 'interger11', 'interger12', 'interger13']]

print(integer_data.shape)

from dask_ml.wrappers import Incremental
from sklearn.linear_model import SGDClassifier


estimator = SGDClassifier(random_state=10, max_iter=30, loss='log')
clf = Incremental(estimator)
clf.fit(integer_data, df.label)

pred = clf.predict_proba(integer_data)
print(pred.mean().compute())
