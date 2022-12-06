import pandas as pd
import numpy as np
import pickle
df=pd.read_csv(r"C:diab.csv")

y = df['Outcome'] # genre variable.
X = df.loc[:, df.columns != 'Outcome'] #select all columns but not the outcome

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=0)
rf.fit(X, y)


pickle.dump(rf,open('model.pkl','wb'))
model= pickle.load(open('model.pkl','rb'))









