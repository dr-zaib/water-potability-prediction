import numpy as np 
import pandas as pd

from sklearn.feature_extraction import DictVectorizer


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import accuracy_score, roc_auc_score


import pickle5
#import pickle




output_file = 'model_h2O_potability.bin'




# dataset preparation

dataset = pd.read_csv(r'/home/zaib/midterm-project/water_potability.csv')

dataset.columns = dataset.columns.str.lower().str.replace(' ', '_')

dataset = dataset.fillna(0)



# dataset spliting 

df_full_train, df_test = train_test_split(dataset, test_size=0.2, random_state=1)
y_full_train = df_full_train.potability.values
y_test = df_test.potability.values



# features

features = [
    'chloramines',
    'conductivity',
    'hardness',
    'organic_carbon',
    'ph',
    'solids',
    'sulfate',
    'trihalomethanes',
    'turbidity'
]




# train and predict functions


def train(df_train, y_train):
    dicts = df_train[features].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    n_estimators=110
    max_depth=18
    min_samples_leaf=1
    max_features=3
    class_weight='balanced'

    model = RandomForestClassifier(n_estimators=n_estimators,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf, 
                                max_features=max_features, 
                                bootstrap=True, 
                                class_weight=class_weight, 
                                random_state=1,
                                n_jobs=-1)
    model.fit(X_train, y_train)
    
    return dv, model



def predict(df, dv, model):
    dicts = df[features].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred




# metrics' functions


def outcomes (y_test, y_pred):
    
    actual_positive = (y_test == 1)
    actual_negative = (y_test == 0)
    
    predict_positive = (y_pred >= 0.5)
    predict_negative = (y_pred < 0.5)
    
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    
    return tp, tn, fp, fn


def precision_score (tp, fp):
    
    p = 0.0
    
    if (tp + fp) != 0:
        p = tp / (tp + fp)
    
    return p


def recall_score (tp, fn):
    
    r = 0.0
    
    if (tp+fn) != 0:
        r = tp / (tp + fn)
    
    return r


def f1_score (p, r):
    
    f1 = 0.0
    
    if (p + r) != 0:
        f1 = 2 * (p * r) / (p + r)
    
    return f1


# validation


print('cross-validation with n_estimators=110, max_depth=18, min_samples_leaf=1, max_features=3 and class_weight="balanced": ')

n_splits=10
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train_cross = df_full_train.iloc[train_idx]
    df_val_cross = df_full_train.iloc[val_idx]

    y_train_cross = df_train_cross.potability.values
    y_val_cross = df_val_cross.potability.values

    dv, model = train(df_train_cross, y_train_cross)
    y_pred = predict(df_val_cross, dv, model)

    auc = roc_auc_score(y_val_cross, y_pred)
    scores.append(auc)

print('mean auc and std.:')    
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))



# final training

dv, model = train(df_full_train, y_full_train)
y_pred = predict(df_test, dv, model)


# performances

print("the model's performances: ")

tp, tn, fp, fn = outcomes(y_test, y_pred)

print(f'tp={tp}, tn={tn}, fp={fp}, fn={fn}')

auc = roc_auc_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred >= 0.5)

precision = precision_score(tp, fp)
recall = recall_score(tp, fn)
f1score = f1_score(precision, recall)

print(f'auc={round(auc, 3)}\naccuracy={round(accuracy, 3)}\nprecision={round(precision, 3)}\nrecall={round(recall, 3)}\nf1 score={round(f1score, 3)}')
print()


# saving the model


with open(output_file, 'wb') as f_out:
    pickle5.dump((dv, model), f_out)

print(f'the model has been saved properly to {output_file}')