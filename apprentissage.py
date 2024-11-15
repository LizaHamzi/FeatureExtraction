import os
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import RobustScaler, Normalizer, Binarizer, MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pandas import read_csv

file = 'combined_signatures.csv'
try:
    data = read_csv(file)
    print('Data loaded successfully')
except Exception as e:
    print(f'Failed to load data {e}')

data_copy = data.copy()
column = data_copy.iloc[:, -1]
encoder = LabelEncoder()
column_transformed = encoder.fit_transform(column)
data_copy.iloc[:, -1] = column_transformed
with open('Models/label_encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

# Remplacement des valeurs infinies par NaN et imputation des valeurs manquantes
data_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean')
data_copy.iloc[:, :] = imputer.fit_transform(data_copy)

# Traitement des valeurs aberrantes avec l'IQR
Q1 = data_copy.quantile(0.25)
Q3 = data_copy.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_copy = data_copy[~((data_copy < lower_bound) | (data_copy > upper_bound)).any(axis=1)]


X = data_copy.iloc[:, :-1].values
Y = data_copy.iloc[:, -1].values


scaler = Normalizer()
X_scaled = scaler.fit_transform(X)


pca = PCA(n_components=10)  
X_pca = pca.fit_transform(X_scaled)


test = 0.20
seed = 10
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=test, random_state=seed)


models = [
    ('GradientBoostingClassifier', GradientBoostingClassifier(n_estimators=100, learning_rate=0.05)),
    ('KNN', KNeighborsClassifier(n_neighbors=5, weights='uniform')),
    ('Naive Bayes', GaussianNB()),
    ('Decision Tree', DecisionTreeClassifier(max_depth=8)),
    ('SVM', SVC(C=10, kernel='rbf')),
    ('Random Forest', RandomForestClassifier(max_depth=12, n_estimators=150)),
    ('LogisticRegression', LogisticRegression(C=1, solver='lbfgs'))
]


metrics = {
    "Accuracy": accuracy_score,
    "Recall": recall_score,
    "Precision": precision_score,
    "F1-score": f1_score
}


save_dir = 'Models'

for metr_name, metric in metrics.items():
    print(f'Metric: {metr_name}\n---------------\n')
    for mod_name, model in models:
        classifier = model
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_test)
        if metr_name == 'Accuracy':
            result = metric(Y_pred, Y_test)
        else:
            result = metric(Y_pred, Y_test, average='micro')
        print(f'{mod_name}: {result*100:.2f}%')

        model_filename = os.path.join(save_dir, f'{mod_name}.pkl')
        with open(model_filename, 'wb') as f:
            pickle.dump(classifier, f)

with open(os.path.join(save_dir, 'pca.pkl'), 'wb') as f:
    pickle.dump(pca, f)

with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
