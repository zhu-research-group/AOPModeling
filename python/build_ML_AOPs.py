
import pandas as pd, numpy as np
import pymongo, toxcast

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, accuracy_score
from sklearn.neighbors import NearestNeighbors


from rdkit.Chem import Descriptors
from rdkit import Chem

import os

ML_DATA_DIR = os.path.join('..', 'data', 'text', 'ML')

def get_estimator_search(cv):
    PARAMETERS = {'lr__C': [0.001, 0.01, 0.1, 1, 10, 100]}

    lr = LogisticRegression(class_weight='balanced', max_iter=500)

    pipe = Pipeline([('lr', lr)])
    grid_search = GridSearchCV(pipe,
                               param_grid=PARAMETERS,
                               cv=cv,
                               refit='AUC')
    return grid_search


def get_data(ENDPOINT, LOG, BALANCE, kind='aop'):
    """ return X, y data for either hybrid or AOP data"""
    client = pymongo.MongoClient()

    pipe = [
        {"$match": {"_id.type": "human"}},
        {
            "$project": {
                "casn": "$_id.casn",
                "aop": "$_id.aop",
                "auc_score": "$auc_norm",
                "_id": 0
            }
        }
    ]

    human_model_scores = pd.DataFrame(client.LiverToxCast.ke_models.aggregate(pipe))

    # set to zero
    human_model_scores.loc[human_model_scores.auc_score <= 0, 'auc_score'] = 0

    modeling_set = pd.DataFrame(client.LiverToxCast.modeling_set.find({}, {"_id": 0, "CASRN": 0}))

    cols = [col for col in modeling_set.columns if "dl500" not in col]

    modeling_set = modeling_set[cols]


    # 922 chemicals calculated had CSS values.
    # find the intersection of this with the modeling
    # set
    css_estimates = pd.read_csv('../data/text/css_estimates_tox21.csv')
    liver = css_estimates.set_index('cas').join(modeling_set.set_index('casn'), how='inner')

    liver = liver[liver[ENDPOINT].notnull()]

    df = human_model_scores.pivot(index='casn', values='auc_score', columns='aop')
    df = df.loc[liver.index]

    if BALANCE:
        inactives = liver[ENDPOINT][liver[ENDPOINT] == 0].index
        actives = liver[ENDPOINT][liver[ENDPOINT] == 1].sample(len(inactives), random_state=10).index

        liver = liver.loc[actives.tolist() + inactives.tolist()]
        df = df.loc[actives.tolist() + inactives.tolist()]

        # inactives = liver[ENDPOINT][liver[ENDPOINT] == 0].index
        # actives = liver[ENDPOINT][liver[ENDPOINT] == 1].index
        # centroid = df.loc[actives].mean()
        # nbrs = NearestNeighbors(n_neighbors=len(inactives)).fit(df.loc[actives])
        # distances, indices = nbrs.kneighbors(centroid.values.reshape(1, -1))
        # df = df.loc[actives[indices[0]].tolist() + inactives.tolist()]

    if LOG:
        df = np.log(df + 0.001)


    AOPS = df.columns.tolist()

    # Taken from the above link
    fxs = [Chem.Lipinski.FractionCSP3,
           Chem.Lipinski.HeavyAtomCount,
           Chem.Lipinski.NHOHCount,
           Chem.Lipinski.NOCount,
           Chem.Lipinski.NumAliphaticCarbocycles,
           Chem.Lipinski.NumAliphaticHeterocycles,
           Chem.Lipinski.NumAliphaticRings,
           Chem.Lipinski.NumAromaticCarbocycles,
           Chem.Lipinski.NumAromaticHeterocycles,
           Chem.Lipinski.NumAromaticRings,
           Chem.Lipinski.NumHAcceptors,
           Chem.Lipinski.NumHDonors,
           Chem.Lipinski.NumHeteroatoms,
           Chem.Lipinski.NumRotatableBonds,
           Chem.Lipinski.NumSaturatedCarbocycles,
           Chem.Lipinski.NumSaturatedHeterocycles,
           Chem.Lipinski.NumSaturatedRings,
           Chem.Lipinski.RingCount
           ]
    names = ['FractionCSP3',
             'HeavyAtomCount',
             'NHOHCount',
             'NOCount',
             'NumAliphaticCarbocycles',
             'NumAliphaticHeterocycles',
             'NumAliphaticRings',
             'NumAromaticCarbocycles',
             'NumAromaticHeterocycles',
             'NumAromaticRings',
             'NumHAcceptors',
             'NumHDonors',
             'NumHeteroatom',
             'NumRotatableBonds',
             'NumSaturatedCarbocycles',
             'NumSaturatedHeterocycles',
             'NumSaturatedRings',
             'RingCount'
             ]

    liver = liver.loc[df.index]

    liver['mol'] = [Chem.MolFromInchi(inchi) for inchi in liver.rdkitInChIClean]
    liver['LogP'] = [Descriptors.MolLogP(mol) for mol in liver.mol]

    def calc_lipinksi(mol):
        data = []
        for fx in fxs:
            data.append(fx(mol))
        return data

    X = []
    for mol in liver.mol:
        X.append(calc_lipinksi(mol))

    lipinksi = pd.DataFrame(X, columns=names)
    lipinksi.index = liver.index

    df['CSS'] = liver['estimates']
    df['CLogP'] = liver['LogP']
    df['Hepatotoxicity'] = liver[ENDPOINT]

    if kind == 'hybrid':
        # df contains AOPs + CSS Clog P and Hepatotoxicity
        X = df.drop(columns=['Hepatotoxicity'])
        X[lipinksi.columns] = lipinksi

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        y = df['Hepatotoxicity']
        return X, y

    elif kind == 'aop':
        ### AOP

        # finally just AOPS
        X = df[AOPS]

        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

        y = df['Hepatotoxicity']
        return X, y
    else:
        return None, None



def do_ml(ENDPOINT, LOG, BALANCE, N_SPLITS = 'LOO'):
    """ train a logistic regression classifier on AOP scores and AOP scores + lipinski """

    session_string = f'{ENDPOINT}_LOG:{LOG}_BALANCE:{BALANCE}_SPLIT:{N_SPLITS}'
    session_folder = os.path.join(ML_DATA_DIR, session_string)

    if not os.path.exists(session_folder):
        os.mkdir(session_folder)



    # Hybrid
    X, y = get_data(ENDPOINT, LOG, BALANCE, kind='hybrid')
    X.to_csv(os.path.join(session_folder, 'features.csv'))
    # CV
    if N_SPLITS == 'LOO':
        cv = KFold(shuffle=True, n_splits=X.shape[0], random_state=2008)
    else:
        cv = StratifiedKFold(shuffle=True, n_splits=N_SPLITS, random_state=2008)

    grid_search = get_estimator_search(cv)

    grid_search.fit(X, y)
    best_estimator_hybrid = grid_search.best_estimator_

    cv_predictions = cross_val_predict(best_estimator_hybrid, X, y, cv=cv, method='predict_proba')[:, 1]
    cv_predictions_class = cross_val_predict(best_estimator_hybrid, X, y, cv=cv)

    hybrid_preds = pd.DataFrame([cv_predictions, cv_predictions_class, y],
                                index=['Probability', 'Prediction Class', 'Hepatotoxicity']).T
    hybrid_preds.index = X.index

    hybrid_recall = recall_score(y, cv_predictions_class)
    hybrid_precision = precision_score(y, cv_predictions_class)
    hybrid_accuracy = accuracy_score(y, cv_predictions_class)

    hybrid_stats = pd.Series([hybrid_recall, hybrid_precision, hybrid_accuracy], index=['Recall', 'Precision', 'Accuracy'])


    # aop
    X, y = get_data(ENDPOINT, LOG, BALANCE, kind='aop')

    grid_search = get_estimator_search(cv)

    grid_search.fit(X, y)
    best_estimator_aop = grid_search.best_estimator_

    cv_predictions = cross_val_predict(best_estimator_aop, X, y, cv=cv, method='predict_proba')[:, 1]
    cv_predictions_class = cross_val_predict(best_estimator_aop, X, y, cv=cv)

    aop_preds = pd.DataFrame([cv_predictions, cv_predictions_class, y],
                             index=['Probability', 'Prediction Class', 'Hepatotoxicity']).T
    aop_preds.index = X.index

    aop_recall = recall_score(y, cv_predictions_class)
    aop_precision = precision_score(y, cv_predictions_class)
    aop_accuracy = accuracy_score(y, cv_predictions_class)

    aop_stats = pd.Series([aop_recall, aop_precision, aop_accuracy], index=['Recall', 'Precision', 'Accuracy'])

    aop_preds.to_csv(os.path.join(session_folder, 'aop_preds.csv'))
    hybrid_preds.to_csv(os.path.join(session_folder, 'hybrid_preds.csv'))

    aop_stats.to_csv(os.path.join(session_folder, 'aop_stats.csv'))
    hybrid_stats.to_csv(os.path.join(session_folder, 'hybrid_stats.csv'))


if __name__ == '__main__':


    LOG = False
    BALANCE = True
    N_SPLITS = 'LOO'

    for ENDPOINT in ['H_HT_class']:
        do_ml(ENDPOINT, LOG, BALANCE, N_SPLITS)
