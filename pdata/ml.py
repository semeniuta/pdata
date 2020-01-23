import pandas as pd
from sklearn.model_selection import cross_val_score


def get_model_name(clf):
    return str(clf.__class__)[1:-1].split()[1][1:-1].split('.')[-1]


def cross_val_for_estimators(estimators, X, y, **cross_val_score_kwargs):

    cv_scores = []
    for est in estimators:
        current_scores = cross_val_score(est, X, y, **cross_val_score_kwargs)
        cv_scores.append(current_scores)

    n_folds = len(cv_scores[0])

    col_names = ['fold_{}'.format(i+1) for i in range(n_folds)]

    summary = pd.DataFrame(cv_scores, columns=col_names)  
    summary['cv_score_mean'] = summary.mean(axis=1)

    return summary
