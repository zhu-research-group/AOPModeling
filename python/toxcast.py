TOXCAST_DATA_CACHE = r'/Users/danielrusso/data/invitro_modeling/cache'
MONGO_HOST = 'localhost'


TARGET_ENDPOINTS = ['human_hb', 'animal_hb', 'animal_dl500_hb', 'human_hc', 'animal_hc', 'animal_dl500_hc']


ANIMAL_ENDPOINTS = ['animal_hb', 'animal_dl500_hb', 'animal_hc', 'animal_dl500_hc']
HUMAN_ENDPOINTS = ['human_hb', 'human_hc', 'linlin_call', 'DILIRank_call']


def auc_score(cr):
    """ give a dataframe with receptors as index and concentraions as columns
    will calculated the auc score for each given receptor """
    signs = cr.diff(axis=1).fillna(1)
    signs[signs < -0.01] = -1
    signs[signs >= -0.01] = 1

    # calc AUC as score = score + sign*x[i]
    # then divide by number of concentrations
    scores = (signs*cr).sum(1) / signs.shape[1]
    return scores.round(3)

def make_roc(ax, fpr, tpr, name, auc):
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
#     ax.set_xticklabels(['{:.2f}'.format(t ) for t in  ax.get_xticks()], fontsize=10)
#     ax.set_yticklabels(['{:.2f}'.format(t ) for t in ax.get_yticks()], fontsize=10)
    
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Target: {}'.format(name), fontsize=22)
    ax.legend(loc="lower right", fontsize=18)
    return ax

def make_preds(ax, data, name):
    data = data.sort_values('pred', ascending=False)
    colors = ['r' if d == 1 else 'g' for d in data.liver]
    ax.scatter(list(range(data.shape[0])), data.pred, c=colors)
    ax.set_title('Target: {}'.format(name), fontsize=22)
    ax.set_ylabel('Pred', fontsize=18)
    return ax

from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score, \
                            cohen_kappa_score, matthews_corrcoef, precision_score, recall_score,\
                            confusion_matrix, roc_curve

class_scoring = {'ACC': make_scorer(accuracy_score), 'F1-Score': make_scorer(f1_score), 'AUC': make_scorer(roc_auc_score),
           'Cohen\'s Kappa': make_scorer(cohen_kappa_score), 'MCC': make_scorer(matthews_corrcoef),
           'Precision': make_scorer(precision_score), 'Recall': make_scorer(recall_score)}



def config_directory():
    """ add the toxcast_flask/dash directory to the notebook """
    import sys, os
    sys.path.insert(0, os.path.join('..', 'toxcast_flask'))