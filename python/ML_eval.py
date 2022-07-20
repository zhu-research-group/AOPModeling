import pandas as pd

hybrid_model_preds = pd.read_csv('/Users/danielrusso/projects/toxcast2/data/text/ML/H_HT_class_LOG:False_BALANCE:True_SPLIT:LOO/hybrid_preds.csv', index_col=0)
aop_model_preds = pd.read_csv('/Users/danielrusso/projects/toxcast2/data/text/ML/H_HT_class_LOG:False_BALANCE:True_SPLIT:LOO/aop_preds.csv', index_col=0)
model_features = pd.read_csv('/Users/danielrusso/projects/toxcast2/data/text/ML/H_HT_class_LOG:False_BALANCE:True_SPLIT:LOO/features.csv', index_col=0)
true_css = pd.read_csv('/Users/danielrusso/projects/toxcast2/data/text/css_estimates_tox21.csv').set_index('cas')

#filder out good preds

aop_model_preds = aop_model_preds[aop_model_preds['Prediction Class'] != aop_model_preds.Hepatotoxicity]

print(aop_model_preds.shape[0], "compounds were incorrectly prediction by AOP model")

incorrect_aop = aop_model_preds.index

hybrid = hybrid_model_preds.loc[incorrect_aop]

hybrid_corrected = hybrid[hybrid['Prediction Class'] == hybrid['Hepatotoxicity']]

print(hybrid_corrected.shape[0], "of these were corrected by including TK info")

hybrid_corrected_fps = hybrid_corrected[hybrid_corrected['Hepatotoxicity'] == 0]
hybrid_corrected_fns = hybrid_corrected[hybrid_corrected['Hepatotoxicity'] == 1]


print(len(hybrid_corrected_fps), "FPS corrected")
fps_desc = model_features.loc[hybrid_corrected_fps.index].mean() - model_features.mean()
print(true_css.loc[hybrid_corrected_fps.index, 'estimates'].sort_values())
print(true_css.loc[model_features.index, 'estimates'].mean())


print(true_css['estimates'].mean())
print((true_css['estimates'].quantile(0.25)))


print(model_features.loc[hybrid_corrected_fps.index].iloc[:, 0:-20].max(1))


# AOP scores
import pymongo
import toxcast
client = pymongo.MongoClient(toxcast.MONGO_HOST, 27017)

pipe = [
    {"$match": {"_id.casn": {"$in": hybrid_corrected_fps.index.tolist()}}},
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

human_model_scores = pd.DataFrame(client.LiverToxCast.aop_models.aggregate(pipe))

# set to zero
human_model_scores.loc[human_model_scores.auc_score <= 0, 'auc_score'] = 0

hs = human_model_scores.pivot(index='casn', values='auc_score', columns='aop')

print((hs.loc[hybrid_corrected_fps.index] > 0).sum(1))


# print(len(hybrid_corrected_fns), "FNS corrected")
# fns_desc = model_features.loc[hybrid_corrected_fns.index].mean() - model_features.mean()
# print(true_css.loc[hybrid_corrected_fns.index, 'estimates'].sort_values())
# print(true_css.loc[model_features.index, 'estimates'].mean())
#
#
#
# print(true_css['estimates'].mean())
# print((true_css['estimates'].quantile(0.25)))
#
#
# print(model_features.loc[hybrid_corrected_fns.index].iloc[:, 0:-20].max(1))