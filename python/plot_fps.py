import pandas as pd
import pymongo
import toxcast

true_css = pd.read_csv('../data/text/css_estimates_tox21.csv').set_index('cas')

# fulvestrant -> 129453-61-8
# benzocaine -> 94-09-7
# Oxybenzone -> 131-57-7
# 17a-Hydroxyprogesterone -> 630-56-8

chem_cas = '131-57-7'
chem_css = true_css.loc[chem_cas, 'estimates']
name = 'Oxybenzone'

client = pymongo.MongoClient()


pipe = [
    {"$match": {"$and": [{"_id.type": "human"}, {"_id.casn": {"$in": [chem_cas]}}, {"auc_norm": {"$gt": 0}}]}},
    { "$project": {
        "_id": 0,
        "mechanism": "$_id.mechanism",
        "receptor": "$_id.aop",
        "casn": "$_id.casn",
        "receptor_conc_score": "$responses_norm",
         "conc": "$concentrations"
    }}
]
receptor_scores = pd.DataFrame(list(client.LiverToxCast.aop_models.aggregate(pipe)))

cr_frame = receptor_scores.set_index(['receptor' , 'casn']).apply(pd.Series.explode).reset_index()
cr_frame['conc'] = cr_frame['conc'].astype(float)
cr_frame['AOP_Score'] = cr_frame['receptor_conc_score'].astype(float)


cr_frame = cr_frame.rename(columns={'conc': "Concentration", "receptor_conc_score": "Response", "receptor": "AOP"})

cr_frame[['Concentration', 'Response']] = cr_frame[['Concentration', 'Response']].astype(float)
#cr_frame['Concentration'] = np.log10(cr_frame.Concentration)

chem_aop = cr_frame[cr_frame.casn == chem_cas]


import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("ticks")
sns.set_context('poster')

sns.set_style("ticks")
sns.set_context('poster')

fig, axarr = plt.subplots(nrows=1, ncols=1, figsize=(8, 7), sharey=True)

# TAM
axarr = sns.lineplot(data=chem_aop,
                     x="Concentration",
                     y="Response",
                     ax=axarr,
                     legend=False,
                     label='Fluvestrant AOP Scores',
                     hue='AOP',
                     palette=['k'] * chem_aop.AOP.nunique(),
                     lw=3,
                     ls='-')

axarr.axvline(chem_css, color='red', ls='--', lw=3)

axarr.set(xscale="log")

axarr.set_ylim(0, 1)

yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

axarr.set_yticks(yticks)
axarr.set_yticklabels(yticks, fontsize=14)

xticks = axarr.get_xticks()[1:-2]

axarr.set_xticks(xticks)
axarr.set_xticklabels(xticks, fontsize=14)

axarr.set_ylabel('Response', fontsize=18)
axarr.set_xlabel('Concentration ($\mu$M)', fontsize=18)
# axarr[i].set_xlim(0.001, 1E5)


axarr.set_title(f'{name} KE Scores')

axarr.text(chem_css + 0.01, 0.95, '$C_{ss} = $' + ' ' + str(round(chem_css, 4)) + ' $\mu$M', ha='left')

axarr.fill_between([xticks[0], chem_css], 0, 1, color='red', alpha=0.1)

# plt.legend(handles=[], loc=2, prop={"size":20})
plt.tight_layout()
sns.despine(offset=10, trim=True)
plt.savefig(f'../data/figures/aop_css_plot_{name}.png', transparent=True)