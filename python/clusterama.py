""" 

Taken from https://github.com/PatWalters/clusterama

"""


import sys
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AtomPairs import Pairs


__all__ = [
    'ButinaCluster',
    'run',
]

class ButinaCluster:
    def __init__(self,fp_type="rdkit"):
        self.fp_type = fp_type

    def cluster_smiles(self,smi_list,sim_cutoff=0.8):
        mol_list = [Chem.MolFromSmiles(x) for x in smi_list]
        return self.cluster_mols(mol_list,sim_cutoff)

    def get_fps(self,mol_list):
        fp_dict = {
            "morgan2" : [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in mol_list],
            "rdkit" : [Chem.RDKFingerprint(x) for x in mol_list],
            "maccs" : [MACCSkeys.GenMACCSKeys(x) for x in mol_list],
            "ap" : [Pairs.GetAtomPairFingerprint(x) for x in mol_list]
            }
        if fp_dict.get(self.fp_type) is None:
            raise KeyError(f"No fingerprint method defined for {self.fp_type}")

        return fp_dict[self.fp_type]
    
    def cluster_mols(self,mol_list,sim_cutoff=0.8):
        dist_cutoff = 1.0 - sim_cutoff
        #fp_list = [rdmd.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mol_list]
        fp_list = self.get_fps(mol_list)
        dists = []
        nfps = len(fp_list)
        for i in range(1,nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fp_list[i],fp_list[:i])
            dists.extend([1-x for x in sims])
        mol_clusters = Butina.ClusterData(dists,nfps,dist_cutoff,isDistData=True)
        cluster_id_list = [0]*nfps
        for idx,cluster in enumerate(mol_clusters,1):
            for member in cluster:
                cluster_id_list[member] = idx
        return [x-1 for x in cluster_id_list]


def run(path, output, cutoff):
    df = pd.read_csv(path, sep=" ", names=["SMILES", "Name"])
    butina_cluster = ButinaCluster("rdkit")
    df['cluster'] = butina_cluster.cluster_smiles(df.SMILES.values, sim_cutoff=cutoff)
    df.sort_values("cluster", inplace=True)
    print(df.cluster.value_counts())
    df.to_csv(output, index=False)