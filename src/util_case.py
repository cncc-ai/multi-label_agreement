
from collections import Counter
from statsmodels.stats.inter_rater import fleiss_kappa
from typing import List
from sklearn.metrics import cohen_kappa_score
from util_common import AgreeCalculator, SimuDataBuilder


import pandas as pd
import numpy as np
import statistic
import util_common
import util_ka_more


def cal_cohen_kappa(anno_data:List[int],  k=2, verbose=False):
    def build_key(c1_, c2_):
        return f"{c1_}_{c2_}"
    
    c1 = [one[0][0] for one in anno_data]
    c2 = [one[1][0] for one in anno_data]
    
    ct = Counter()
    for c1_,c2_ in zip(c1,c2):
        ct.update({
            build_key(c1_, c2_): 1
        })
    
    matx = np.full((k,k), -1)                   
    for row in range(k):
        for col in range(k):
            matx[row][col] = ct[build_key(row, col)]            

    tp = matx[0][0]
    tn = matx[1][1]
    fp = matx[0][1]
    fn = matx[1][0]
    
    N = len(anno_data)
    po = (tp+tn)/N
    pe = (tp+fp)/N * (tp+fn)/N + (fn+tn)/N*(fp+tn)/N
    kappa = util_common.cal_kappa(po,pe)
    
    if verbose:
        print(f"c1={c1}")
        print(f"c2={c2}")
        print(f" N={N}")

        print(f"ct={ct}")
        print("matx\n", matx)
        print("po", po)
        print("pe", pe)
        print("kappa", kappa)
        print("cohen_kappa_score", cohen_kappa_score(c1, c2), cohen_kappa_score(c2, c1))

    assert abs(cohen_kappa_score(c1, c2) - kappa) < 1e-5
    return kappa, po, pe

def proc_cohen(anno_data, simu_data_len=4000, verbose=False):
    kappa,po,pe = cal_cohen_kappa(anno_data, verbose=verbose)
    assert kappa==util_common.cal_kappa(po, pe)
    
    ac = AgreeCalculator(anno_data, k=2, mla_only=True)
    mla_kappa, mla_po, mla_pe = ac.mla_kappa(simu_data_len, verbose=verbose)
    assert mla_kappa==util_common.cal_kappa(mla_po, mla_pe)

    # df = pd.DataFrame({
    #     # "anno_data":[build_str_for_anno_data(anno_data)],
    #     "anno_data":[anno_data],
    #     "cohen po":[po],444444444444444444
    #     "mla   po":[po_mla],

    # })
    # df

    print(f"anno_data:\n{util_common.build_str_for_anno_data(anno_data)}")
    print(f"cohen kappa:{kappa:.3f}\t\tpo:{po}\t\tpe:{pe:.3f}")      
    print(f"mla   kappa:{mla_kappa:.3f}\t\tpo:{mla_po}\t\tpe:{mla_pe:.3f}\t\tsimu_data_num:{simu_data_len}")   