
from collections import Counter
from statistics import mean
from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa
from typing import List
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
from util import util_common, util_ka_more, util_fleiss_more
from util.util_common import AgreeCalculator, SimuDataBuilder, MlaLogger

import pandas as pd
import numpy as np


def cal_cohen_kappa(logger:MlaLogger, anno_data:List[int], k):
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
    
    logger.add_log(f" N={N}")
    logger.add_log(f"ct={ct}")
    logger.add_log(f"matx\n {matx}")
    logger.add_log(f"po:    {po}")
    logger.add_log(f"pe:    {pe}")
    logger.add_log(f"kappa  {kappa}")
    logger.add_log(f"cohen_kappa_score  {cohen_kappa_score(c1, c2)}  {cohen_kappa_score(c2, c1)}")

    assert abs(cohen_kappa_score(c1, c2) - kappa) < 1e-5
    return kappa, po, pe, {"tp":tp, "tn":tn,"fp":fp,"fn":fn}

def __build_data_df(data:List[List[int]], mla_pos:List[float]):
    item_num  = len(data)
    coder_num = len(data[0])
    df = pd.DataFrame()
    df['item'] = list(range(1, 1+item_num))
    for i in range(coder_num):
        df[f'coder_{i+1}'] = [util_common.get_expr_of_arr(row[i]) for row in data]
    df['mla_po'] = mla_pos
    return df

def gen_anno_data_4_cohen(logger, num_tp:int, num_tn, num_fp, num_fn)->List[List[int]]:
    d_00 = [[0], [0]]
    d_01 = [[0], [1]]
    d_11 = [[1], [1]]
    d_10 = [[1], [0]]
    nums = [num_tp, num_tn, num_fp, num_fn]
    total = sum(nums)

    logger.add_log(F"gen_anno_data_4_cohen: tot_num:{total}   num_tp:{num_tp}    num_tn:{num_tn}    num_fp:{num_fp}  num_fn:{num_fn}")
    
    all_data = []
    for d_, num_ in zip([d_00, d_11, d_01, d_10], [num_tp, num_tn, num_fp, num_fn]):
        all_data.extend([d_]*(num_))
        
    all_data = resample(all_data, replace=False)
    assert np.array(all_data).shape == (total, 2, 1)
    return all_data

def _gen_coder_data_4_mul_class(tot_num, k, class_probs_of_coder:List[float]):
    assert abs(1-sum(class_probs_of_coder)) < 1e-5
    coder_data  = []
    coder_class_nums = [(int)(tot_num*p_) for p_ in class_probs_of_coder]
    for cls_, num_ in zip(range(1,k+1), coder_class_nums):
        # coder_data.extend([cls_]*(num_))
        # coder_data.extend(cls_*num_)
        for n_ in range(num_):
            coder_data.append([cls_])

    coder_data = resample(coder_data, replace=False)
    # assert len(coder_data) == tot_num
    assert np.array(coder_data).shape == (tot_num, 1)
    return coder_data

def gen_anno_data_4_mul_class(tot_num, k, probs:List[List[float]])->List[List[float]]:
    '''probs: row: coder, col: class prob
    '''
    assert k == len(probs[0])
    coder_data_all = []
    for row_ in probs:
        coder_data_all.append(_gen_coder_data_4_mul_class(tot_num,k,row_))
    rest_ = np.array(coder_data_all).T.reshape(tot_num, -1, 1)

    # TODO check   


    return rest_

def proc_2coder_2class(logger:MlaLogger, desc, k,anno_data, tot_simu_data, repeat):

    df_result = pd.DataFrame()
    def add_result_2_df(df_, name, ka, po, pe, type, note):
        df_ = df_.append({
            "name" : name,
            "kappa": ka,
            "p_o"  : po,
            "p_e"  : pe,
            "type" : type,
            "note" : note
        }, ignore_index=True)
        return df_

    kappa,po,pe,note = cal_cohen_kappa(logger, anno_data, k)
    assert kappa==util_common.cal_kappa(po, pe)

    ac = AgreeCalculator(logger, anno_data, k=k, mla_only=True)
    anno_mla_po, anno_mla_pos = ac.mla_po()
    df_anno = __build_data_df(anno_data, anno_mla_pos)
    logger.add_log(f"cohen kappa :{kappa:.3f}\tpo:{po:.3f}\tpe:{pe:.3f}")  
    df_result = add_result_2_df(df_result, "cohen", ka=kappa, po=po, pe=pe, type="anno",note=note)
        
    df_simus = []
    for r in range(repeat):
        mla_kappa, (mla_po, mla_pos), (mla_pe, mla_pes), simu_data = \
                ac.mla_kappa(tot_simu_data)
        assert mla_kappa==util_common.cal_kappa(mla_po, mla_pe)
        assert anno_mla_po == mla_po
        df_simus.append(__build_data_df(simu_data, mla_pes))
        logger.add_log(f"mla_kappa_{r+1:02}:{mla_kappa:.3f}\tpo:{mla_po:.3f}\tpe:{mla_pe:.3f}\tsimu_data_num:{tot_simu_data}")   
        df_result = add_result_2_df(df_result, f"mla_in_simu_{r+1:01}", ka=mla_kappa, po=mla_po, pe=mla_pe, 
            type="simu",note= f"total_simu_data_len:{tot_simu_data}")
    
    # select specific rows
    mean_vals = df_result.loc[1:].mean()
    mean_ka = mean_vals['kappa']
    mean_po = mean_vals['p_o']
    mean_pe = mean_vals['p_e']
    mean_note = f"simu_data_len:{tot_simu_data} repeat:{repeat}"
    logger.add_log(f"mla_kappa   :{mean_ka:.3f}\tpo:{mean_po:.3f}\tpe:{mean_pe:.3f}\t{note}")   
    df_result = add_result_2_df(df_result, f"average mla", ka=mean_ka, po=mean_po, pe=mean_pe, 
            type="result",note=mean_note)
    df_result = df_result[['name', 'kappa','p_o', 'p_e', 'type', 'note']]

    # common proc: result_str
    for df_ in [df_anno]+df_simus:
        df_['result_str'] = df_.iloc[:,1:-1].astype(str).agg(', '.join, axis=1)

    # add dfs to logger
    logger.add_df("anno_data", df_anno)
    for ind_, df_ in enumerate(df_simus):
        logger.add_df(F"simu_data_{ind_+1:02}", df_)
    logger.add_df("result", df_result)


def cal_krippendorff_alpla(ac:AgreeCalculator):
    '''return kf_alpha, (d_o_sum, d_e_sum), (d_o_avg, d_e_avg) 
    '''
    return ac.krippendorff_alpla()
 
def cal_mla_alpla(ac:AgreeCalculator, simu_data_len:int):
    '''return mla_kappa, (mla_po, mla_po_details), (mla_pe, mla_pe_details), simu_data 
    '''
    return  ac.mla_kappa(simu_data_len = simu_data_len)

def proc_ka_multiclass(logger, anno_data:List[List[int]], k:int, simu_data_len:int):
    tot = len(anno_data)*len(anno_data[0])

    ac = AgreeCalculator(logger, anno_data, k=k)
    alpha, d_o_sum, d_e_sum = ac.krippendorff_alpla()
    do_avg = d_o_sum/tot
    de_avg = d_e_sum/tot

    kappa_mla, (po_mla, po_mla_details), (pe_mla, pe_mla_details), simu_data = \
        ac.mla_kappa(simu_data_len = simu_data_len)

    print(f"krippendorff_alpla alpha:{alpha:.3f}\t  1-do_avg:{1-do_avg:.3f}\t  1-de_avg:{1-de_avg:.3f}")      
    print(f"mla                kappa:{kappa_mla:.3f}\t\tpo:{po_mla:.3f}\t\tpe:{pe_mla:.3f}\t\tsimu_data_num:{simu_data_len}")    


def proc_2coder_mul_class(logger, anno_data:List[List[int]], k:int, simu_data_len:int):

    desc = "2coder_mul_class"

    fleiss_input = util_common.convert_anno_data_to_fleiss(anno_data, k)
    fleiss_kappa, po, pe = util_fleiss_more.fleiss_kappa_return_more(fleiss_input, method="fleiss")

    ac = AgreeCalculator(logger, anno_data, k=k)

    kf_alpha, (d_o_sum, d_e_sum), (d_o_avg, d_e_avg)  = cal_krippendorff_alpla(ac)
    mla_kappa, (mla_po, mla_po_details), (mla_pe, mla_pe_details), simu_data = \
        cal_mla_alpla(ac, simu_data_len)

    __build_data_df(logger, desc, anno_data, simu_data, mla_po_details, mla_pe_details)

    df_result = pd.DataFrame()
    df_result['name'] = ['fleiss', 'kf_alpha', "mla"]
    df_result['kappa|alpha'] = [fleiss_kappa, kf_alpha, mla_kappa]
    df_result['p_o|1-do_avg'] = [po, 1-d_o_avg, mla_po]
    df_result['p_e|1-de_avg'] = [pe, 1-d_e_avg, mla_pe]
    df_result['note'] = [
        f"anno_data_len:{len(anno_data)}", 
        f"anno_data_len:{len(anno_data)}", 
        f"simu_data_len:{simu_data_len}"]
    logger.add_df(f"{desc}_result", df_result)

    logger.add_log(f"fleiss  kappa:{fleiss_kappa:+.3f}\t   po:{po:.3f}\t   pe:{pe:.3f}")
    logger.add_log(f"kridrof alpha:{kf_alpha:+.3f}  1-do_avg:{1-d_o_avg:.3f}  1-de_avg:{1-d_e_avg:.3f}")
    logger.add_log(f"mla     kappa:{mla_kappa:+.3f}\t   po:{mla_po:.3f}\t   pe:{mla_pe:.3f}\tsimu_data_num:{simu_data_len}")   
