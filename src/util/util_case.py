
from collections import Counter
from nltk import agreement
from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import jaccard_distance, masi_distance
from statistics import mean
from statsmodels.stats.inter_rater import fleiss_kappa, cohens_kappa
from typing import List
from sklearn.metrics import cohen_kappa_score
from sklearn.utils import resample
from util import util_common, util_ka_more, util_fleiss_more
from util.util_common import AgreeCalculator, SimuDataBuilder, MlaLogger, _build_data_df

import pandas as pd
import numpy as np


def gen_anno_data_4_cohen(logger, num_tp:int, num_tn, num_fp, num_fn)->List[List[int]]:
    d_00 = [[0], [0]]
    d_01 = [[0], [1]]
    d_11 = [[1], [1]]
    d_10 = [[1], [0]]
    nums = [num_tp, num_tn, num_fp, num_fn]
    total = sum(nums)

    logger.add_log(F"gen_anno_data_4_cohen: tot_num:{total}   num_tp:{num_tp}    num_tn:{num_tn}    num_fp:{num_fp}  num_fn:{num_fn}")
    
    all_data = []
    for d_, num_ in zip([d_00, d_11, d_10, d_01], [num_tp, num_tn, num_fp, num_fn]):
        all_data.extend([d_]*(num_))
        
    all_data = resample(all_data, replace=False)
    assert np.array(all_data).shape == (total, 2, 1)
    return all_data

def _gen_coder_data_4_mul_class(tot_num, k, class_probs_of_coder:List[float]):
    assert abs(1-sum(class_probs_of_coder)) < 1e-5, F"sum of class_probs_of_coder {class_probs_of_coder}, but it is {sum(class_probs_of_coder)}"
    coder_data  = []
    coder_class_nums = [(int)(tot_num*p_) for p_ in class_probs_of_coder]
    for cls_, num_ in zip(range(1,k+1), coder_class_nums):
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
    rest_ = np.array(coder_data_all, dtype=object).T.reshape(tot_num, -1, 1)
    return rest_

def proc_2coder_2class(logger:MlaLogger, desc, k,anno_data, total_simu_data, repeat):
    
    assert 2 ==len(anno_data[0])
    df_result = pd.DataFrame()
    
    kappa,po,pe,note_anno = util_common.cal_cohen_kappa(logger, anno_data, k)
    assert kappa==util_common.cal_kappa(po, pe)
    logger.add_section_result()
    logger.add_log(util_common.formated_result_4_kappa("cohen kappa", kappa, po, pe, note_anno))
    df_result = _add_result_2_df(df_result, "cohen kappa", ka=kappa, po=po, pe=pe, type="anno",note=note_anno)

    ac = AgreeCalculator(logger, anno_data, k=k, mla_only=False)
    alpha_total = len(anno_data)*len(anno_data[0])
    kf_alpha, (d_o_sum, d_e_sum), (d_o_avg, d_e_avg)  = util_common.cal_krippendorff_alpla(ac)
    assert abs(kf_alpha  - (1 - d_o_sum / d_e_sum )) < 1e-5
    assert abs(kf_alpha  - (1 - d_o_avg / d_e_avg )) < 1e-5
    assert d_o_avg == d_o_sum / alpha_total
    assert d_e_avg == d_e_sum / alpha_total
    logger.add_log(util_common.formated_result_4_alpha(kf_alpha, d_o_avg, d_e_avg, note_anno))
    df_result = _add_result_2_df(df_result, "kf alpha", ka=kf_alpha, po=1-d_o_avg, pe=1-d_e_avg, type="anno",note=note_anno)

    common_post_process(logger, ac, df_result, total_simu_data, repeat, is_kf_alpha_included=True)

def proc_ka_multiclass(logger, anno_data:List[List[List[int]]], k:int, simu_data_len:int):
    tot = len(anno_data)*len(anno_data[0])

    ac = AgreeCalculator(logger, anno_data, k=k)
    alpha, d_o_sum, d_e_sum = ac.krippendorff_alpla()
    do_avg = d_o_sum/tot
    de_avg = d_e_sum/tot

    kappa_mla, (po_mla, po_mla_details), (pe_mla, pe_mla_details), simu_data = \
        ac.mla_kappa(simu_data_len = simu_data_len)

    print(f"krippendorff_alpla alpha:{alpha:.3f}\t  1-do_avg:{1-do_avg:.3f}\t  1-de_avg:{1-de_avg:.3f}")      
    print(f"mla                kappa:{kappa_mla:.3f}\t\tpo:{po_mla:.3f}\t\tpe:{pe_mla:.3f}\t\tsimu_data_num:{simu_data_len}")    


def formated_result_4_kappa(name, kappa, po, pe, note):
    fmt_str = "{:10}:{:.3f}\t{:10}:{.3f}\t{:10}:{.3f}\t{:40}"
    return fmt_str.format(name,  kappa, "po", po, "pe", pe, note)


def formated_result_4_alpha(kf_alpha, do_avg, de_avg, note):
    fmt_str = "{:10}:{:.3f}\t{:10}:{.3f}\t{:10}:{.3f}\t{:40}"
    return fmt_str.format("kf_alpha",  kf_alpha, "1-do_avg", 1-do_avg, "1-de_avg", 1-de_avg, note)


def _add_result_2_df(df_, name, ka, po, pe, type, note):
    df_ = df_.append({
        "name" : name,
        "kappa": ka,
        "p_o"  : po,
        "p_e"  : pe,
        "type" : type,
        "note" : note
    }, ignore_index=True)
    return df_


def _simu_data(logger, ac, df_result, tot_simu_data, repeat):
    df_simus = []
    existing_num = len(df_result)
    for r in range(repeat):
        mla_kappa, (mla_po, mla_pos), (mla_pe, mla_pes), simu_data = \
                ac.mla_kappa(tot_simu_data)
        assert mla_kappa==util_common.cal_kappa(mla_po, mla_pe)
        
        df_simus.append(_build_data_df(simu_data, mla_pes))
        # logger.add_log(f"mla_kappa_{r+1:02}:{mla_kappa:.3f}\tpo:{mla_po:.3f}\tpe:{mla_pe:.3f}\tsimu_data_num:{tot_simu_data}")   
        logger.add_log(util_common.formated_result_4_kappa(
            f"mla_kappa_{r+1:02}", mla_kappa, mla_po, mla_pe, f"tot_simu_data:{tot_simu_data}"))
        df_result = _add_result_2_df(
            df_result, f"mla_in_simu_{r+1:02}", ka=mla_kappa, po=mla_po, pe=mla_pe, 
            type="simu",note= f"total_simu_data_len:{tot_simu_data}")
    
    # select specific rows
    mean_vals = df_result.loc[existing_num:].mean()
    mean_ka = mean_vals['kappa']
    mean_po = mean_vals['p_o']
    mean_pe = mean_vals['p_e']
    mean_note = f"tot_simu_data:{tot_simu_data} repeat:{repeat}"
    # logger.add_log(f"mla_kappa   :{mean_ka:.3f}\tpo:{mean_po:.3f}\tpe:{mean_pe:.3f}\t{mean_note}")   
    logger.add_log(util_common.formated_result_4_kappa(
        "mla_kappa", mean_ka, mean_po, mean_pe, mean_note))
    df_result = _add_result_2_df(
        df_result, f"average mla", ka=mean_ka, po=mean_po, pe=mean_pe, 
        type="result",note=mean_note)
    return  df_result, df_simus

def common_post_process(logger, ac, df_result, total_simu_data, repeat, is_kf_alpha_included=False):
    
    anno_mla_po, anno_mla_pos = ac.mla_po()
    df_anno = _build_data_df(ac.get_anno_data(), anno_mla_pos)
    df_result, df_simus = _simu_data(logger, ac, df_result, total_simu_data, repeat)

    df_result = df_result[['name', 'kappa','p_o', 'p_e', 'type', 'note']]
    if (is_kf_alpha_included):
        df_result = df_result.rename(columns={'kappa': 'kappa | kf_alpha', 'p_o': 'p_o | 1-do_avg', 'p_e': 'p_e| 1-de_avg'})

    # common proc: result_str
    for df_ in [df_anno]+df_simus:
        df_['result_str'] = df_.iloc[:,1:-1].astype(str).agg('|'.join, axis=1)

    # add dfs to logger
    logger.add_df("anno_data", df_anno)
    for ind_, df_ in enumerate(df_simus):
        logger.add_df(F"simu_data_{ind_+1:02}", df_)
    logger.add_df("result", df_result)


def proc_multi_coder_multi_class(logger, desc, anno_data:List[List[List[int]]], 
        k:int, total_simu_data:int, repeat:int):
    df_result = pd.DataFrame()
 
    n_coder = len(anno_data[0])
    note_anno = f"n_coder={n_coder} k={k} total={len(anno_data)}"
    
    logger.add_section_result()
    if (n_coder == 2):
        fleiss_input = util_common.convert_anno_data_to_fleiss(anno_data, k)
        fleiss_kappa, po, pe = util_fleiss_more.fleiss_kappa_return_more(fleiss_input, method="fleiss")
        assert fleiss_kappa ==  util_common.cal_kappa(po, pe)
        logger.add_log(util_common.formated_result_4_kappa("fleiss kappa", fleiss_kappa, po, pe, note_anno))
        df_result = _add_result_2_df(df_result, "fleiss kappa", ka=fleiss_kappa, po=po, pe=pe, type="anno",note=note_anno)
    
    alpha_total = len(anno_data)*len(anno_data[0])
    ac = AgreeCalculator(logger, anno_data, k=k)
    kf_alpha, (d_o_sum, d_e_sum), (d_o_avg, d_e_avg)  = util_common.cal_krippendorff_alpla(ac)
    assert abs(kf_alpha  - (1 - d_o_sum / d_e_sum )) < 1e-5
    assert abs(kf_alpha  - (1 - d_o_avg / d_e_avg )) < 1e-5
    assert d_o_avg == d_o_sum / alpha_total
    assert d_e_avg == d_e_sum / alpha_total
    logger.add_log(util_common.formated_result_4_alpha(kf_alpha, d_o_avg, d_e_avg, note_anno))
    df_result = _add_result_2_df(df_result, "kf alpha", ka=kf_alpha, po=1-d_o_avg, pe=1-d_e_avg, type="anno",note=note_anno)

    common_post_process(logger, ac, df_result, total_simu_data, repeat, is_kf_alpha_included=True)


def proc_multi_coder_multi_label(logger, desc, anno_data:List[List[List[int]]], 
        k:int, total_simu_data:int, repeat:int):
    df_result = pd.DataFrame()
    logger.add_section_result()
    ac = AgreeCalculator(logger, anno_data, k=k, mla_only=True)
    common_post_process(logger, ac, df_result, total_simu_data, repeat, is_kf_alpha_included=False)


def proc_po_in_2coder_multi_label(logger:MlaLogger, desc, one_data):
    logger.add_section_result()
    df = pd.DataFrame()
    for index_, one_data in enumerate(one_data):
        set0 = set(one_data[0])
        set1 = set(one_data[1])       
        mla,_= util_common.cal_mla_of_one_item(one_data)
        
        df = df.append({
            'item': index_+1,
            'data': str(one_data),
            'aug_kappa':           util_common.cal_po_by_aug_ka(one_data),
            'f1':                  util_common.cal_po_by_f1(one_data),
            "1-jaccard_distance" : 1-jaccard_distance(set0, set1),
            "1-masi_distance" :    1-masi_distance(set0, set1),
            'mla': mla
        }, ignore_index=True)
        df = df [['item', 'data', 'aug_kappa', 'f1', '1-jaccard_distance', "1-masi_distance", 'mla']]
        
    logger.add_log('\n'+str(df))
    logger.add_df("result",df)