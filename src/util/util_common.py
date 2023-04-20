

from __future__ import annotations

from collections import Counter
from datetime import datetime

from typing import List
from sklearn.utils import resample
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from util import util_ka_more

import argparse
import collections
import json
import numpy as np
import numpy.typing as npt
import os
import random
import pandas as pd
import statistics

def get_expr_of_arr(arr):
    return ",".join([str(one) for one in sorted(arr)])

def stat_prob(data:List[int|List[int]])->dict:
    '''data: list of class index. eg. [1,2,3,2...] or [[1,2], [3], 4]'''
    
    ct = Counter()
    for each_ in data:
        if isinstance(each_, collections.abc.Sequence):
            ct.update({get_expr_of_arr(each_):1})
        else:
            ct.update({str(each_):1})
    
    # pdb.set_trace()
    tot = len(data)
    stat_dict = {}
    # pdb.set_trace()
    for key_ in ct.keys():
        stat_dict[key_] = ct[key_] / tot
    return stat_dict

class SimuDataBuilder(object):
    def __init__(self, init_data):
        ''' 
        sample_data: [
                        [[],[],..],
                        [[],[],..],
                    ]        
        '''
        self.init_data = init_data
        self.init_data_len = len(init_data)
    
    def simu_data(self, total_simu:int):
        assert (total_simu % self.init_data_len) == 0, f"tot_simu:{total_simu} init_data_len:{self.init_data_len}"
        simu_data_cols = []
        for col_index in range(len(self.init_data[0])):
            simu_data_cols.append(self.__rand_column(col_index, total_simu))

        shape_len = len(np.array(self.init_data).shape)
        if (shape_len == 2):
            return np.array(simu_data_cols).T.reshape(total_simu, len(self.init_data[0]))
        else:
            return np.array(simu_data_cols).T.reshape(total_simu, len(self.init_data[0]), -1)
        
    def __rand_column(self, column_index:int, tot_number:int) -> List[int|List[int]]:
        all_data = []
        # pdb.set_trace()
        col_init_data = [row_[column_index] for row_ in self.init_data]
        for ind_ in range((int)(tot_number/self.init_data_len)):
            # all_data.extend(col_init_data)
            for row_ in col_init_data:
                all_data.append(row_)
            
        simu_col_data = resample(all_data, replace=False)
        assert len(simu_col_data) == tot_number
        assert stat_prob(col_init_data) == stat_prob(simu_col_data) 
        return simu_col_data
    
class MultiLabelDataBuilder(object):
    def __init__(self, tot_data, k, probs:List):
        '''probs:[one_coder_prob] 
        one_coder_prob: format [(probs of single class), double-class probs, 3-class probs]
            e.g. [(.3,.3,.3),.25,.05]
        '''
        self.tot_data = tot_data
        self.probs = probs
        self.k = k
        self.max_sample_fail_num = 100

    def _gen_data(self, current_data:List[int], num_of_label_in_anno:int, gen_data_count:int):
        
        gen_data = []
        while (len(gen_data) < gen_data_count):
            gen_data.append(self._gen_one_data(current_data, num_of_label_in_anno))
        assert len(gen_data) ==gen_data_count
        return gen_data

    def _gen_one_data(self, current_data:List[int], num_of_label_in_anno:int):
        current_data_types = set(current_data)
        assert len(current_data_types) >= num_of_label_in_anno, \
            f"num_of_label_in_anno={num_of_label_in_anno}, current_data_types={current_data_types}"

        selected_labels = []

        fail_num = 0
        while (len(selected_labels)<num_of_label_in_anno and fail_num < self.max_sample_fail_num):
            sample_ = random.choice(current_data)
            if sample_ in selected_labels:
                fail_num += 1
                continue
            selected_labels.append(sample_)
            current_data.remove(sample_)

        assert fail_num < self.max_sample_fail_num, \
            f"fail_num:{fail_num}, num_of_label_in_anno:{num_of_label_in_anno}"
        return sorted(selected_labels)

    def _build_4_coder(self, one_coder_prob:List):
        tot_data = self.tot_data

        prob_class_1 = one_coder_prob[0]  # List
        assert self.k >= len(prob_class_1)
        prob_class_2 = one_coder_prob[1]  # float
        prob_class_3 = one_coder_prob[2]  # float

        current_data = []
        for num_, prob_ in enumerate(prob_class_1):
            current_data.extend([num_+1]*(int)(tot_data*prob_))
        assert self.tot_data == len(current_data)

        coder_data = []
        for prob_, num_label_in_anno_ in zip([prob_class_3, prob_class_2], [3,2]):
            gen_data_count = (int)(tot_data*prob_)
            data_3 = self._gen_data(current_data, num_label_in_anno_, gen_data_count)
            coder_data.extend(data_3)

        # coder_data.extend(current_data)
        while (len(coder_data)<self.tot_data):
            sample_ = random.choice(current_data)
            coder_data.append([sample_])

        assert len(coder_data) == self.tot_data
        return resample(coder_data, replace=False)

    def build_data(self):
        data_coders = []
        coder_num = len(self.probs)
        for coder_prob in self.probs:
            data_coders.append(self._build_4_coder(coder_prob))

        return np.array(data_coders).T.reshape(len(data_coders[0]), coder_num, )
        # return data_coders


def cal_kappa(po, pe):
    return (po-pe)/(1-pe)

def build_reliability_data(reliability_data_str:set())->List[int]:
    return [[np.nan if v == "*" else int(v) for v in coder.split()] for coder in reliability_data_str]

def build_value_counts(anno_data:List[List[int]], k:int)->List[int]:
    '''
    anno_data: [
        [[],[],...],
        [[],[],...],
    ]
    '''
    val_data = []
    ct = Counter()
    for row_ in anno_data:
        ct.clear()
        for each_ in row_:
            ct.update(each_)
        if k==2:
            # binary 0,1
            val_data.append([ct[one] for one in range(0,k)])  
        else:
            val_data.append([ct[one] for one in range(1,k+1)])
        
    return val_data

def cal_mla_of_one_item(one_item_anno:list):
    '''
    one_item_anno: [[],[],...]
    '''
    ct = Counter()
    for one_anno in one_item_anno:
        ct.update(one_anno)
    
    n = len(one_item_anno)
    mla = []
    for one_anno in one_item_anno:
        mla.append(statistics.mean([(ct[cls_] - 1) / (n-1) for cls_ in one_anno] ))
    return statistics.mean(mla), mla


def build_str_for_anno_data(show_anno_data, space_width:int=5):
    ret_str = []
    for row_ in show_anno_data:
        ret_str.append ("".join([F"{str(each_anno):>{space_width}}" for each_anno in row_]))
    return "\n".join(ret_str)

class AgreeCalculator(object):
    def __init__(self, logger, anno_data, k, mla_only=False):
        '''
        anno_data: 
        k: class number
        '''
        self.logger = logger
        self.k = k
        self.anno_data = anno_data
        if (not mla_only):
            self.value_counts = build_value_counts(anno_data, k)
    
    def krippendorff_alpla(self):
        anno_data = self.anno_data
        tot = len(anno_data)*len(anno_data[0])
        alpha, d_o_sum, d_e_sum = util_ka_more.alpha_with_dode(
            value_counts=self.value_counts, level_of_measurement="nominal")
        return alpha, (d_o_sum, d_e_sum), (d_o_sum/tot, d_e_sum/tot)
    
    def mla_po(self):
        all_mla = []
        for row_ in self.anno_data:
            val_ = cal_mla_of_one_item(row_)
            all_mla.append(val_[0])
        return statistics.mean(all_mla), all_mla
    
    def mla_kappa(self, simu_data_len:int):
        logger = self.logger
        po, mlas_po = self.mla_po()
        data_builder = SimuDataBuilder(self.anno_data)
        simu_data = data_builder.simu_data(simu_data_len)
                
        ac = AgreeCalculator(logger, simu_data, self.k, mla_only=True)
        pe, mlas_pe = ac.mla_po()
        kappa = cal_kappa(po, pe)
        return kappa, (po, mlas_po), (pe, mlas_pe), simu_data

    def get_anno_data(self):
        return self.anno_data

class MlaLogger(object):

    def __init__(self, inst_id:str, desc="NoDesc", log_parent_dir:str=None):
        self.inst_id = inst_id
        self.sheet_df = {}
        self.logs = []

        if not log_parent_dir:
            cur_dir = os.path.dirname(__file__)
            log_parent_dir = os.path.join(cur_dir, "../../logs")

        log_path = os.path.join(log_parent_dir, f"{desc}_{inst_id}")
        os.makedirs(log_path)
        self.log_path = os.path.abspath(log_path)

        self.log_files = [
            os.path.join(log_path, f"mla_{self.inst_id}.log"),
            os.path.join(log_path, f"mla_{self.inst_id}.xlsx")]

        print(f"log path: {self.log_path}")

    @staticmethod
    def get_inst_id()->str:
        return datetime.now().strftime("%d%H%M%S")

    def __get_time(self)->str:
        return f'{datetime.now().strftime("%m%d%H%M%S")}'

    def add_log(self, log:str):
        str_ = f'[{self.__get_time()}]  {log}'
        self.logs.append(str_)
        print(f"{self.inst_id} {str_}")

    def add_df(self, sheet_name:str, df:pd.DataFrame):
        self.sheet_df[sheet_name] = df

    def __add_section(self, section_name):
        self.add_log('*'*30 + "{:^15}".format(section_name) + '*'*30)
    def add_section_input(self):
        self.__add_section("input & proc")
    def add_section_result(self):
        self.__add_section("result")

    def save(self):
        files = self.log_files

        with open(files[0], "w") as log_writer:
            log_writer.write("\n".join(self.logs))

        if (self.sheet_df):
            with pd.ExcelWriter(files[1]) as writer:
                for sheet_name, df_ in self.sheet_df.items():
                    df_.to_excel(writer, sheet_name=sheet_name, index=False)
        
        print(f"DONE! See results in {self.log_path}/")

def convert_anno_data_to_fleiss(anno_data:List[List[int]], k):
    assert len(anno_data[0]) == 2 # only 2 coders allowed for comparision

    ct = Counter()
    fleiss_input = []
    for row in anno_data:
        ct.clear()
        for val in row:
            ct.update(val)
        assert 0 not in ct.keys()  # class range from 1 to k
        fleiss_input.append([ct[one] for one in range(1, k+1)])
    return fleiss_input


def formated_result_4_kappa(name, kappa, po, pe, note):
    fmt_str = "{:>15}:{:+.3f}  {:>10}:{:.3f}  {:>10}:{:.3f}  {}"
    return fmt_str.format(name,  kappa, "po", po, "pe", pe, note)

def formated_result_4_alpha(kf_alpha, do_avg, de_avg, note):
    fmt_str = "{:>15}:{:+.3f}  {:>10}:{:.3f}  {:>10}:{:.3f}  {}"
    return fmt_str.format("kf_alpha",  kf_alpha, "1-do_avg", 1-do_avg, "1-de_avg", 1-de_avg, note)

def formated_str_4_var(var_name, val):
    return  "{:>15}:{}".format(var_name,  val)

def parse_2d_array(string):
    try:
        array = json.loads(string)
        assert isinstance(array, list)
        assert all(isinstance(row, list) for row in array)
        return array
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError('Invalid 2D array')

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
    kappa = cal_kappa(po,pe)
    
    logger.add_log(f" N={N}")
    logger.add_log(f"ct={ct}")
    logger.add_log(f"matx\n {matx}")
    logger.add_log(f"po:    {po}")
    logger.add_log(f"pe:    {pe}")
    logger.add_log(f"kappa  {kappa}")
    logger.add_log(f"cohen_kappa_score  {cohen_kappa_score(c1, c2)}  {cohen_kappa_score(c2, c1)}")

    assert abs(cohen_kappa_score(c1, c2) - kappa) < 1e-5
    return kappa, po, pe, {"tp":tp, "tn":tn,"fp":fp,"fn":fn}

def cal_krippendorff_alpla(ac:AgreeCalculator):
    '''return kf_alpha, (d_o_sum, d_e_sum), (d_o_avg, d_e_avg) 
    '''
    return ac.krippendorff_alpla()
 
def cal_mla_alpha(ac:AgreeCalculator, simu_data_len:int):
    '''return mla_kappa, (mla_po, mla_po_details), (mla_pe, mla_pe_details), simu_data 
    '''
    return  ac.mla_kappa(simu_data_len = simu_data_len)

def cal_po_by_aug_ka(anno_data:List[List[int]])->float:
    assert len(anno_data) == 2
    set_0 = set(anno_data[0])
    set_1 = set(anno_data[1])
    instersect_labels = set_0 & set_1
    
    kappa = 0
    for label_ in instersect_labels:
        kappa += 1/len(anno_data[0])*1/len(anno_data[1])
    return kappa

def cal_po_by_f1(anno_data:List[List[int]])->float:
    assert len(anno_data) == 2
    set_0 = set(anno_data[0])
    set_1 = set(anno_data[1])
    instersect_labels = set_0 & set_1
    
    r = len(instersect_labels) / len(set_0)
    p = len(instersect_labels) / len(set_1)

    if (p+r) == 0:
        assert p==0 and r==0
        return 0
    f1 = 2*p*r/(p+r)        
    return f1

if __name__ == "__main__":
    print("D")