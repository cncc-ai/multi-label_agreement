

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import List
from pandas.core.indexes.base import Index
from sklearn.utils import resample
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

import collections
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
import statistics
from util import util_ka_more

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
        # pdb.set_trace()
        assert stat_prob(col_init_data) == stat_prob(simu_col_data) 
        return simu_col_data
    

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
        return datetime.now().strftime("%m%d%H%M%S")

    def __get_time(self)->str:
        return f'{datetime.now().strftime("%m%d%H%M")}'

    def add_log(self, log:str):
        str_ = f'[{self.__get_time()}]  {log}'
        self.logs.append(str_)
        print(f"{self.inst_id} {str_}")

    def add_df(self, sheet_name:str, df:pd.DataFrame):
        self.sheet_df[sheet_name] = df

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

if __name__ == "__main__":
    print("D")