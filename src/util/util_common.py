

from __future__ import annotations

from collections import Counter
from datetime import datetime

from typing import List
from sklearn.utils import resample
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
from tqdm import tqdm
from tqdm._tqdm import trange
from util import util_ka_more

import argparse
import collections
import copy
import json
import numpy as np
import numpy.typing as npt
import os
import pandas as pd
import random
import statistics
import time


def _build_data_df(data:List[List[int]], mla_po:List[float]=None):
    item_num  = len(data)
    coder_num = len(data[0])
    df = pd.DataFrame()
    df['item'] = list(range(1, 1+item_num))
    for i in range(coder_num):
        df[f'coder_{i+1:02}'] = [get_expr_of_arr(row[i]) for row in data]
    if mla_po:
        df['mla_po'] = mla_po
    return df

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
            return np.array(simu_data_cols, dtype=object).T.reshape(total_simu, len(self.init_data[0]))
        else:
            return np.array(simu_data_cols, dtype=object).T.reshape(total_simu, len(self.init_data[0]), -1)
        
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

def count_data_by_len(data:List[int], len_:int):
    return sum([1 if len(one)==len_ else 0 for one in data ])

def count_data_by_val(data:List[int], val_:int):
    # l_ = [1 if val_ in one else 0 for one in data ]
    # print("val="+str(val_), l_)
    return sum([1 if val_ in one else 0 for one in data ])
    
def count_1data_by_val(data:List[int], val_:int):
    return sum([1 if val_ in one and len(one)==1 else 0 for one in data ])

class MultiLabelDataBuilder(object):
    def __init__(self, logger, tot_data, k, probs:List):
        '''probs:[one_coder_prob] 
        one_coder_prob: format [(probs of single class), double-class probs, 3-class probs]
            e.g. [(.3,.3,.3),.25,.05]
        '''
        self.tot_data = tot_data
        self.probs = probs
        self.k = k
        self.max_sample_fail_num = 100
        self.logger = logger

    def _gen_data(self, current_data:List[int], num_of_label_in_anno:int, gen_data_count:int):
        
        gen_data = []
        while (len(gen_data) < gen_data_count):
            gen_data.append(self._gen_one_data(current_data, num_of_label_in_anno))
        assert len(gen_data) == gen_data_count
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

        prob_single_class = one_coder_prob[0]  # List
        assert self.k >= len(prob_single_class)
        prob_double_class = one_coder_prob[1]  # float
        prob_triple_class = one_coder_prob[2]  # float
        double_class_total = (int)(tot_data*prob_double_class)
        triple_class_total = (int)(tot_data*prob_triple_class)
        single_class_total = tot_data - double_class_total - triple_class_total

        pool_data = []
        pool_data_n = 2
        
        # build pool
        assert abs(1-sum(prob_single_class)) < 1e-5, F"sum of prob_single_class {prob_single_class}, but it is {sum(prob_single_class)}"
        for num_, prob_ in enumerate(prob_single_class):
            pool_data.extend([num_+1]*(int)(tot_data*pool_data_n*prob_))
        assert self.tot_data*pool_data_n == len(pool_data)

        coder_data = []
        stack_data = []
        double_class_count = 0
        triple_class_count = 0
        single_class_count = 0

        def is_single_OK():
            nonlocal single_class_count
            nonlocal single_class_total
            return single_class_count == single_class_total
        def is_double_OK():
            nonlocal double_class_count
            nonlocal double_class_total
            return double_class_count == double_class_total
        def is_triple_OK():
            nonlocal triple_class_count
            nonlocal triple_class_total
            return triple_class_count == triple_class_total

        def gen_proc_fail_msg(val):
            nonlocal single_class_count
            nonlocal single_class_total
            nonlocal double_class_count
            nonlocal double_class_total
            nonlocal triple_class_count
            nonlocal triple_class_total
            nonlocal pool_data
            msg = F"Fail to proc {val}. Details: pool stat:{stat_prob(pool_data)}" \
                    + F" single:{single_class_count}/{single_class_total}" \
                    + F" double:{double_class_count}/{double_class_total}" \
                    + F" triple:{triple_class_count}/{triple_class_total}"
            return msg
        def add_to_coder_data(val):
            nonlocal single_class_count
            nonlocal single_class_total
            nonlocal double_class_count
            nonlocal double_class_total
            nonlocal triple_class_count
            nonlocal triple_class_total

            nonlocal stack_data
            nonlocal coder_data
            nonlocal tot_data

            if tot_data == (single_class_count + double_class_count + triple_class_count):
                return

            # if isinstance(val, int):
            #     a = "3"
            #     pass
            n_class = len(val)
            if (n_class == 1):
                if is_single_OK():
                    raise ProcFail(message=gen_proc_fail_msg(val), errors = None)
                coder_data.append(val)
                stack_data.remove(val[0])
                single_class_count += 1
            elif(n_class == 2):
                if is_double_OK():
                    raise ProcFail(message=gen_proc_fail_msg(val), errors = None)
                coder_data.append(sorted([val[0], val[1]]))
                for val_ in val:
                    stack_data.remove(val_)
                double_class_count += 1
            else:
                assert n_class == 3
                if is_triple_OK():
                    raise ProcFail(message=gen_proc_fail_msg(val), errors = None)
                coder_data.append(sorted([val[0], val[1], val[2]]))
                for val_ in val:
                    stack_data.remove(val_)
                triple_class_count += 1
        def add_to_stack(val):
            nonlocal stack_data
            assert val not in stack_data
            stack_data.append(val)

        while (len(coder_data) < tot_data):
            if len(stack_data) != len(set(stack_data)):
                a = "3"

            assert len(stack_data) == len(set(stack_data))

            sample_ = random.choice(pool_data)
            pool_data.remove(sample_)
            if (len(stack_data) == 0):
                stack_data.append(sample_)
                continue

            if (sample_ not in stack_data):
                if (len(stack_data) == 3):
                    if not is_triple_OK():
                        add_to_coder_data([one for one in stack_data])
                    elif not is_double_OK():
                        add_to_coder_data([one for one in stack_data[:2]])
                    else:
                        for one in stack_data:
                            add_to_coder_data([one])
                            
                stack_data.append(sample_)
                continue

            # sample is in the stack
            # print(f"## {len(stack_data)} sample_:{sample_}, triple:{is_triple_OK()} double:{is_double_OK()}")
            if len(stack_data) == 1:
                add_to_coder_data([stack_data[0]])                
            elif len(stack_data) == 2:
                if not is_triple_OK():
                    add_to_coder_data([sample_])                    
                elif not is_double_OK():
                    add_to_coder_data([one for one in stack_data]) # add double
                else:
                    add_to_coder_data([sample_])    
            else:
                assert len(stack_data) == 3
                if not is_triple_OK():
                    add_to_coder_data([one for one in stack_data])
                elif not is_double_OK():
                    # find the different data
                    diff_data = None
                    for one_ in stack_data:
                        if sample_ != one_:
                            diff_data = one_
                            break
                    add_to_coder_data([sample_, diff_data])
                else:
                    clone_data = copy.deepcopy(stack_data)
                    for one in clone_data:
                        add_to_coder_data([one])
                        if len(coder_data) == tot_data:
                            break
                    if len(coder_data) == tot_data:
                        break

            assert sample_ not in stack_data
            stack_data.append(sample_)            

        assert self.tot_data == len(coder_data)
 
        log_msgs = []
        log_msgs.append(F"one_coder_prob: {one_coder_prob}")
        class_n_nums = [count_data_by_val(coder_data, n) for n in range(1, self.k+1)]
        log_msgs.append(F"class_n_nums: {class_n_nums}")
        
        stand_coder_prob = [round(one/min(prob_single_class),3) for one in prob_single_class]
        stand_class_n_nums = [round(one/min(class_n_nums),3) for one in class_n_nums]
        self.logger.add_log(". ".join(log_msgs))
        self.logger.add_log(F"standardized: {stand_coder_prob} {stand_class_n_nums}")
        max_stand_diff = max([abs(set_-gen_)/set_ for set_, gen_ in zip(stand_coder_prob, stand_class_n_nums)])
        self.logger.add_log(F"max single class standardized-prob diff:{max_stand_diff:.3f}")

        return resample(coder_data, replace=False)

    def build_data(self):
        data_coders = []
        coder_num = len(self.probs)
        for coder_prob in self.probs:
            data_coders.append(self._build_4_coder(coder_prob))

        return np.array(data_coders, dtype=object).T.reshape(len(data_coders[0]), coder_num, )


def check_anno_simu_num(anno_num, simu_num):
    assert (simu_num % anno_num) == 0, F"The simulated data number {simu_num} must be an integer multiple of the annotated data number {anno_num}."

def change_anno_data_2_data_coders(anno_data:List[List[List[int]]]):
    item_num = len(anno_data)
    coder_num = len(anno_data[0])
    return np.array(anno_data, dtype=object, copy=False).T.reshape(coder_num, item_num,).tolist()
    

def change_data_coders_2_anno_data(data_coders:List[List[List[int]]]):
    coder_num = len(data_coders)
    item_num = len(data_coders[0])
    return np.array(data_coders, dtype=object, copy=False).T.reshape(item_num, coder_num,).tolist()


def swap_anno_data_in_place(anno_data:List[List[List[int]]], row_index1, row_index2, col_index):

    row_data_1 = anno_data[row_index1]
    row_data_2 = anno_data[row_index2]
    prev_val_src = anno_data[row_index1][col_index]
    prev_val_tar = anno_data[row_index2][col_index]
    # print("@@@@-bef", prev_val_src, prev_val_tar)
        
    tmp_val = copy.deepcopy(row_data_1[col_index])
    # row_data_1[col_index], row_data_2[col_index] = \
    #     row_data_2[col_index], row_data_1[col_index]
    row_data_1[col_index] = row_data_2[col_index]
    row_data_2[col_index] = tmp_val    

    prev_val_src = anno_data[row_index1][col_index]
    prev_val_tar = anno_data[row_index2][col_index]
    # print("@@@@-aft", prev_val_src, prev_val_tar)

    return anno_data


def find_same_data_rows(anno_data:List[List[List[int]]]):
    positions = []
    for pos_, data_ in enumerate(anno_data):
        if(all(sorted(data_[0]) == sorted(each_) for each_ in data_[1:])):
            positions.append(pos_)
    return positions

def find_adjustable_rows(anno_data:List[List[List[int]]]):
    positions = []
    for pos_, data_ in enumerate(anno_data):
        if(any(sorted(data_[0]) != sorted(each_) for each_ in data_[1:])):
            positions.append(pos_)
    return positions


class ProcFail(Exception):
    def __init__(self, message, errors):
        super().__init__(message)
        self.errors = errors
    def print_errors(self):
        print(self.errors)


class SameDataAdjuster(object):
    '''adjust annotated data to meet the set number of the same data'''
    def __init__(self, logger:MlaLogger, anno_data:List[List[List[int]]], same_data_num):
        self.logger = logger
        self.anno_data = copy.deepcopy(anno_data)
        self.same_data_num = same_data_num
        self.max_try_num = 1000
        self.item_num = len(anno_data)
        self.coder_num = len(anno_data[0])
        self.logger.save_to_tmp(anno_data)

        # probs before same
        self.coder_prob_init = self.get_coder_probs(anno_data)

    def get_coder_probs(self, anno_data):
        data_coders = change_anno_data_2_data_coders(anno_data)
        stat_coder_probs = []
        for coder in data_coders:
            stat_coder_probs.append(stat_prob(coder))
        return stat_coder_probs

    def _adjust_one_data(self, anno_data:List[List[List[int]]], row_source:int, 
            candidate_positions:List[int]):        

        val = anno_data[row_source][0]
        data_coders = change_anno_data_2_data_coders(anno_data)
        for coder_index_ in range(1, self.coder_num):
            cur_pos = 0
        
            while(True):
                try:
                    coder_data = data_coders[coder_index_]
                    row_target = coder_data.index(val, cur_pos)
                    if (row_source == row_target):
                        break

                    if row_target not in candidate_positions:
                        cur_pos = row_target + 1
                        continue
                    assert val == coder_data[row_target]
                    swap_anno_data_in_place(anno_data, row_source, row_target, coder_index_)
                    break
                except (IndexError, ValueError) as e:   # fail to find  val in array
                    raise ProcFail(
                        f'Fail to adjust coder:{coder_index_+1} row:{row_source} for val:{val}', e)
        
        return anno_data

    def check_4_porb_change(self, data, desc):
        coder_prob_ = self.get_coder_probs(data) 
        for index_, (prob_init_, prob_samed_) in enumerate(zip(self.coder_prob_init, coder_prob_)):
            # self.logger.add_log(F" {desc}:{prob_init_==prob_samed_}  before:{sorted(prob_init_.items())}, samed:{sorted(prob_samed_.items())}")   
            # self.logger.add_log(F" {desc}:{prob_init_==prob_samed_}")   
            if (prob_init_ != prob_samed_):
                return False
        return True

    def _try_to_adjust(self, data, row_index:int, candidate_positions:List[int]):
        cloned_data = copy.deepcopy(data)
        candidate_positions_cloned = copy.deepcopy(candidate_positions)
        assert row_index in candidate_positions_cloned

        data_changed = self._adjust_one_data(cloned_data, row_index, candidate_positions_cloned)        
        candidate_positions.remove(row_index)
        return data_changed

    def adjust(self):
        logger = self.logger
        
        current_same_positions = find_same_data_rows(self.anno_data)
        date_samed_number_begin = len(current_same_positions)
        assert date_samed_number_begin <= self.same_data_num, F"Fail to set {self.same_data_num}: existing {date_samed_number_begin} samed"

        changed_data = self.anno_data
        candidate_positions = find_adjustable_rows(changed_data)
        # logger.add_log(F"cur_same:{len(current_same_positions)}, can_pos:{len(candidate_positions)}")

        with tqdm(total=self.same_data_num) as pbar:
            try_count = 1
            samed_data_num = self.item_num - len(candidate_positions)
            pbar.update(samed_data_num)
            samed_data_num_prev = samed_data_num
            while (samed_data_num < self.same_data_num) and (try_count < self.max_try_num):
                change_row = random.choice(candidate_positions)
                try:
                    changed_data = self._try_to_adjust(changed_data, change_row, candidate_positions)
                    candidate_positions = find_adjustable_rows(changed_data)
                    samed_data_num = self.item_num - len(candidate_positions)
                    pbar.update(samed_data_num-samed_data_num_prev)
                    samed_data_num_prev = samed_data_num
                except ProcFail as e:
                    try_count += 1
        date_samed_number_end = len(find_same_data_rows(changed_data))
        if date_samed_number_end >= self.same_data_num:
            logger.add_log(F"Samed OK. from {date_samed_number_begin} to {date_samed_number_end}")
            coder_prob_samed = self.get_coder_probs(changed_data)
            # for index_, (prob_init_, prob_samed_) in enumerate(zip(self.coder_prob_init, coder_prob_samed)):
            #     logger.add_log(F"coder {index_:02} prob equal:{prob_init_==prob_samed_}  init:{prob_init_}, samed:{prob_samed_}")
            return changed_data

        logger.add_log(F"Samed Fail to {self.same_data_num}. OK from {date_samed_number_begin} to {date_samed_number_end}")
        logger.save_to_tmp(changed_data)        
        raise ProcFail(F"Cann't set {self.same_data_num} samed data. Save to tmp log", None)

def cal_kappa(po, pe):
    return (po-pe)/(1-pe)

def build_reliability_data(reliability_data_str:set())->List[int]:
    return [[np.nan if v == "*" else int(v) for v in coder.split()] for coder in reliability_data_str]

def build_value_counts(anno_data:List[List[List[int]]], k:int)->List[int]:
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

        self.log_parent_dir = log_parent_dir
        log_path = os.path.join(log_parent_dir, f"{desc}_{inst_id}")
        os.makedirs(log_path, exist_ok=True)  # for temp debug log may reuse dir
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

    def save_to_tmp(self, temp_data:List[List[List[int]]]):
        '''for debug use'''
        os.makedirs(f"{self.log_parent_dir}/tmp/", exist_ok=True)
        files = [
            f"{self.log_parent_dir}/tmp/tmp.log",
            f"{self.log_parent_dir}/tmp/tmp.xlsx",
        ]

        with open(files[0], "w") as log_writer:
            log_writer.write("\n".join(self.logs))

        ac = AgreeCalculator(self, temp_data, k=-999, mla_only=True)
        anno_mla_po, anno_mla_pos = ac.mla_po()
        df_tmp = _build_data_df(temp_data, anno_mla_pos)
        with pd.ExcelWriter(files[1]) as writer:
            df_tmp.to_excel(writer, sheet_name="df_tmp", index=False)

def convert_anno_data_to_fleiss(anno_data:List[List[List[int]]], k):
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

def change_to_joint_proportion(probs, coder_num:int):
    assert len(probs) == coder_num
    jp = np.mean(np.array(probs), axis = 0)
    jp_probs = [jp] * coder_num
    assert np.array(probs).shape == np.array(jp_probs).shape
    return jp_probs

def parse_2d_array(string):
    try:
        array = json.loads(string)
        assert isinstance(array, list)
        assert all(isinstance(row, list) for row in array)
        return array
    except (ValueError, AssertionError):
        raise argparse.ArgumentTypeError('Invalid 2D array')

def cal_cohen_kappa(logger:MlaLogger, anno_data:List[List[int]], k):
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

    yy = matx[0][0]
    nn = matx[1][1]
    yn = matx[0][1]
    ny = matx[1][0]
    
    item_num = len(anno_data)
    po = (yy+nn)/item_num
    pe = (yy+ny)/item_num * (yy+yn)/item_num + (yn+nn)/item_num*(ny+nn)/item_num
    kappa = cal_kappa(po,pe)
    
    logger.add_log(f"ct={ct}")
    logger.add_log(f"contingency table\n {matx}")
    logger.add_log(f"po:    {po}")
    logger.add_log(f"pe:    {pe}")
    logger.add_log(f"kappa  {kappa}")
    logger.add_log(f"cohen_kappa_score  {cohen_kappa_score(c1, c2)}  {cohen_kappa_score(c2, c1)}")

    assert abs(cohen_kappa_score(c1, c2) - kappa) < 1e-5
    return kappa, po, pe, {"y_y":yy, "n_n":nn,"y_n":yn,"n_y":ny}

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