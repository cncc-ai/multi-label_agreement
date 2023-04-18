'''
1) 2-coders 2-classes (cohen)
2) 2-coders mul-classes (ka, fleiss)
3) mul-coders mul-classes (k)
4) mul-coders mul-labels
'''

from util import util_common
from util.util_common import MlaLogger
from util import util_case

import argparse


def proc_multi_coder_multi_class(args):
    
    k = args.category_count
    probs = args.probs
    total_anno_data = args.total_anno_data
    total_simu_data = args.total_simu_data
    repeat_times = args.repeat

    n_coder = len(probs)
    assert n_coder > 2

    desc = f"multi_coder_multi_class_{n_coder}_{k}_{total_anno_data}_{total_simu_data}_{repeat_times}"
    inst_id = MlaLogger.get_inst_id()
    logger = MlaLogger(inst_id, desc=desc)
    logger.add_section_input()
    desc = f"multi_coder_multi_class_{k}_{total_anno_data}_{total_simu_data}_{repeat_times}"
    logger.add_log(F"multi_coder_multi_class input: {str(args)}")
    total_anno_data = args.total_anno_data
    for var_ in ["n_coder", "k", "probs", "total_anno_data", "total_simu_data", "repeat_times"]:
        logger.add_log(util_common.formated_str_4_var(var_, eval(var_)))

    anno_data = util_case.gen_anno_data_4_mul_class(total_anno_data, k = k, probs=probs)
    assert len(anno_data) == total_anno_data
    util_case.proc_multi_coder_multi_class(logger, desc, anno_data, k, total_simu_data, repeat_times)

    logger.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', "--category_count", help="category count of classes",
        required=False, type=int, default=4) 
    parser.add_argument('-ps', "--probs", help="category probabilities of each coder",
        required=False, type=util_common.parse_2d_array, default='[[0.1, 0.5, 0.3, 0.10], [0.3, 0.5, 0.10, 0.10], [0.2, 0.1, 0.15, 0.55]]')         
    parser.add_argument('-ta', "--total_anno_data", help="total annotation data",
        required=False, type=int, default=800)   
    parser.add_argument('-ts', "--total_simu_data", help="total simulated data",
        required=False, type=int, default=3200)        
    parser.add_argument('-r', "--repeat", help="repeat times of simulation",
        required=False, type=int, default=5)
    
    args = parser.parse_args()
    proc_multi_coder_multi_class(args)

if __name__ == "__main__":
    main()
