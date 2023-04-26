
from util import util_common
from util.util_common import MlaLogger, SameDataAdjuster
from util import util_case

import argparse
import numpy as np


def proc_multi_coder_multi_class_jp(args):
    
    k = args.class_num
    probs = args.probs
    samed_num = args.samed_num
    total_anno_data = args.total_anno_data
    total_simu_data = args.total_simu_data
    repeat_times = args.repeat

    n_coder = len(probs)
    assert n_coder > 2
  
    jp_probs = util_common.change_to_joint_proportion(probs, n_coder)
    np.set_printoptions(precision=3)
    
    anno_data = util_case.gen_anno_data_4_mul_class(total_anno_data, k = k, probs=jp_probs)
    util_common.check_anno_simu_num(total_anno_data, total_simu_data)

    desc = f"multi_coder_multi_class_jp_{n_coder}_{k}_{samed_num}-{total_anno_data}_{total_simu_data}_{repeat_times}"
    inst_id = MlaLogger.get_inst_id()
    logger = MlaLogger(inst_id, desc=desc)
    logger.add_section_input()
    desc = f"multi_coder_multi_class_jp_{k}_{total_anno_data}_{total_simu_data}_{repeat_times}"
    logger.add_log(F"multi_coder_multi_class_jp input: {str(args)}")
    total_anno_data = args.total_anno_data
    for var_ in ["n_coder", "k", "probs", "jp_probs", "samed_num", "total_anno_data", "total_simu_data", "repeat_times"]:
        logger.add_log(util_common.formated_str_4_var(var_, eval(var_)))

    assert len(anno_data) == total_anno_data
    if samed_num > 0:
        samedProc = SameDataAdjuster(logger, anno_data, samed_num)
        anno_data = samedProc.adjust()
        assert samedProc.check_4_porb_change(anno_data, "check prob change")

    act_samed_data = util_common.find_same_data_rows(anno_data)
    assert len(act_samed_data) >= samed_num
    logger.add_log(F"samed data num: set:{samed_num} | actual:{len(act_samed_data)}")

    util_case.proc_multi_coder_multi_class(logger, desc, anno_data, k, total_simu_data, repeat_times)
    logger.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', "--class_num", help="class number (default:%(default)s)",
        required=False, type=int, default=4) 
    parser.add_argument('-ps', "--probs", help="class probabilities of each coder (default:\'%(default)s\')",
        required=False, type=util_common.parse_2d_array, default='[[0.1, 0.5, 0.3, 0.10], [0.3, 0.5, 0.10, 0.10], [0.2, 0.1, 0.15, 0.55]]')         
    parser.add_argument('-s', "--samed_num", help="samed number (default:%(default)s)",
        required=False, type=int, default=-1)
    parser.add_argument('-ta', "--total_anno_data", help="total annotated data (default:%(default)s)",
        required=False, type=int, default=800)   
    parser.add_argument('-ts', "--total_simu_data", help="total simulated data (default:%(default)s)",
        required=False, type=int, default=3200)        
    parser.add_argument('-r', "--repeat", help="repeat times of simulation (default:%(default)s)",
        required=False, type=int, default=5)
    
    args = parser.parse_args()
    proc_multi_coder_multi_class_jp(args)

if __name__ == "__main__":
    main()
