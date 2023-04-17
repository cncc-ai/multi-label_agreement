'''
1) 2-coders 2-classes (cohen)
2) 2-coders mul-classes (ka, fleiss)
3) mul-coders mul-classes (k)
4) mul-coders mul-labels
'''

from util.util_common import MlaLogger
from util import util_case

import argparse

def proc_2coder_2class(tp,tn,fp,fn,
        total_simu_data, repeat_times):

    # anno_data = [
    #     [[1], [1]],
    #     [[1], [0]],
    #     [[0], [0]],
    #     [[1], [1]],
    #     [[1], [0]],
    #     [[0], [1]],
    #     [[0], [1]],
    #     [[0], [1]],
    # ]
    desc = f"2coder_2class_({tp},{tn},{fp},{fn})_{total_simu_data}_{repeat_times}"
    
    inst_id = MlaLogger.get_inst_id()
    logger = MlaLogger(inst_id, desc=desc)
    
    # anno_data = util_case.gen_anno_data_4_cohen(
    #     logger, tot_num=100, p_00=0.3, p_11=0.1, p_01=0.3)
    anno_data = util_case.gen_anno_data_4_cohen(
        logger, num_tp=tp, num_tn=tn, num_fp=fp, num_fn=fn)

    util_case.proc_2coder_2class(
        logger, desc, k=2, anno_data=anno_data, 
        tot_simu_data=total_simu_data, repeat=repeat_times)

    logger.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp', "--tp_num", help="tp number",
        required=False, type=int, default=20)
    parser.add_argument('-tn', "--tn_num", help="tn number",
        required=False, type=int, default=5)        
    parser.add_argument('-fp', "--fp_num", help="fp number",
        required=False, type=int, default=10)
    parser.add_argument('-fn', "--fn_num", help="fn number",
        required=False, type=int, default=15)
    parser.add_argument('-ts', "--total_simu_data", help="total simulated data",
        required=False, type=int, default=1000)        

    parser.add_argument('-r', "--repeat", help="repeat times of simulation",
        required=False, type=int, default=3)
    
    args = parser.parse_args()
    proc_2coder_2class(args.tp_num, args.tn_num, args.fp_num, args.fn_num, 
        args.total_simu_data, args.repeat)

if __name__ == "__main__":
    main()
    