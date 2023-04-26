
from util.util_common import MlaLogger
from util import util_case, util_common

import argparse

def proc_2coder_2class(args):
    yy = args.yy_num
    nn = args.nn_num
    yn = args.yn_num
    ny = args.ny_num
    total_simu_data = args.total_simu_data
    repeat_times = args.repeat
    k = 2

    desc = f"2coder_2class_({yy},{nn},{yn},{ny})_{total_simu_data}_{repeat_times}"
    inst_id = MlaLogger.get_inst_id()
    logger = MlaLogger(inst_id, desc=desc)

    logger.add_section_input()
    logger.add_log(F"2coder_2class input: {str(args)}")
    for var_ in ["yy", "nn","yn","ny","total_simu_data", "repeat_times", "k"]:
        logger.add_log(util_common.formated_str_4_var(var_, eval(var_)))
   
    anno_data = util_case.gen_anno_data_4_cohen(
        logger, num_yy=yy, num_nn=nn, num_yn=yn, num_ny=ny)
    util_common.check_anno_simu_num(len(anno_data), total_simu_data)
    
    util_case.proc_2coder_2class(
        logger, desc, k=k, anno_data=anno_data, 
        total_simu_data=total_simu_data, repeat=repeat_times)

    logger.save()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-yy', "--yy_num", help="Yes-Yes number (default:%(default)s)",
        required=False, type=int, default=20)
    parser.add_argument('-nn', "--nn_num", help="No -No  number (default:%(default)s)",
        required=False, type=int, default=5)        
    parser.add_argument('-yn', "--yn_num", help="Yes-No  number (default:%(default)s)",
        required=False, type=int, default=10)
    parser.add_argument('-ny', "--ny_num", help="No -Yes number (default:%(default)s)",
        required=False, type=int, default=15)
    parser.add_argument('-ts', "--total_simu_data", help="total simulated data (default:%(default)s)",
        required=False, type=int, default=1000)
    parser.add_argument('-r', "--repeat", help="repeat times of simulation (default:%(default)s)",
        required=False, type=int, default=3)
    
    args = parser.parse_args()
    proc_2coder_2class(args)
    # proc_2coder_2class(args.tp_num, args.tn_num, args.fp_num, args.fn_num, 
    #     args.total_simu_data, args.repeat)

if __name__ == "__main__":
    main()
    