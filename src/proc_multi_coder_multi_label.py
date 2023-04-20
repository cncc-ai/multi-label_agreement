'''
1) 2-coders 2-classes (cohen)
2) 2-coders mul-classes (ka, fleiss)
3) mul-coders mul-classes (k)
4) mul-coders mul-labels
'''

from util import util_common
from util.util_common import MlaLogger, MultiLabelDataBuilder
from util import util_case
from argparse import ArgumentParser


def proc(args:ArgumentParser):

    n_coder = args.coder_num
    k = args.class_num
    probs = args.probs
    total_anno_data = args.total_anno_data
    total_simu_data = args.total_simu_data
    repeat_times = args.repeat

    desc = f"multi_coder_multi_label_{n_coder}_{k}_{total_anno_data}_{total_simu_data}_{repeat_times}"

    inst_id = MlaLogger.get_inst_id()
    logger = MlaLogger(inst_id, desc=desc)
    logger.add_section_input()
    logger.add_log(F"multi_coder_multi_class input: {str(args)}")
    for var_ in ["n_coder", "k", "probs", "total_anno_data", "total_simu_data", "repeat_times"]:
        logger.add_log(util_common.formated_str_4_var(var_, eval(var_)))
    
    assert n_coder == len(probs)
    builder = MultiLabelDataBuilder(total_anno_data, k, probs)
    anno_data = builder.build_data()

    util_case.proc_multi_coder_multi_label(logger, desc, anno_data, k, total_simu_data, repeat_times)
    
    logger.save()
    
def main():
    parser = ArgumentParser()
    parser.add_argument('-n', "--coder_num", help="coder number",
        required=False, type=int, default=3) 
    parser.add_argument('-k', "--class_num", help="class number",
        required=False, type=int, default=4) 
    parser.add_argument('-ps', "--probs", help="class probabilities of each coder",
        required=False, type=util_common.parse_2d_array, \
        default='[[[0.1,0.2,0.3,0.4], 0.2, 0.04],[[0.3,0.1,0.45,0.15], 0.3, 0.05],[[0.3,0.1,0.45,0.15], 0.3, 0.01]]')
    parser.add_argument('-ta', "--total_anno_data", help="total annotated data",
        required=False, type=int, default=800)   
    parser.add_argument('-ts', "--total_simu_data", help="total simulated data",
        required=False, type=int, default=3200)        
    parser.add_argument('-r', "--repeat", help="repeat times of simulation",
        required=False, type=int, default=5)    

    args = parser.parse_args()
    proc(args)

if __name__ == "__main__":
    main()
