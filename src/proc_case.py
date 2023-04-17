'''
1) 2-coders 2-classes (cohen)
2) 2-coders mul-classes (ka, fleiss)
3) mul-coders mul-classes (k)
4) mul-coders mul-labels
'''

from util.util_common import MlaLogger
# from util import util_case
from util import util_case
# from util.util_case import pro\\


def doproc_2coder_mul_class(logger:MlaLogger):

    anno_data = [
        [[1], [1]],
        [[2], [2]],
        [[1], [3]],
        [[3], [2]],   
    ]
    desc = "2coder_mul_class"
    k = 3
    
    simu_data_multiplier = 200
    
    # anno_data = util_case.gen_anno_data_4_cohen(
    #     tot_num=100, p_00=0.3, p_11=0.1, p_01=0.3, logger=logger)

    # util_case.proc_cohen(anno_data, 
    #     simu_data_len=simu_data_multiplier*len(anno_data), 
    #     k=2, desc=desc, logger=logger)
    simu_data_multiplier = 10

    k = 4
    anno_data = util_case.gen_anno_data_4_mul_class(logger, 10, k = k, 
        probs= [[0.1, 0.5, 0.3, 0.10], [0.3, 0.5, 0.10, 0.10]])

    util_case.proc_2coder_mul_class(logger, anno_data, k, 100)
              # proc_2coder_mul_class(logger, anno_data:List[List[int]], k:int, simu_data_len:int):

if __name__ == "__main__":
    inst_id = MlaLogger.get_inst_id()
    logger = MlaLogger(inst_id)
    
    doproc_2coder_mul_class(logger)

    logger.save()