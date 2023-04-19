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


def proc():    
    anno_data = [
        [[1,6], [2,3,4,5]],
        [[1,2,6], [2,3,4,5]],
        [[1,2,6], [2,3,4]],
        [[1,2], [2,3,4,5]],
        [[1,2,6], [2,3,]],
        [[1,2], [2,3,4]],
        [[1,2], [2,3]],
        [[1,2,3], [2,3,4,5]],
        [[1], [1,2]],
        [[1,2], [1,2,3]],
        [[1,2,3], [1,2,3,4]],
        [[1,2], [1,2]],
        [[1,2,3,4], [1,2,3,4]],    
    ]

    desc = f"po_in_2coder_multi_class"
    inst_id = MlaLogger.get_inst_id()
    logger = MlaLogger(inst_id, desc=desc)
    
    util_case.proc_po_in_2coder_multi_class(logger, desc, anno_data)

    logger.save()
    

if __name__ == "__main__":
    proc()
