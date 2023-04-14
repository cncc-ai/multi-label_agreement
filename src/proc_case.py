'''
1) 2-coders 2-classes (cohen)
2) 2-coders mul-classes (ka, fleiss)
3) mul-coders mul-classes (k)
4) mul-coders mul-labels
'''
import util_case

def proc_2coder_2class():

    anno_data = [
        [[1], [1]],
        [[1], [0]],
        [[0], [0]],
        [[1], [1]],
        [[1], [0]],
        [[0], [1]],
        [[0], [1]],
        [[0], [1]],
    ]
    util_case.proc_cohen(anno_data, simu_data_len= 8, verbose=True)
    pass


if __name__ == "__main__":
    proc_2coder_2class()