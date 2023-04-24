
import unittest

from util.util_common import SameDataAdjuster, change_anno_data_2_data_coders, find_same_data_rows, stat_prob, MlaLogger


class TestSameDataAdjuster(unittest.TestCase):

    def test_adjust(self):

        inst_id = "debug_id"
        logger = MlaLogger(inst_id, desc="debug")

        anno_data = [
            [[1],[2],[1]],
            [[1],[1],[2]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2]],
        ]
        assert 1 == len(find_same_data_rows(anno_data))
        
        sameDataAdjuster = SameDataAdjuster(logger, anno_data, same_data_num=2)
        data_samed = sameDataAdjuster.adjust()
        assert 2 == len(find_same_data_rows(data_samed))
        assert 1 == len(find_same_data_rows(anno_data))
        
        data_coders = change_anno_data_2_data_coders(anno_data)
        data_coders_samed = change_anno_data_2_data_coders(data_samed)

        for data, data_samed in zip(data_coders, data_coders_samed):
            assert stat_prob(data) == stat_prob(data_samed)
        
        
    def test_adjust_coder2(self):

        inst_id = "debug_id"
        logger = MlaLogger(inst_id, desc="debug")

        anno_data = [
            [[2],[2],[1]],
            [[1],[1],[2]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2]],
        ]
        assert 1 == len(find_same_data_rows(anno_data))
        
        sameDataAdjuster = SameDataAdjuster(logger, anno_data, same_data_num=2)
        data_samed = sameDataAdjuster.adjust()
        assert 3 == len(find_same_data_rows(data_samed))
        assert 1 == len(find_same_data_rows(anno_data))
        
        data_coders = change_anno_data_2_data_coders(anno_data)
        data_coders_samed = change_anno_data_2_data_coders(data_samed)

        for data, data_samed in zip(data_coders, data_coders_samed):
            assert stat_prob(data) == stat_prob(data_samed)
        
        
    def test_adjust_coders(self):

        inst_id = "debug_id"
        logger = MlaLogger(inst_id, desc="debug")

        anno_data = [
            [[1],[2],[2]],
            [[3],[1],[1]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2]],
            [[1],[2],[3]],
        ]
        assert 1 == len(find_same_data_rows(anno_data))
        
        sameDataAdjuster = SameDataAdjuster(logger, anno_data, same_data_num=2)
        data_samed = sameDataAdjuster.adjust()
        assert 2 == len(find_same_data_rows(data_samed))
        assert 1 == len(find_same_data_rows(anno_data))
        
        data_coders = change_anno_data_2_data_coders(anno_data)
        data_coders_samed = change_anno_data_2_data_coders(data_samed)

        for data, data_samed in zip(data_coders, data_coders_samed):
            assert stat_prob(data) == stat_prob(data_samed)

    def test_adjust_coders_2(self):

        inst_id = "debug_id"
        logger = MlaLogger(inst_id, desc="debug")

        anno_data = [
            [[1],[2],[2]],
            [[3],[1],[1]],
            [[1,2,3,4],[1,3,2,4],[1,4,2,3]],
            [[1,2,3,4],[1,3,2,4],[1,4,2]],
            [[1],[2],[3]],
        ]
        assert 1 == len(find_same_data_rows(anno_data))
        
        sameDataAdjuster = SameDataAdjuster(logger, anno_data, same_data_num=2)
        data_samed = sameDataAdjuster.adjust()
        assert 2 == len(find_same_data_rows(data_samed))
        assert 1 == len(find_same_data_rows(anno_data))
        
        data_coders = change_anno_data_2_data_coders(anno_data)
        data_coders_samed = change_anno_data_2_data_coders(data_samed)

        for data, data_samed in zip(data_coders, data_coders_samed):
            assert stat_prob(data) == stat_prob(data_samed)
