from statsmodels.stats.inter_rater import to_table
from tqdm import tqdm
from typing import List

from util.util_common import MultiLabelDataBuilder, count_data_by_len, count_1data_by_val, count_data_by_val
from util.util_common import MlaLogger

import pandas as pd
import unittest



class TestMultiLabelDataBuilder(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        tot_data = 1000
        k = 4 
        coder_prob_1 = [(.1,.2,.3,.4), 0.2, 0.05]
        coder_prob_2 = [(.3,.1,.45,.15), 0.3, 0.05]
        coder_prob_3 = [(.3,.1,.45,.15), 0.3, 0.05]

        inst_id = "debug_id"
        self.logger = MlaLogger(inst_id, desc="debug")
        
        
        builder = MultiLabelDataBuilder(self.logger, tot_data=tot_data, k=k, probs=[coder_prob_1, coder_prob_2, coder_prob_3])
        self.builder = builder
        self.tot_data = tot_data
        self.k = k
        self.coder_prob_1 = coder_prob_1
        self.coder_prob_2 = coder_prob_2
        self.coder_prob_3 = coder_prob_3
   

    @classmethod
    def tearDownClass(self) -> None:
        self.logger.save()
        print(f'tearDownClass completed')

    def test_build_4_coder(self):
        # self._proc_code_prob(self.coder_prob_1)
        for index_, coder_prob in enumerate([self.coder_prob_1, self.coder_prob_2, self.coder_prob_3]):
            self._proc_code_prob(coder_prob, index_)

    def _proc_code_prob(self, coder_prob, index_):
        gen_coder_data = self.builder._build_4_coder(coder_prob)
        df_ = pd.DataFrame(gen_coder_data)
        df_['str'] = df_.iloc[:].astype(str).agg('|'.join, axis=1)
        self.logger.add_df(F"coder_{index_:02}", df_)
        
        prob_1class = coder_prob[0]
        prob_double = coder_prob[1]
        prob_tripple = coder_prob[2]
        assert prob_tripple == count_data_by_len(gen_coder_data, 3) / self.tot_data
        assert prob_double == count_data_by_len(gen_coder_data, 2) / self.tot_data      

        class_n_nums = [count_data_by_val(gen_coder_data, n) for n in range(1, self.k+1)]

        for index_, (prob_1class_n, class_n_num) in \
                enumerate(zip(prob_1class, class_n_nums)):

            stand_set = [one/min(prob_1class) for one in prob_1class]
            stand_gen = [one/min(class_n_nums) for one in class_n_nums]
            diff_stand = [abs(set_-gen_)/set_ for set_, gen_ in zip(stand_set, stand_gen)]
            assert max(diff_stand) < 0.15, F"diff_stand: {diff_stand}"

    def _get_max_diff(self, prob):
        ret_diff = 0.5
        if (prob < 0.3):
            return 0.1
        if (prob<0.5):
            return 0.16
        return 0.18

    def test_build_data(self):
        coder_prob = self.coder_prob_1
        data_conders = self.builder.build_data()

        pass
  
        
       

