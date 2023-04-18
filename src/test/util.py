from typing import List
import numpy as np

def _count_label(label:int, labels_for_one_record:List[List[int]])->float:
	return sum(sublist.count(label) for sublist in labels_for_one_record) - 1

def mla_score(labels_for_one_record:List[List[int]])->float:
	rater_num = len(labels_for_one_record)
	score = 0
	for label_for_one_rater in labels_for_one_record:
		label_num = len(label_for_one_rater)
		score += sum([_count_label(l, labels_for_one_record) for l in label_for_one_rater]) / label_num
	score = score / (rater_num * (rater_num - 1))
	return score

def avg_mla_score(labels_for_all_records:List[List[List[int]]])->float:
	return np.mean([mla_score(labels_for_one_record) for labels_for_one_record in labels_for_all_records])
          
if __name__ == '__main__':
	anno_data = [[[9, 7, 29], [9, 7, 19], [9, 7]], [[9, 7, 29], [9, 7, 19], [9, 7]]]
	mla_score = avg_mla_score(anno_data)
	assert 0.78 == round(mla_score, 2)