from typing import List
import glob
import os
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

def f1_score(coder1:List[int], coder2:List[int])->float:
	coder1, coder2 = set(coder1), set(coder2)
	intersection_len = len(coder1 & coder2)
	p = intersection_len / len(coder1)
	r = intersection_len / len(coder2)
	if (p + r) == 0:
		return 0
	else:
		return 2 * p * r / (p + r)
	
def jaccard(list1, list2):
  intersection = len(list(set(list1).intersection(list2)))
  union = (len(list1) + len(list2)) - intersection
  return float(intersection) / union
          
def get_latest_file(file_prefix:str)->str:
	cur_file_path = os.path.dirname(__file__)
	log_path = os.path.abspath(os.path.join(cur_file_path, "../../logs/"))

	folders = glob.glob(f'{log_path}/{file_prefix}*')
	assert len(folders)>0, F"No result can be found in {f'{log_path}/{file_prefix}*'}. Please run 'python proc_{file_prefix}.py' first."
	res_folder = max(folders, key=os.path.getctime)

	files = glob.glob(f'{res_folder}/m*.xlsx')
	assert len(files)>0, F"No result can be found in {f'{res_folder}'}. Please run 'python proc_{file_prefix}.py' first."
	latest_file = max(files, key=os.path.getctime)
	return latest_file

if __name__ == '__main__':
	anno_data = [[[9, 7, 29], [9, 7, 19], [9, 7]], [[9, 7, 29], [9, 7, 19], [9, 7]]]
	mla_score = avg_mla_score(anno_data)
	assert 0.78 == round(mla_score, 2)