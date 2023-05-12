# multi-label_agreement
This repo contains implementation for paper "Annotation Quality Measurement in Multi-Label Annotations".  
It enables users to specify the distribution of data and generate annotations that mimic different annotators. It also computes the agreement coefficient for the simulated annotations. Various annotation scenarios are supported, such as multi-rater multi-class, multi-rater multi-label etc.

## How to use the code
Execute the python command under the src directory to simulate different scenarios. The operation result will be placed in the log directory at the same level as src.

### 2-coder 2 class 
python proc_2coder_2class.py -yy 10 -nn 20 -yn 5 -ny 25 -ts 6000 -r 10

### 2-coder multi-class 
python proc_2coder_multi_class.py -k 5 -ps '[[0.1,0.2,0.3,0.15,0.25],[0.2,0.3,0.15,0.25,0.1]]' -s 300 -ta 600 -ts 6000 -r 5

### multi-coder multi-class 
python proc_multi_coder_multi_class.py -k 5 -ps '[[0.1,0.2,0.3,0.15,0.25],[0.2,0.3,0.15,0.25,0.1],[0.13,0.37,0.15,0.25,0.1],[0.20,0.2,0.2,0.2,0.2]]' -s 200 -ta 600 -ts 6000 -r 5

### multi-coder multi-class using joint proportion
python proc_multi_coder_multi_class_jointprob.py -k 5 -ps '[[0.1,0.2,0.3,0.15,0.25],[0.2,0.3,0.15,0.25,0.1],[0.13,0.37,0.15,0.25,0.1],[0.20,0.2,0.2,0.2,0.2]]' -s 200 -ta 600 -ts 6000 -r 5

### po in 2-coder multi-label
python proc_po_in_2coder_multi_label.py

### multi-coder multi-label
 python proc_multi_coder_multi_label.py -n 3 -k 4 -ps '[[[0.1,0.2,0.3,0.4], 0.2, 0.04],[[0.3,0.1,0.45,0.15], 0.3, 0.05],[[0.3,0.1,0.45,0.15], 0.3, 0.01]]' -s 200 -ta 800 -ts 3200 -r 5

### unit test
python -m unittest  
python -m test.test_util_case