# multi-label_agreement

## 2 coder 2 class
python proc_2coder_2class.py -tp 10 -tn 20 -fp 5 -fn 25 -ts 6000 -r 3

## 2 coder nulti-class
python proc_2coder_mul_class.py -k 5 -ps [[0.1,0.2,0.3,0.15,0.25],[0.2,0.3,0.15,0.25,0.1]] -ta 600 -ts 6000 -r 3

## unit test
 python -m unittest
