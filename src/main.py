import sys
import os
sys.path.extend([os.path.dirname(os.getcwd()), os.getcwd()])
import pickle
import numpy as np
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--data_split_type', required = True, choices = ['proportional', 'cross_validate'])
parser.add_argument('--data_source', required=True, choices=['finance','it'])
args = parser.parse_args()     





if args.data_source == 'it':
    ## load the data and optimization for IT dataset
    from src.data_process import data
    from src.NDP_JSB import optimization
else:
    ## load the data and optimization for Finance dataset
    from src.data_process_finance import data
    from src.NDP_JSB_finance import optimization
data = data()

if args.data_split_type=='cross_validate':
    ## This is for 5-fold training
    for lu in [0,1]:
        if lu==0:
            print('===============this is for lower bound================')
        else:
            print('===============this is for upper bound================')
        data_gen = data.generate_fold(lower_or_upper=lu, construct_baseline_file=False)
        for fold in range(5):
            print('This is fold: {}'.format(fold))
            input = list(next(data_gen))
            opt = optimization()    
            opt.train_predict(input)
else:  
    ##This is for proportional training
    for lu in [0,1]:
        for p in [0.5,0.6,0.7,0.8,0.9]:
            if lu==0:
                print('===============this is for lower bound, proportional=%.2f================'%p)
            else:
                print('===============this is for upper bound, proportional=%.2f================'%p)
            data_gen=data.generate_proportional_data(train_proportion=p,lower_or_upper=lu,construct_baseline_file=False, seed=4)
            input=next(data_gen)
            opt=optimization()
            opt.train_predict(input)
