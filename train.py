from __future__ import print_function

import argparse
import pdb
import os
import math
import shutil

# internal imports
from utils.file_utils import save_pkl, load_pkl
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset

# pytorch imports
import torch
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--data_root_dir', type=str, default=None, 
                    help='data directory')
parser.add_argument('--max_epochs', type=int, default=200,
                    help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-5,
                    help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, 
                    help='manually specify the set of splits to use, ' 
                    +'instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--testing', action='store_true', default=False, help='debugging tool')
parser.add_argument('--early_stopping', action='store_true', default=False, help='enable early stopping')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--drop_out', action='store_true', default=False, help='enabel dropout (p=0.25)')
parser.add_argument('--bag_loss', type=str, choices=['svm', 'ce'], default='ce',
                     help='slide-level classification loss function (default: ce)')
parser.add_argument('--model_type', type=str, choices=['hipt_lgp', 'hipt_n', 'hit', 'clam_sb', 'clam_mb', 'mil', 'dsmil', 'pam', 'transmil', 'dtfdmil', 's4', 'abmil'], default='pam', 
                    help='type of model (default: clam_sb, clam w/ single attention branch)')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--weighted_sample', action='store_true', default=False, help='enable weighted sampling')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model, does not affect mil')
parser.add_argument('--task', type=str, choices=['RCC', 'BRCA', 'NSCLC', 'ESCA'])
### CLAM specific options
parser.add_argument('--no_inst_cluster', action='store_true', default=False,
                     help='disable instance-level clustering')
parser.add_argument('--inst_loss', type=str, choices=['svm', 'ce', None], default=None,
                     help='instance-level clustering loss function (default: None)')
parser.add_argument('--subtyping', action='store_true', default=False, 
                     help='subtyping problem')
parser.add_argument('--bag_weight', type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--B', type=int, default=8, help='numbr of positive/negative patches to sample for clam')
parser.add_argument('--test_name', type=str, help='Test name')

## A-ViT
parser.add_argument('--ponder_token_scale', default=0.0005, type=float, help="")
parser.add_argument('--pretrained', action='store_true',
                    help='raise to load pretrained.')
parser.add_argument('--act_mode', default=4, type=int,
                    help='4-token act, make sure this is always 4, other modes are only used for initial method comparison and exploration')
parser.add_argument('--tensorboard', action='store_true',
                    help='raise to load pretrained.')
parser.add_argument('--gate_scale', default=100., type=float, help="constant for token control gate rescale")
parser.add_argument('--gate_center', default= 3., type=float, help="constant for token control gate re-center, negatived when applied")
parser.add_argument('--warmup_epoch', default=0, type=int, help="warm up epochs for act")
parser.add_argument('--distr_prior_alpha', default=0.001, type=float, help="scaling for kl of distributional prior")

parser.add_argument('--survival_pred', action='store_true', default=False, 
                     help='survival prediction problem')

args = parser.parse_args()



def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    all_test_auc = []
    all_val_auc = []
    all_test_acc = []
    all_val_acc = []
    folds = np.arange(start, end)
    for i in folds:
        seed_torch(args.seed)
        # import ipdb; ipdb.set_trace()
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        # import ipdb;ipdb.set_trace()
        datasets = (train_dataset, val_dataset, test_dataset)
        # import ipdb;ipdb.set_trace()
        results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
        all_test_auc.append(test_auc)
        all_val_auc.append(val_auc)
        all_test_acc.append(test_acc)
        all_val_acc.append(val_acc)
        #write results to pkl
        filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        save_pkl(filename, results)

        fold_col = folds[:i+1].tolist()

        if i == folds[-1]:
            fold_col.append('mean')
            all_test_auc.append(np.mean(all_test_auc))
            all_val_auc.append(np.mean(all_val_auc))
            all_test_acc.append(np.mean(all_test_acc))
            all_val_acc.append(np.mean(all_val_acc))

            fold_col.append('std')
            all_test_auc.append(np.std(all_test_auc))
            all_val_auc.append(np.std(all_val_auc))
            all_test_acc.append(np.std(all_test_acc))
            all_val_acc.append(np.std(all_val_acc))

        final_df = pd.DataFrame({'folds': fold_col, 'test_auc': all_test_auc, 
            'val_auc': all_val_auc, 'test_acc': all_test_acc, 'val_acc' : all_val_acc})

        if len(folds) != args.k:
            save_name = 'summary_partial_{}_{}.csv'.format(start, end)
        else:
            save_name = 'summary.csv'
        final_df.to_csv(os.path.join(args.results_dir, save_name))



device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 1024
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size': args.model_size,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'opt': args.opt}

if args.model_type in ['clam_sb', 'clam_mb']:
   settings.update({'bag_weight': args.bag_weight,
                    'inst_loss': args.inst_loss,
                    'B': args.B})

print('\nLoad Dataset')

if args.task == 'NSCLC':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-NSCLC/tcga-nsclc_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'LUAD':0, 'LUSC':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'BRCA':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-BRCA/tcga-brca_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'IDC':0, 'ILC':1},
                            patient_strat=False,
                            ignore=['MDLC', 'PD', 'ACBC', 'IMMC', 'BRCNOS', 'BRCA', 'SPC', 'MBC', 'MPT'])

elif args.task == 'RCC':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-RCC/tcga-kidney_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'CCRCC':0, 'PRCC':1, 'CHRCC':2},
                            patient_strat=False,
                            ignore=[]),
    dataset = dataset[0]

elif args.task == 'ESCA':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = '../Dataset/TCGA-ESCA/tcga-esca_label.csv',
                            data_dir= args.data_root_dir,
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Adenocarcinoma':0, 'Squamous cell carcinoma':1},
                            patient_strat=False,
                            ignore=['Basaloid squamous cell carcinoma', 'Tubular adenocarcinoma']),
    dataset = dataset[0]
else:
    raise NotImplementedError
    
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# args.results_dir = os.path.join(args.results_dir, str(args.exp_code) + '_s{}'.format(args.seed) + '_' + args.test_name)
args.results_dir = os.path.join(args.results_dir, str(args.exp_code), 's{}'.format(args.seed) + '_' + args.test_name)
# if not os.path.isdir(args.results_dir):
if os.path.exists(args.results_dir):
    shutil.rmtree(args.results_dir)
os.makedirs(args.results_dir)

if args.split_dir is None:
    args.split_dir = os.path.join('10fold_splits', args.task + '_{}'.format(int(args.label_frac*100)))
else:
    args.split_dir = os.path.join('10fold_splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)
    print("finished!")
    print("end script")


