CUDA_VISIBLE_DEVICES=0  \
python train.py  \
--drop_out  \
--early_stopping    \
--lr 5e-5   \
--k 10  \
--label_frac 1    \
--weighted_sample   \
--bag_loss ce   \
--inst_loss svm \
--task BRCA   \
--log_data  \
--data_root_dir ../Dataset/TCGA-BRCA/patch_4096/convnexts_512_4096    \
--exp_code brca_5124096   \
--model_type mamba    \
--test_name pam_is_2000  \
