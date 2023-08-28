source ../../../env_vq.sh
###### Training scripts
#python train_single.py --data_root ['your dataset path'] -m ['Subject_name'] --load_path ['Pretrain Model path'] --gpu ["gpu_id"]
# python train_single.py --data_root ../../../Backup/Dataset/Dataset/Obama/ -m Obama --gpu 0 --load_path ../Pretrained/taming/Obama/N-Step-Checkpoint_21_270000.ckpt

###### Testing scripts
#python test_grid.py --data_root ['your dataset path'] -m ['Subject_name'] --load_path ['Pretrain Model path'] --gpu ["gpu_id"] --enc_path [Encoder path] --cls_path [Classifier path]
python test_grid.py --data_root Dataset/Obama/ -m Obama_self --gpu 0
