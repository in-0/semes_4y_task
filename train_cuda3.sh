export CUDA_VISIBLE_DEVICES=3
export MASTER_ADDR=localhost
export MASTER_PORT=29503
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --data sms --comment sms_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --data sms --comment sms_vision

# Fusion 모델 (기본) - 원본 설정 (k=4096)
# python train.py --num_epochs 100 --batch_size 32 --data sms --lr 0.01 --moco_k 4096 --moco_m 0.999 --moco_t 0.07 --use_cb --use_paco --use_dim_matching_layer --save_path ./results_4y --comment sms_fusion_matching_dim_layer_moco_k_4096_m_0.999_t_0.07_lr_0.01_epoch_100

# Fusion 모델 (k=2048에 최적화된 설정)
python train.py --num_epochs 100 --batch_size 64 --data sms --lr 0.001 --moco_k 1024 --moco_m 0.99 --moco_t 0.07 --use_cb --use_paco --save_path ./results_4y --comment sms_fusion_moco_k_1024_m_0.99_t_0.07_lr_0.001_epoch_100_batch_64_aug_strong

# GDCM
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --comment gdcm_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --comment gdcm_vision

# Fusion 모델 (기본)
# python train.py --num_epochs 30 --modality fusion --batch_size 32 --comment gdcm_fusion