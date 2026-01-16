export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=localhost
export MASTER_PORT=29500
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --data sms --comment sms_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --data sms --comment sms_vision

# Fusion 모델 (기본)
python train.py --num_epochs 30 --batch_size 32 --data sms --use_cb --use_paco --use_dim_matching_layer --save_path ./results_4y --comment sms_fusion_matching_dim_layer

# GDCM
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --comment gdcm_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --comment gdcm_vision

# Fusion 모델 (기본)
# python train.py --num_epochs 30 --modality fusion --batch_size 32 --comment gdcm_fusion