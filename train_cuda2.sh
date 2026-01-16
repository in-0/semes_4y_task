export CUDA_VISIBLE_DEVICES=2
export MASTER_ADDR=localhost
export MASTER_PORT=29502
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --data sms --comment sms_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --data sms --comment sms_vision

# Fusion 모델 (기본)
python train.py --num_epochs 30 --batch_size 32 --data sms --use_cb --use_paco --save_path ./results_4y --comment sms_fusion_reproduce

# GDCM
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --comment gdcm_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --comment gdcm_vision

# Fusion 모델 (기본)
# python train.py --num_epochs 30 --modality fusion --batch_size 32 --comment gdcm_fusion