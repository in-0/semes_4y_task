export MASTER_PORT=29501
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --data sms --comment sms_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --data sms --comment sms_vision

# # Fusion 모델 (기본)
# python train.py --num_epochs 30 --batch_size 32 --data sms --comment sms_fusion

# GDCM
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --comment gdcm_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --comment gdcm_vision

# Fusion 모델 (기본)
# python train.py --num_epochs 30 --modality fusion --batch_size 32 --comment gdcm_fusion


# ------------------------------------------------------------
# start 250603 num 1
# Fusion + use_textemb
CUDA_VISIBLE_DEVICES=1 python train.py --num_epochs 30 --modality fusion --use_textemb --batch_size 32 --comment sms_fusion_textemb --data sms

# ------------------------------------------------------------
# start 250703 num 2
# Fusion + use_textemb + MTMLoss
CUDA_VISIBLE_DEVICES=1 python train.py --num_epochs 30 --modality fusion --use_textemb --use_mtm --batch_size 32 --comment sms_fusion_textemb_mtm --data sms