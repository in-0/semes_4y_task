export MASTER_PORT=29502
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
# start 250703 num 2
# Fusion + use_textemb + MTMLoss
CUDA_VISIBLE_DEVICES=2 python train.py --num_epochs 100 --modality fusion --use_textemb --use_mtm --batch_size 32 --comment sms_fusion_textemb_mtm_lambda0.5 --data sms --mtm_lambda 0.5
# for i in {0.5..0.1..0.9}; do
#     CUDA_VISIBLE_DEVICES=2 python train.py --num_epochs 100 --modality fusion --use_textemb --use_mtm --batch_size 32 --comment sms_fusion_textemb_mtm_lambda${i} --data sms --mtm_lambda $i
# done