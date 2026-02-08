# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --data sms --comment sms_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --data sms --comment sms_vision

# Fusion 모델 (기본)
# python train.py --config configs/sms_fusion_balanced_matching_dim_layer_aug_strong_use_textemb_lr0.0003_t0.1_alpha0.3_lamb_paco0.5_ce2.0_epoch100_batch64_schedule_step_20.yaml

python train.py --config configs/sms_fusion_batch128_no_paco_no_aug_strong_lr0.001_schedule_step20.yaml
python train.py --config configs/sms_fusion_batch128_no_paco_no_aug_strong_lr0.0003_schedule_step20.yaml
python train.py --config configs/sms_fusion_matching_dim_layer_batch128_no_paco_no_aug_strong_lr0.001_schedule_step20.yaml
python train.py --config configs/sms_fusion_matching_dim_layer_batch128_no_paco_no_aug_strong_lr0.0003_schedule_step20.yaml
# GDCM
# # Sensor만 학습
# python train.py --modality sensor --num_epochs 10 --batch_size 32 --comment gdcm_sensor

# # Vision만 학습  
# python train.py --modality vision --num_epochs 10 --batch_size 32 --comment gdcm_vision

# Fusion 모델 (기본)
# python train.py --num_epochs 30 --modality fusion --batch_size 32 --comment gdcm_fusion