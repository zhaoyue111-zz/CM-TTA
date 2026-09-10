export HF_ENDPOINT=https://hf-mirror.com

python run_sfda_voxtell.py --data_dir /mnt/afs2/zy/CT_MRI_DATA_3D --voxtell_root /mnt/afs2/zy/VoxTell_from_disk --model_dir /mnt/afs2/zy/VoxTell_from_disk/model --prompt liver --epochs 20 --eval_interval 5 --record_soft_prompt_grad_norm
