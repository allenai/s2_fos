# Bash script for running model training
# Training parameters and Asymetric loss function parameters were obtained through running ablation experiments
python train_net.py --text_fields title abstract journal_name  --save_path /stage/output/ \
--train True --model_checkpoint_path /stage/output/ --project_name <weights and biases project name> \
--batch_size 32 --learning_rate 1e-5 --warmup_ratio 0.06 --wandb_name <run name> --wandb_run_desc <run description> \
 --log_dir /output --loss Asymmetric --asym_gamma_neg 4 --asym_gamma_pos 2 --asym_clip 0.1