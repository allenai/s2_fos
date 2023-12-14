# Bash script for running model training
# Training parameters and Asymetric loss function parameters were obtained through running ablation experiments
python train_net.py --train_data <path to training data> --test_data <path to test data> \
--val_data <path to validation data> --text_fields title abstract journal_name  --save_path /stage/output/ \
--train True --model_checkpoint_path /stage/output/ --project_name <weights and biases project name> \
--batch_size 32 --learning_rate 1e-5 --warmup_ratio 0.06 --wandb_name <run name> --wandb_run_des <run description> \
 --log_dir /output --loss Asymmetric --asym_gamma_neg 4 --asym_gamma_pos 2 --asym_clip 0.1