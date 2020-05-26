CUDA_VISIBLE_DEVICES=1 python run_glue.py \
--per_gpu_train_batch_size 17 \
--per_gpu_eval_batch_size 17 \
--gradient_accumulation_steps 10 \
--wandb_name snli-synthesizer-full_att-mix-mmae \
--full_att \
--mix \
--albert \
--synthesizer \
--do_train \
--do_eval \
--evaluate_during_training \
--output_dir output/SNLI-mix-mmae \
--model_name_or_path albert-base-v2 \
--task_name SNLI \
--data_dir data_snli/SNLI \

