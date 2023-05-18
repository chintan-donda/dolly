export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='dolly'
export input_model='EleutherAI/pythia-2.8b'
export checkpoint_dir_name="${model_name}_${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_output_dir="./${checkpoint_dir_name}"

# Change values for below flag based on the requirement. It's the GPU index that would be used.
# To check for the free GPU index: `gpustat` or `nvidia-smi`. And pick the index of your choice and pass it here.
# Below values "2,3,5" means the training would be done on 3 GPUs (2nd, 3rd, and 5th index GPUs)
export CUDA_VISIBLE_DEVICES="2,3,5"
export DS_SKIP_CUDA_CHECK=1
export CUDA_LAUNCH_BLOCKING=1

# Train the model using deepspeed
# NOTE: Change the --per-device-train-batch-size to higher number if we've high end GPU machine. Ex: for 8 A100 GPUs, we can set it to 3/8/higher number based on experimentation and tuning.
deepspeed \
    --module training.trainer \
    --deepspeed $deepspeed_config \
    --epochs 1 \
    --local-output-dir $local_output_dir \
    --dbfs-output-dir "" \
    --per-device-train-batch-size 1 \
    --per-device-eval-batch-size 1 \
    --lr 1e-5 \
    --warmup-steps 50 \
    --input-model $input_model \
    --logging-steps 1000 \
    --test-size 1000
