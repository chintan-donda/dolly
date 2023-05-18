# pip install -r requirements_dev.txt

# wget "https://cloud.tsinghua.edu.cn/f/498512c3c1724558830d/?dl=1" -O parquet-train.arrow
# mkdir -p ./model
# pushd ./model
# wget "https://cloud.tsinghua.edu.cn/f/8bfd19e6cb1a4a289c1b/?dl=1" -O added_tokens.json
# wget "https://cloud.tsinghua.edu.cn/f/231ddebf6caf49b38ce8/?dl=1" -O config.json
# wget "https://cloud.tsinghua.edu.cn/f/79e402dcc503430db9a1/?dl=1" -O merges.txt
# wget "https://cloud.tsinghua.edu.cn/f/001e6641d7324635bc77/?dl=1" -O special_tokens_map.json
# wget "https://cloud.tsinghua.edu.cn/f/2d68e62358da4b7f94e6/?dl=1" -O tokenizer.json
# wget "https://cloud.tsinghua.edu.cn/f/5ebcd4f2380147e3bee8/?dl=1" -O tokenizer_config.json
# wget "https://cloud.tsinghua.edu.cn/f/34aa75355590497ba28b/?dl=1" -O vocab.json
# wget "https://cloud.tsinghua.edu.cn/f/cd59c04366674ab592b0/?dl=1" -O pytorch_model.bin
# popd

export timestamp=`date +%Y-%m-%d_%H-%M-%S`
export model_name='dolly'
export input_model='EleutherAI/pythia-2.8b'
export checkpoint_dir_name="${model_name}_${timestamp}"
export deepspeed_config=`pwd`/config/ds_z3_bf16_config.json
export local_output_dir="./${checkpoint_dir_name}"
export dbfs_output_dir=''
export tensorboard_display_dir="${local_output_dir}/runs"
export DATASET_FILE_PATH=`pwd`/parquet-train.arrow
export MODEL_PATH=`pwd`/model/


# timestamp = '2023-05-15 04:04:55'
# model_name = 'dolly'
# input_model = 'EleutherAI/pythia-2.8b'
# checkpoint_dir_name = f"{model_name}_{timestamp}"
# deepspeed_config = f'{os.getcwd()}/config/ds_z3_bf16_config.json'
# local_output_dir = f"./{checkpoint_dir_name}"
# dbfs_output_dir = ''
# tensorboard_display_dir = f"{local_output_dir}/runs"
# DATASET_FILE_PATH = f'{os.getcwd()}/parquet-train.arrow'
# MODEL_PATH = f'{os.getcwd()}/model/'
# CUDA_VISIBLE_DEVICES = 1
# PATH = f"/usr/local/cuda/bin:{os.getenv('PATH')}"


export CUDA_VISIBLE_DEVICES="2,3,5"
export DS_SKIP_CUDA_CHECK=1
export CUDA_LAUNCH_BLOCKING=1
# --num_gpus=1
taskset -c 0-5 deepspeed \
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



# CUDA SETUP: Problem: The main issue seems to be that the main CUDA runtime library was not detected.
# CUDA SETUP: Solution 1: To solve the issue the libcudart.so location needs to be added to the LD_LIBRARY_PATH variable
# CUDA SETUP: Solution 1a): Find the cuda runtime library via: find / -name libcudart.so 2>/dev/null
# CUDA SETUP: Solution 1b): Once the library is found add it to the LD_LIBRARY_PATH: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:FOUND_PATH_FROM_1a
# CUDA SETUP: Solution 1c): For a permanent solution add the export from 1b into your .bashrc file, located at ~/.bashrc

# CUDA SETUP: Solution 2: If no library was found in step 1a) you need to install CUDA.
# CUDA SETUP: Solution 2a): Download CUDA install script: wget https://github.com/TimDettmers/bitsandbytes/blob/main/cuda_install.sh
# CUDA SETUP: Solution 2b): Install desired CUDA version to desired location. The syntax is bash cuda_install.sh CUDA_VERSION PATH_TO_INSTALL_INTO.
# CUDA SETUP: Solution 2b): For example, "bash cuda_install.sh 113 ~/local/" will download CUDA 11.3 and install into the folder ~/local
