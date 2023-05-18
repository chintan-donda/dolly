# Dolly

Databricks’ [Dolly](https://huggingface.co/databricks/dolly-v2-12b) is an instruction-following large language model trained on the Databricks machine learning platform
that is licensed for commercial use. Based on `pythia-12b`, Dolly is trained on ~15k instruction/response fine tuning records
[`databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) generated
by Databricks employees in capability domains from the InstructGPT paper, including brainstorming, classification, closed QA, generation,
information extraction, open QA and summarization. `dolly-v2-12b` is not a state-of-the-art model, but does exhibit surprisingly
high quality instruction following behavior not characteristic of the foundation model on which it is based.

Databricks is committed to ensuring that every organization and individual benefits from the transformative power of artificial intelligence. The Dolly model family represents our first steps along this journey, and we’re excited to share this technology with the world.

The model is available on Hugging Face as [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b).

## Model Overview

`dolly-v2-12b` is a 12 billion parameter causal language model created by [Databricks](https://databricks.com/) that is derived from
[EleutherAI’s](https://www.eleuther.ai/) [Pythia-12b](https://huggingface.co/EleutherAI/pythia-12b) and fine-tuned
on a [~15K record instruction corpus](https://github.com/databrickslabs/dolly/tree/master/data) generated by Databricks employees and released under a permissive license (CC-BY-SA)


## Known Limitations

### Performance Limitations
**`dolly-v2-12b` is not a state-of-the-art generative language model** and, though quantitative benchmarking is ongoing, is not designed to perform
competitively with more modern model architectures or models subject to larger pretraining corpuses.

The Dolly model family is under active development, and so any list of shortcomings is unlikely to be exhaustive, but we include known limitations and misfires here as a means to document and share our preliminary findings with the community.
In particular, `dolly-v2-12b` struggles with: syntactically complex prompts, programming problems, mathematical operations, factual errors,
dates and times, open-ended question answering, hallucination, enumerating lists of specific length, stylistic mimicry, having a sense of humor, etc.
Moreover, we find that `dolly-v2-12b` does not have some capabilities, such as well-formatted letter writing, present in the original model.

### Dataset Limitations
Like all language models, `dolly-v2-12b` reflects the content and limitations of its training corpuses.

- **The Pile**: GPT-J’s pre-training corpus contains content mostly collected from the public internet, and like most web-scale datasets,
it contains content many users would find objectionable. As such, the model is likely to reflect these shortcomings, potentially overtly
in the case it is explicitly asked to produce objectionable content, and sometimes subtly, as in the case of biased or harmful implicit
associations.

- **`databricks-dolly-15k`**: The training data on which `dolly-v2-12b` is instruction tuned represents natural language instructions generated
by Databricks employees during a period spanning March and April 2023 and includes passages from Wikipedia as references passages
for instruction categories like closed QA and summarization. To our knowledge it does not contain obscenity, intellectual property or
personally identifying information about non-public figures, but it may contain typos and factual errors.
The dataset may also reflect biases found in Wikipedia. Finally, the dataset likely reflects
the interests and semantic choices of Databricks employees, a demographic which is not representative of the global population at large.

Databricks is committed to ongoing research and development efforts to develop helpful, honest and harmless AI technologies that
maximize the potential of all individuals and organizations.

## Getting Started with Response Generation

If you'd like to simply test the model without training, the model is available on Hugging Face as [databricks/dolly-v2-12b](https://huggingface.co/databricks/dolly-v2-12b).

To use the model with the `transformers` library on a machine with A100 GPUs:

```
from transformers import pipeline
import torch

instruct_pipeline = pipeline(model="databricks/dolly-v2-12b", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
```

You can then use the pipeline to answer instructions:

```
instruct_pipeline("Explain to me the difference between nuclear fission and fusion.")
```

### Generating on Other Instances

A100 instance types are not available in all cloud regions, or can be hard to provision. Inference is possible on other GPU instance types.

#### A10 GPUs

The 6.9B and 2.8B param models should work as-is.

To generate using the 12B param model on A10s (ex: `g5.4xlarge`, 1 x A10 24GB), it's necessary to load and run generating using 8-bit weights, which impacts the results slightly:

- Also install `bitsandbytes`
- Add `model_kwargs={'load_in_8bit': True}` to the `pipeline()` command shown above

#### V100 GPUs

When using V100s (ex: `p3.2xlarge`, 1 x V100 16GB, `NC6s_v3`), in all cases, set `torch_dtype=torch.float16` in `pipeline()` instead.

Otherwise, follow the steps above. The 12B param model may not function well in 8-bit on V100s.

## Getting Started with Training

- Add the `dolly` repo to Databricks (under Repos click Add Repo, enter `https://github.com/databrickslabs/dolly.git`, then click Create Repo).
- Start a `12.2 LTS ML (includes Apache Spark 3.3.2, GPU, Scala 2.12)` single-node cluster with node type having 8 A100 GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`). Note that these instance types may not be available in all regions, or may be difficult to provision. In Databricks, note that you must select the GPU runtime first, and unselect "Use Photon", for these instance types to appear (where supported).
- Open the `train_dolly` notebook in the Repo (which is the `train_dolly.py` file in the Github `dolly` repo), attach to your GPU cluster, and run all cells.  When training finishes, the notebook will save the model under `/dbfs/dolly_training`.

### Training on Other Instances

A100 instance types are not available in all cloud regions, or can be hard to provision. Training is possible on other GPU instance types, 
for smaller Dolly model sizes, and with small modifications to reduce memory usage.
These modifications are not optimal, but are simple to make.

#### A10 GPUs

Training the 12B param model is not recommended on A10s.

To train the 6.9B param model on A10 instances (ex: `g5.24xlarge`, 4 x A10 24GB; `Standard_NV72ads_A10_v5`, 2 x A10), make the following changes:

- Set `per-device-train-batch-size` and `per-device-eval-batch-size` to 3 in the `train_dolly.py` invocation of `deepspeed`
- Modify the deepspeed config file `ds_z3_bf16_config.json` to configure optimizer offload. Within the `"zero_optimization"` section, add:
  ```
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  },
  ```
- Set the `num_gpus` widget in `train_dolly` to the number of GPUs in your instance, such as 2 or 4, before running

To train the 2.8B param model:

- Instead, only set `per-device-train-batch-size` and `per-device-eval-batch-size` to 3 in the `train_dolly.py` invocation of `deepspeed`

#### V100 GPUs

To run on V100 instances with 32GB of GPU memory (ex: `p3dn.24xlarge` or `Standard_ND40rs_v2`), follow instructions above, and add:

- Modify `training/trainer.py` to disable `bf16` and enable `fp16` in `TrainingArguments`:
  ```
  ...
  fp16=True,
  bf16=False,
  ...
  ```
  
You may be able to slightly increase the batch size with 32GB instances, compared to what works above for 24GB A10s.


### Training on On-Prem GPUs

To train the model on On-Prem GPUs like V100 with 32GB of GPU memory (ex: `Tesla V100-SXM2-32GB`), follow the below instructions:
- Make sure you've `python3.8` or higher version.
- `git clone git@github.com:chintan-donda/dolly.git`
- Create Virtual env
  - If not installed: `python3.8 -m pip install virtualenv`
  ```
  python3.8 -m virtualenv venv_dolly
  source venv_dolly/bin/activate
  ```
- Install these additional NVIDIA libraries for Databricks Runtime 13.0 ML. Install these first and then only install the requirements.txt
  ```
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb -O /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-7_11.10.1.25-1_amd64.deb -O /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb -O /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-7_10.2.10.91-1_amd64.deb -O /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb
  dpkg -i /tmp/libcusparse-dev-11-7_11.7.3.50-1_amd64.deb
  dpkg -i /tmp/libcublas-dev-11-7_11.10.1.25-1_amd64.deb
  dpkg -i /tmp/libcusolver-dev-11-7_11.4.0.1-1_amd64.deb
  dpkg -i /tmp/libcurand-dev-11-7_10.2.10.91-1_amd64.deb
  ```
- Install requirements: `python3.8 -m pip install -r requirements_dev.txt`
- Make sure that `nvcc -V` and `nvidia-smi` CUDA version are same or at least with same major version.
  - Like `nvcc -V == 11.3` and `nvidia-smi == 11.7` ⇒ this works as the major version `11.x` are same
  - If not, upgrade/downgrade the CUDA versions to make it compatible with each other
    ```
    pip install torch==2.0.0+cu117 torchaudio==2.0.2+cu117 torchvision==0.15.2+cu117
    pip install pytorch-cuda==11.7
    ```
- If there is OOM (Out-Of-Memory) issue during the training, we can try training in the low bit mode as below:
  - Change the below lines `training/trainer.py`
    ```
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_cache=False if gradient_checkpointing else True,
        load_in_8bit=True, device_map='auto'
        offload_folder='./offload', offload_state_dict=True,
    )
    ```
  - You need to install: `pip install bitsandbytes`

Once the installation is done, train the model as below:
```
chmod +x train.sh
./train.sh
```

- train.sh has some hard-coded values. Change it based on the requirements. Ex: It uses `EleutherAI/pythia-2.8b` as the default base model. You can change it to any one from the [supported models](https://github.com/chintan-donda/dolly/blob/master/training/consts.py#L2).
- By default, it starts training on all the available GPUs using the argument `--num_gpus=8` in the `deepspeed` command. To force it using only a couple of them, remove `--num_gpus=8` and export the `CUDA_VISIBLE_DEVICES` as below:
  - `export CUDA_VISIBLE_DEVICES="1,2,3"` It's the GPU index that would be used. To check for the free GPU index: `gpustat` or `nvidia-smi`. And pick the index of your choice and pass it here. Below values "1,2,3" means the training would be done on 3 GPUs (1st, 2nd, and 3rd index GPUs).


### Common issues and their fixes when training on On-Prem GPUs

- If error like: `Exception: >- DeepSpeed Op Builder: Installed CUDA version 9.1 does not match the version torch was compiled with 11.8, unable to compile cuda/cpp extensions without a matching cuda version`,
    - Add in `~/.bashrc`
        ```
        export CUDA_HOME="/usr/local/cuda-11.3”
        export LD_LIBRARY_PATH="/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH"
        export PATH="/usr/local/cuda-11.3/bin:$PATH"
        ```
      - NOTE: Replace `11.3` with your installed CUDA version. `nvcc -V` and `nvidia-smi` major versions should match. Ex: `11.x` works fine. But the one having `11.x` and the other having `10.x` doesn't work.
- If error like: `ValueError: Tokenizer class GPTNeoXTokenizer does not exist or is not currently imported`
    ```
    python3.8 -m pip install sentencepiece
    python3.8 -m pip install git+https://github.com/huggingface/transformers
    python3.8 -m pip install git+https://github.com/zphang/transformers.git@neox20b
    python3.8 -m pip install huggingface-hub
    ```
- If error like: `ModuleNotFound error: tensorboardX not found`:
    ```
    python3.8 -m pip install tensorboardX
    pip install tensorboard==1.15.0
    ```
- If error like: **`cannot import name '_psutil_linux' from partially initialized module 'psutil' (most likely due to a circular import) (/usr/lib/python3/dist-packages/psutil/__init__.py)`**
    - `python3.8 -m pip install -U psutil`


## Generate sentences using the newly trained model

Generate the sentences using the newly trained model as below:
```
model_path = '/path/to/checkpoint'
from training.generate import load_model_tokenizer_for_generate, generate_response
model, tokenizer = load_model_tokenizer_for_generate(model_path)
instruction='Write a tweet to introduce Dolly, a model to mimic ChatGPT.'
response = generate_response(instruction, model, tokenizer)
print(response)
```


## Running Unit Tests Locally

```
pyenv local 3.8.13
python -m venv .venv
. .venv/bin/activate
pip install -r requirements_dev.txt
./run_pytest.sh
```
