import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace

# gets role for executing training job
# role = sagemaker.get_execution_role()
iam = boto3.client('iam')
role = iam.get_role(RoleName='SageMakerRole-to-train=dolly-v2-3b')['Role']['Arn']

hyperparameters = {
	'model_name_or_path': 'databricks/dolly-v2-3b',
	'output_dir': '/opt/ml/model',
    'dataset_name': 'databricks/databricks-dolly-15k',
    'per_device_train_batch_size': 3,
    'per_device_eval_batch_size': 3,
    'do_train': True
	# add your remaining hyperparameters
	# more info here https://github.com/huggingface/transformers/tree/v4.17.0/examples/pytorch/language-modeling
}

# git configuration to download our fine-tuning script
git_config = {'repo': 'https://github.com/huggingface/transformers.git', 'branch': 'v4.17.0'}

# creates Hugging Face estimator
huggingface_estimator = HuggingFace(
	entry_point='run_qa.py',
	source_dir='./examples/pytorch/question-answering',
	instance_type='ml.g4dn.2xlarge',
	instance_count=1,
	role=role,
	git_config=git_config,
	transformers_version='4.17.0',
	pytorch_version='1.10.2',
	py_version='py38',
	hyperparameters = hyperparameters
)

# starting the train job
huggingface_estimator.fit()