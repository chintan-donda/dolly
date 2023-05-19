import torch
from langchain.llms.base import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

class CustomLLM(LLM):
    def __init__(self, model_name_or_path):
        self.model_name = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )

    def _call(self, prompt, stop=None):
        return self.pipeline(prompt, max_length=1024)[0]["generated_text"]
 
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    def _llm_type(self):
        return "custom"