from training.generate import InstructionTextGenerationPipeline, load_model_tokenizer_for_generate
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# Load model and tokenizer
input_model = "databricks/dolly-v2-3b"
input_model = './dolly_2023-05-19_13-19-23'

# Open QA
instructions = [
    "Explain to me the difference between nuclear fission and fusion.",
    "Give me a list of 5 science fiction books I should read next.",
]

# Closed QA
context = (
    """Monitoring insect infestations
        1. Frequent monitoring of grain storage conditions to identify possible problems associated with storage moisture and temperature and pest infestation is important. Due to their small size, visual assessment of insect infestation in grain stores is often difficult or ineffective, unless large numbers of insects are present.
        2. Traps baited with synthetic aggregation pheromones have been developed for detection and monitoring of stored grain insect pests.
        3. Low cost, combined with species specificity, makes pheromone-baited traps ideal monitoring tools in developing countries (Campion et al., 1987).
        4. Grains should be also kept along with, insecticides (chemical, botanical) to control the infestation of grains from storage pests.

        DO’S AND DON’TS IN IPM
        Do’s
        Don’ts
        1
        Clean the area from all existing vegetation, stumps, roots and stones
        Don’t select plain area for nursery bed.
        2
        Prepare bed with 1 meter width, 20 cm height and of required length
        Don’t make too wide nursery bed
        3
        Fumigate the beds with 2% formalin (2 l/100 l of water) under polythene cover for 48 hrs (10 l/bed) or do solarization.
        Don’t sow seed within week of fumigation
        4
        Grow only recommended varieties.
        Do not grow varieties not suitable for the season or the region.
        5
        Collect ripened bold capsules from disease free mother clumps from 2nd and 3rdharvests for seed extraction.
        Don’t collect unripened capsules for seed
        6
        Sow the seed in September preferably
        Avoid sowing before September
        7
        Always treat the seeds with approved biopesticides/ chemicals for the control of seed borne diseases/ pests.
        Do not use seeds without seed treatment with biopesticides/chemicals.
        8
        Cover the bed with mulch material either with pot grass or paddy straw
        Don’t throw away the topsoil
    """
)

question = "What are the DO’S AND DON’TS IN IPM?"


def huggingface_pipeline(input_model):
    model, tokenizer = load_model_tokenizer_for_generate(input_model)
    import pdb; pdb.set_trace()

    # template for an instrution with no input
    prompt = PromptTemplate(
        input_variables=["instruction"],
        template="{instruction}")

    # template for an instruction with input
    prompt_with_context = PromptTemplate(
        input_variables=["instruction", "context"],
        template="{instruction}\n\nInput:\n{context}")

    # task can be any of the listed here: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task
    hf_pipeline = HuggingFacePipeline(
        pipeline=InstructionTextGenerationPipeline(
            # Return the full text, because this is what the HuggingFacePipeline expects.
            model=model, tokenizer=tokenizer, return_full_text=True, task="text-generation"))

    llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
    llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

    # Use the model to generate responses for each of the instructions above.
    for instruction in instructions:
        response = llm_chain.predict(instruction=instruction)
        print(f"Instruction: {instruction}\n\n{response}\n\n-----------\n")

    response = llm_context_chain.predict(instruction=question, context=context)
    print(f"Instruction: {question}\n\nContext:\n{context}\n\nResponse:\n{response}\n\n-----------\n")

huggingface_pipeline(input_model)



# ================== Custom LLM Wrapper ==================
from custom_llm import CustomLLM
from llama_index import LLMPredictor

llm = LLMPredictor(
    llm=CustomLLM(
        model_name_or_path=input_model),
    load_in_8bit=True,
    device_map='auto'
)

model, tokenizer = load_model_tokenizer_for_generate(input_model)
llm = HuggingFacePipeline(
    pipeline=InstructionTextGenerationPipeline(
        # Return the full text, because this is what the HuggingFacePipeline expects.
        model=model, tokenizer=tokenizer, return_full_text=True, task="text-generation"))

prompt_template = """You are a helpful AI assistant. Use the following pieces of context to answer the question comprehensively at the end. Start the answer by giving short summary and write the answer starting with Here are some of the key points:. Write each sentence separately with numbering. If you don't know the answer, just say that you don't know, don't try to make up an answer. If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

    {context}

    Question: {question}

    Answer in English:
"""

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS

text_splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separator='\n'
)
texts = text_splitter.split_text(context)

# pip install sentence_transformers
embeddings = SentenceTransformerEmbeddings()

# pip install faiss-cpu==1.7.3
# Find similar docs that are relevant to the question
docsearch = FAISS.from_texts(
    texts, embeddings,
    metadatas=[{"source": str(i+1)} for i in range(len(texts))]
)

# Search for the similar docs
docs = docsearch.similarity_search(question, k=2)

from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
chain = load_qa_chain(llm, chain_type='stuff', prompt=PROMPT)
out_dict = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
print(out_dict['output_text'])
# ========================================================================
