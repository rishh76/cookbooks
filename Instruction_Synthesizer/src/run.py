from vllm import LLM, SamplingParams
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from datasets import DatasetDict
from huggingface_hub import HfApi, HfFolder
from typing import List, Dict
import tqdm
import random
import os

# Templates (please do NOT change them)
context_template = ' <CON> {context} </CON>'
QA_template = '<QUE> {question} <ANS> {answer} </END>'
delimiter = '\n\n'
bos_token = '<s>'
eos_token = '</s>'

# Create a sampling params object.
temp = float(os.getenv("temprature")) if os.getenv("temprature") else 0
maxt = int(os.getenv("max_tokens")) if os.getenv("temprature") else 400

sampling_params = SamplingParams(temperature=temp, max_tokens=maxt)

# Load the model and tokenizer
llm = LLM(model="instruction-pretrain/instruction-synthesizer", max_model_len=4096)

def cook_context(raw_context):
    """Format the context."""
    return bos_token + context_template.replace('{context}', raw_context) + delimiter

def cook_instruction_response_pairs(QA_list):
    """Format downstream instruction(Q)-response(A) pairs."""
    ins_res_list = []
    for qa_entry in QA_list:
        qa = QA_template.replace('{question}', qa_entry['Q']).replace('{answer}', qa_entry['A'])
        ins_res_list.append(qa)
    return delimiter.join(ins_res_list) + eos_token

def parse_pred(pred):
    """Extract the list of instruction-response pairs from the prediction"""
    QA_str_list = pred.split('</END>')
    if not pred.endswith('</END>'):
        QA_str_list = QA_str_list[:-1]

    QA_list = []
    raw_questions = []
    for QA_str in QA_str_list:
        try:
            assert len(QA_str.split('<ANS>')) == 2, f'invalid QA string: {QA_str}'
            Q_str, A_str = QA_str.split('<ANS>')
            Q_str, A_str = Q_str.strip(), A_str.strip()
            assert Q_str.startswith('<QUE>'), f'invalid question string: {Q_str} in QA_str: {QA_str}'
            assert len(A_str) > 0, f'invalid answer string in QA_str: {QA_str}'
            Q_str = Q_str.replace('<QUE>', '').strip()
            assert Q_str.lower() not in raw_questions, f'duplicate question: {Q_str}'
            QA_list.append({'Q': Q_str, 'A': A_str})
            raw_questions.append(Q_str.lower())
        except:
            pass
    return QA_list

def get_instruction_response_pairs(context):
    '''Prompt the synthesizer to generate instruction-response pairs based on the given context'''
    outputs = llm.generate(context, sampling_params, use_tqdm=False)
    pred = outputs[0].outputs[0].text
    return parse_pred(pred)


def load_and_batch_dataset(dataset_name, k=2, seed=42):
    """
    Load a dataset from Hugging Face, split it into batches of size k with random sampling without repetition.

    Args:
    dataset_name (str): The name of the dataset to load.
    k (int): The size of each batch.
    seed (int): The random seed for reproducibility. Default is 42.

    Returns:
    list of list: A list of batches, each containing k samples.
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)

    # Ensure there is a 'train' split to work with
    if 'train' not in dataset:
        raise ValueError(f"The dataset '{dataset_name}' does not have a 'train' split.")

    # Extract the 'text' column from the dataset
    text_data = dataset['train']['text']

    # Shuffle the text data with a fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(text_data)

    # Create batches of size k
    batches = [text_data[i:i + k] for i in range(0, len(text_data), k)]

    return batches

def process_batches_to_hf_dataset(batches: List[List[str]]) -> DatasetDict:
    """
    Process each batch of texts and generate instruction-response pairs in multi-round inference.
    Return a Hugging Face dataset with columns 'text' and 'instruction_response'.

    Args:
    batches (List[List[str]]): List of batches, where each batch is a list of texts.

    Returns:
    DatasetDict: A Hugging Face DatasetDict containing the processed data.
    """
    processed_data = {'text': [], 'instruction_response': []}

    for batch in tqdm.tqdm(batches):
        previous_examples = []

        for cur_text in batch:
            # Prepend raw texts and instruction-response pairs of previous examples to the current text
            context = ''
            for previous_example in previous_examples:
                context += cook_context(previous_example['text'])
                context += cook_instruction_response_pairs(previous_example['instruction_response_pairs'])
            context += cook_context(cur_text)

            # Get the generated instruction-response pairs
            instruction_response_pairs = get_instruction_response_pairs(context)

            # Append the current text and its instruction-response pairs to the processed data
            processed_data['text'].append(cur_text)
            processed_data['instruction_response'].append(instruction_response_pairs)

            # Update previous examples
            previous_examples.append({'text': cur_text, 'instruction_response_pairs': instruction_response_pairs})

    # Create a Hugging Face Dataset from the processed data
    dataset = Dataset.from_dict(processed_data)

    # Create a DatasetDict with 'train' split (or any other split name you prefer)
    dataset_dict = DatasetDict({'train': dataset})

    return dataset_dict

def push_to_hf_hub(dataset_dict: DatasetDict, dataset_name: str, username: str, token: str = "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"):
    """
    Push a dataset to the Hugging Face Hub.

    Args:
    dataset_dict (DatasetDict): The Hugging Face DatasetDict to push.
    dataset_name (str): The name of the dataset.
    repo_id (str): The repository ID on Hugging Face Hub (e.g., 'username/dataset_name').
    token (str): The Hugging Face authentication token. If None, it will use the token stored in HfFolder.

    Returns:
    None
    """
    # Initialize the HfApi
    api = HfApi()

    # Use provided token or the one stored in HfFolder
    if token is None:
        token = HfFolder.get_token()

    if token is None:
        raise ValueError("You must provide a Hugging Face authentication token.")
    repo_id = username+"/"+dataset_name
    # Push the dataset to the Hugging Face Hub
    dataset_dict.push_to_hub(repo_id=repo_id, token=token)

    print(f"Dataset '{dataset_name}' has been pushed to the Hugging Face Hub at '{repo_id}'.")


input_dataset_name = os.getenv("input_dataset_name")
output_dataset_name = os.getenv("output_dataset_name")
username = os.getenv("username")
hf_token = os.getenv("hf_token")
batch_size = int(os.getenv("batch_size")) if os.getenv("batch_size") else 2
seed = int(os.getenv("seed")) if os.getenv("seed") else 42
    
# Load and batch the dataset
batches = load_and_batch_dataset(input_dataset_name, k=batch_size, seed=seed)
    
# Process batches to a Hugging Face dataset
processed_dataset_dict = process_batches_to_hf_dataset(batches)
    
# Push the processed dataset to the Hugging Face Hub
push_to_hf_hub(processed_dataset_dict, output_dataset_name,username,hf_token)