import pandas as pd
from typing import List
import logging
import sys
from HuggingFaceInference import HuggingFaceModel
from pathlib import Path
from argparse import Namespace, ArgumentParser
import shutil
import os
import ast
import json
from function_extractors.c_function_extractor import extract_c_function

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

MODEL_CLASSES = {
    "codegen": '',
    "codegen2": '',
    "codet5p": ''
    #Add more model classes
}

file_types = {
    "python": ".py",
    "c": ".c",
    "cpp": ".cpp",
    "js": ".js"
}

MAX_NUMBER_OF_RETRIES = 5

FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = f'{__name__}.log'
logger = logging.getLogger(__name__)
logger.propagate = False
timeout = 600

def get_console_handler() -> logging.StreamHandler:
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.DEBUG)
    c_handler.setFormatter(FORMATTER)
    return c_handler

def get_file_handler() -> logging.FileHandler:
    f_handler = logging.FileHandler(LOG_FILE, "w+")
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(FORMATTER)
    return f_handler

def get_logger() -> None:
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.setLevel(logging.DEBUG)

def load_prompts(path: str, column_name: str) -> List[str]:
    if(not Path(path).exists()):
        logger.warning(f'File {path} does not exist. Aborting...')
        sys.exit(1)
    if(path.endswith('.csv')):
        df = pd.read_csv(path)
    elif(path.endswith('.zip')):
        df = pd.read_csv(path, compression= 'zip')
    elif path.endswith('.json'):
        df = pd.read_json(path)
    elif path.endswith('.parquet.gzip'):
        df = pd.read_parquet(path)

    prompts_outer = list()
    prompts = df[column_name].tolist()
    descriptions = df['description'].tolist()
    for idx, prompt_list in enumerate(prompts):
        prompts_inner = list()
        prompt_list =ast.literal_eval(prompt_list)
        for prompt in prompt_list:
            prompts_inner.append(f"{descriptions[idx]}\n{prompt}")
        prompts_outer.append(prompts_inner)
    return prompts_outer

def load_model(args: Namespace) -> HuggingFaceModel:
    HuggingFaceModelInstance = HuggingFaceModel(args.dtype, args.bf, args.deepspeed)
    try:
        HuggingFaceModelInstance.load(args.model_type, args.model_name_or_path)
    except Exception as e:
        logger.debug(f'Error loading Huggingface model. Output: {e}')
        sys.exit(1)
    return HuggingFaceModelInstance

def create_files(
        prompt_idx: int, 
        sample_idx: int,
        outputs: List[str], 
        path: Path, 
        lang: str
) -> None:
    sample_dir = (path / f"sample{sample_idx}")
    if(not sample_dir.exists()):
        sample_dir.mkdir()
    prompt_dir = (sample_dir / f"prompt{prompt_idx}")
    if(prompt_dir.exists()):
        shutil.rmtree(prompt_dir)
    prompt_dir.mkdir()
    file_type = file_types[lang]
    for i, output in enumerate(outputs):
        filename = f'output{i}{file_type}'
        filepath = prompt_dir / filename
        with open(filepath, "w") as f:
            f.write(output)


def extract_function(decoded_outputs: List[str], lang: str, prompt: str) -> List[str]:
    if(lang.lower() == "c"):
        new_outputs = extract_c_function(decoded_outputs, prompt)
    elif(lang.lower() == "python"):
        pass
        #new_outputs = extract_python_function(decoded_outputs, prompt)
    elif(lang.lower() == "cpp"):
        pass
        #new_outputs = extract_cpp_function(decoded_outputs, prompt)
    return new_outputs

def functional_output_generation(args: Namespace, prompt: str, model: HuggingFaceModel) -> List[str]:
    correct_outputs = list()
    correct_outputs_amount = 0
    sample_size = args.sample_size
    num_retries = 0
    while(correct_outputs_amount < args.sample_size and num_retries < MAX_NUMBER_OF_RETRIES):
        try:
            output = model.generate(
                prompt=prompt, 
                k=args.k, 
                p=args.p, 
                temp=args.temp, 
                repetition_penalty=args.repetition_penalty,
                sample_size=sample_size
            )
        except Exception as e:
            logger.debug(f'Error generating output. Output: {e}')
            return []
        decoded_output = model.decode_tensor_batch(output)
        extracted_functions = extract_function(decoded_output, args.lang, prompt)
        correct_outputs.append(extracted_functions)
        correct_outputs_amount+=len(extracted_functions)
        sample_size-=correct_outputs_amount
        num_retries+=1
    return [item for sublist in correct_outputs for item in sublist]

def main(args: Namespace, path: Path) -> None:
    logger.info(args)
    logger.debug(f'Loading prompts...')
    prompts = load_prompts(args.prompts, args.column_name)
    logger.debug(f'{len(prompts)} prompts loaded...')
    logger.debug(f'Loading model...')
    model = load_model(args)
    for sample_idx, prompt_list in enumerate(prompts):
        for p_idx, prompt in enumerate(prompt_list):
            logger.debug(f'Generating {args.sample_size} functional output(s) for sample {sample_idx}, prompt {p_idx}')
            print(prompt)
            correct_outputs = functional_output_generation(args, prompt, model)
            logger.debug(f'{len(correct_outputs)}/{args.sample_size} correct outputs generated.')
            if(len(correct_outputs) > 0):
                logger.debug(f"Creating files...")
                create_files(p_idx, sample_idx, correct_outputs, path, args.lang.lower())
            else:
                logger.debug(f"No correct outputs for prompt {p_idx}. Not creating files. Skipping...")
    stats_file = (path / 'stats.txt')
    with open(stats_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)

if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    parser = ArgumentParser()
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument("--sample_size", type=int, default=50, help="#Samples created by the model for each prompt" )
    parser.add_argument("--temp", type=float, default=0.5, help="Temperature of the model")
    parser.add_argument("--path", type=str, help="Directory for the created code files")
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default='fp32', help="Dtype of the model (int8, fp16, fp32)")
    parser.add_argument("--deepspeed", action='store_true', help="Whether to use DeepSpeed library")
    parser.add_argument("--bf", action='store_true', help="Whether to use BetterTransformer")
    parser.add_argument("--prompts", type=str, help="Path to the file containing the prompts. (csv, json, parquet)")
    parser.add_argument("--column_name", type=str, help="Column name of the prompts in the dataframe")
    parser.add_argument("--lang", type=str, help="Programming language")
    

    arguments = parser.parse_args()

    get_logger()
    file_path = Path(arguments.path).resolve()
    if(not Path(arguments.path).exists()):
        logger.debug(f'Creating target directory...')
        file_path.mkdir()
    main(args=arguments, path=file_path)
