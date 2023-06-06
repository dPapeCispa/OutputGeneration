# OutputGeneration

## Arguments
```
usage: generate_output.py [-h] --model_type MODEL_TYPE --model_name_or_path MODEL_NAME_OR_PATH [--sample_size SAMPLE_SIZE] [--temp TEMP] [--path PATH] [--k K] [--p P]
                          [--repetition_penalty REPETITION_PENALTY] [--dtype DTYPE] [--deepspeed] [--bf] [--prompts PROMPTS] [--column_name COLUMN_NAME] [--lang LANG]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        Model type selected in the list: codegen, codegen2, codet5p
  --model_name_or_path MODEL_NAME_OR_PATH
                        Path to pre-trained model or shortcut name selected in the list: codegen, codegen2, codet5p
  --sample_size SAMPLE_SIZE
                        #Samples created by the model for each prompt
  --temp TEMP           Temperature of the model
  --path PATH           Directory for the created code files
  --k K
  --p P
  --repetition_penalty REPETITION_PENALTY
  --dtype DTYPE         Dtype of the model (int8, fp16, fp32)
  --deepspeed           Whether to use DeepSpeed library
  --bf                  Whether to use BetterTransformer
  --prompts PROMPTS     Path to the file containing the prompts. (csv, json, parquet)
  --column_name COLUMN_NAME
                        Column name of the prompts in the dataframe
  --lang LANG           Programming language
 ```

## Sample Usage

```
python3 generate_output.py --model_type codegen --model_name_or_path Salesforce/Codegen-2B-multi --sample_size 1 --temp 0.2 --path Code_files --k 0 --p 0.95 --dtype fp16 --bf --prompts DatasetGeneration/CWE_119_CWE_120_C_DB_finished.parquet.gzip --column_name prompts --lang C --deepspeed --repetition_penalty 1.01
```

## Setup
Use virtual environment with `python3.9.15`
### Requirements
To install requirements and dependencies, use 
```
pip install -r requirements.txt
```
