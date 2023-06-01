from dataclasses import dataclass
from optimum.bettertransformer import BetterTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import deepspeed

MODEL_CLASSES = {
    "codegen": (AutoModelForCausalLM, AutoTokenizer),
    "codegen2": (AutoModelForCausalLM, AutoTokenizer),
    "codet5p" : (AutoModelForSeq2SeqLM, AutoTokenizer)
    #Add more model classes
}

@dataclass
class HuggingFaceModel():
    dtype: str
    bf: bool
    deepspeed: bool


    def load(
            self, 
            model_type: str, 
            model_name_or_path: str
) -> None:
        try:
            model_class, tokenizer_class = MODEL_CLASSES[model_type.lower()]

            self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
            if(self.deepspeed):
                self.__load_model_deepspeed(model_class, model_name_or_path)
            else:
                self.__load_model(model_class, model_name_or_path)
            self.model.eval()
        except KeyError:
            raise KeyError(f"The model you specified is not supported ({model_type})")
        except OSError as e:
            raise OSError(f'Error loading tokenizer/model. Output: {e}')
        except ValueError:
            raise 


    def __load_model_deepspeed(self, model_class, model_name_or_path):
        model =  model_class.from_pretrained(model_name_or_path)
        dtype = torch.float if self.dtype == "fp32" else torch.half 
        config = {"tensor_parallel": {"tp_size": torch.cuda.device_count()}, 
                  'dtype': dtype, 
                  "replace_with_kernel_inject": True}
        self.model = deepspeed.init_inference(model=model, config=config)

    def __load_model(self, model_class, model_name_or_path):
        if(self.dtype == "int8"):
            args = {'device_map': 'auto', 'torch_dtype': torch.half}
            self.model = model_class.from_pretrained(model_name_or_path, load_in_8bit=True, **args)
        elif(self.dtype == "fp16"):
            args = {'device_map': 'auto', 'torch_dtype': torch.half}
            self.model = model_class.from_pretrained(model_name_or_path, **args)
        elif(self.dtype == "fp32"):
            args = {'device_map': 'auto'}
            self.model = model_class.from_pretrained(model_name_or_path, **args)
        if(self.bf):
            self.model = BetterTransformer.transform(self.model)


    def generate(
            self, 
            prompt: str, 
            max_new_tokens: int,
            k: int,
            p: float,
            temp: float, 
            repetition_penalty: float,
            sample_size: int
) -> torch.LongTensor:
        device = torch.device("cuda")
        try:
            with torch.no_grad():
                ids = self.tokenizer(prompt, truncation=True, return_tensors="pt").input_ids.to(device)
                generated_ids = self.model.generate(
                    ids, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=True, 
                    top_k=k, 
                    temperature=temp, 
                    top_p=p, 
                    num_return_sequences=sample_size, 
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        except:
            raise
        return generated_ids

    def decode_tensor(self, input: torch.LongTensor) -> str:
        return self.tokenizer.decode(input, skip_special_tokens=True)

    def decode_tensor_batch(self, input: list[torch.LongTensor]) -> list[str]:
        return self.tokenizer.batch_decode(input, skip_special_tokens=True)
    