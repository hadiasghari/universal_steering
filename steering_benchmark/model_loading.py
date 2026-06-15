"""
Shared model selection and loading for the steering benchmark and VLM scripts.

resolve_model_args() fills in per-family defaults so the scripts only need
--model_set (phi/llama/falcon/mistral/gemma); all four benchmark scripts embed
MODEL_VERSION and MODEL_SIZE in their cache/CSV filenames, so they must agree.
select_llm() was previously duplicated (and drifting) between run.py,
eval_generations.py, and vlm_steering/main.py; all now import from here so
direction extraction and steered generation always load models identically.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collections import namedtuple

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    MllamaForConditionalGeneration, AutoProcessor,
    LlavaForConditionalGeneration
)

from utils import LLMType

LLM = namedtuple('LLM', ['language_model', 'tokenizer', 'processor', 'name', 'model_type'])

# (default version, default size) per family -- the smaller variants
DEFAULTS = {
    'llama': ('3.1', '8B'),
    'falcon': ('3', '3B'),
    'phi': ('3-medium-4k-instruct', '14B'),
    'mistral': ('Small-Instruct-2409', '7B'),
    'gemma': ('2', '9B'),
    'llama-vision': ('3.2', '90B'),
    'llava': ('1.5', '7B'),
}

# size implied by a specific version, when --model_version is given without --model_size
VERSION_SIZES = {
    ('llama', '3.3'): '70B',
    ('mistral', 'Large-Instruct-2407-4bit'): '120B',
    ('phi', '4'): '14B',
}


def resolve_model_args(model_set, model_version=None, model_size=None):
    """Fill in missing model_version/model_size with the family defaults."""
    if model_set not in DEFAULTS:
        raise ValueError(f"Model set {model_set} not supported; choose from {list(DEFAULTS)}")
    default_version, default_size = DEFAULTS[model_set]
    if model_version is None:
        model_version = default_version
    if model_size is None:
        model_size = VERSION_SIZES.get((model_set, model_version), default_size)
    return model_version, model_size


def select_llm(model_type, MODEL_VERSION=None, MODEL_SIZE=None):
    """
    Load and configure a language model for steering (direction computation
    and steered generation).

    Args:
        model_type: Model family ('llama', 'phi', 'falcon', 'mistral', 'gemma',
                    'llama-vision', 'llava')
        MODEL_VERSION: Specific version within the family (family default if None)
        MODEL_SIZE: Model size variant (family default if None)

    Returns:
        LLM namedtuple containing the model, tokenizer, and metadata
    """
    MODEL_VERSION, MODEL_SIZE = resolve_model_args(model_type, MODEL_VERSION, MODEL_SIZE)

    if model_type == 'llama':
        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_id = "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_id = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

        if MODEL_SIZE == '8B':
            language_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
        else:
            language_model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto"  # no bf16 for unsloth's 4-bit models
            )

        use_fast_tokenizer = "LlamaForCausalLM" not in language_model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast_tokenizer, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0

        if MODEL_VERSION == '3.1' and MODEL_SIZE == '8B':
            model_name = 'llama_3_8b_it_eng_only'
        elif MODEL_VERSION == '3.1' and MODEL_SIZE == '70B':
            model_name = "llama_3.1_70b_it_eng_only"
        elif MODEL_VERSION == '3.3' and MODEL_SIZE == '70B':
            model_name = "llama_3.3_70b_it_eng_only"

        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'phi':
        if MODEL_VERSION == '4':
            model_id = "microsoft/phi-4"
            model_name = 'phi_4'
        elif MODEL_VERSION == '3':
            model_id = "microsoft/phi-3"
            model_name = 'phi_3'
        elif MODEL_VERSION == '3-medium-4k-instruct':
            model_id = "microsoft/Phi-3-medium-4k-instruct"
            model_name = 'phi_3_medium_4k_instruct'

        print(f"Loading Phi {MODEL_VERSION} model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'falcon':
        if MODEL_SIZE == '10B' and MODEL_VERSION == '3':
            model_id = "tiiuae/Falcon3-10B-Instruct"
            model_name = 'falcon3_10b_it'
        if MODEL_SIZE == '3B' and MODEL_VERSION == '3':
            model_id = "tiiuae/Falcon3-3B-Instruct"
            model_name = 'falcon3_3b_it'
        if MODEL_SIZE == '40B' and MODEL_VERSION == '1':
            model_id = "tiiuae/falcon-40B-instruct"
            model_name = 'falcon_40b_it'

        print(f"Loading Falcon {MODEL_SIZE} model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.padding_side = "left"  # ?
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'mistral':
        if MODEL_VERSION == 'Small-Instruct-2409':
            model_id = "mistralai/Mistral-Small-Instruct-2409"
            model_name = 'mistral_small_2409_it'
        elif MODEL_VERSION == 'Large-Instruct-2407-4bit':
            model_id = "unsloth/Mistral-Large-Instruct-2407-bnb-4bit"
            model_name = 'mistral_large_2407_4bit_it'

        print(f"Loading Mistral {MODEL_VERSION} model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'gemma':
        if MODEL_VERSION == '2' and MODEL_SIZE == '9B':
            model_id = "google/gemma-2-9b-it"
            model_name = 'gemma_2_9b_it'

        print(f"Loading Gemma {MODEL_VERSION} model")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        language_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        processor = None
        llm_type = LLMType.TEXT

    elif model_type == 'llama-vision':
        model_id = "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit"

        language_model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0
        processor = AutoProcessor.from_pretrained(model_id)
        model_name = 'llama_3_90b_4bit_it'
        llm_type = LLMType.MULTIMODAL

    elif model_type == 'llava':
        model_id = "llava-hf/llava-1.5-7b-hf"

        language_model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", legacy=False)
        tokenizer.pad_token_id = 0
        processor = AutoProcessor.from_pretrained(model_id)
        model_name = 'llava-1.5-7b'
        llm_type = LLMType.MULTIMODAL

    llm = LLM(language_model, tokenizer, processor, model_name, llm_type)
    return llm
