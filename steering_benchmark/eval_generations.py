"""
Steering Benchmark: Generation Evaluation

This script generates steered outputs using precomputed directions and various
steering coefficients. Results are cached to disk for subsequent evaluation.

Usage:
    python -m steering_benchmark.eval_generations --model_set phi --model_version 3-medium-4k-instruct
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from neural_controllers import NeuralController
from tqdm import tqdm
import pickle
import gc
import os
from gpu_setup import device, empty_cache
from steering_benchmark.model_loading import LLM, select_llm, resolve_model_args

SEED = 0
torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def generate(concept, llm, prompt, image=None, coefs=[0.4], control_method='rfm', max_tokens=100, gen_orig=True):
    """
    Generate steered outputs for a concept with various steering coefficients.

    Args:
        concept: The concept to steer towards
        llm: The language model
        prompt: Input prompt for generation
        image: Optional image input (for multimodal models)
        coefs: List of steering coefficients to try
        control_method: Steering algorithm
        max_tokens: Maximum tokens to generate
        gen_orig: Whether to also generate unsteered output

    Returns:
        List of (coefficient, output) tuples
    """
    controller = NeuralController(
        llm,
        llm.tokenizer,
        rfm_iters=8,
        control_method=control_method,
        n_components=1
    )

    controller.load(concept=concept, model_name=llm.name, path='directions/')

    if gen_orig:
        original_output = controller.generate(prompt, image=image, max_new_tokens=max_tokens, do_sample=False)
        print(original_output)

    outputs = []
    target_keys = set(range(-1, -80, -1))
    print("Steering layers: ", controller.hidden_layers & target_keys)

    for coef in coefs:
        print(f"Coeff: {coef} ==========================================================")
        steered_output = controller.generate(
            prompt,
            image=image,
            layers_to_control=controller.hidden_layers & target_keys,
            control_coef=coef,
            max_new_tokens=max_tokens,
            do_sample=False
        )
        outputs.append((coef, steered_output))
        print(steered_output)
    return outputs


def read_file(fname, lower=True):
    """Read concepts from a text file, one per line."""
    concepts = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            if lower:
                concepts.append(line.strip().lower())
            else:
                concepts.append(line.strip())
    concepts = sorted(list(set(concepts)))
    return concepts


# Prompts for each concept category and version
PROMPTS = {
    'fears': {
        1: "What is the scariest thing in the world? Answer in 20 words or less.",
        2: "Is there anything you secretly hope never happens to you? Answer in 20 words or less.",
        3: "What would a movie villain based on your worst fear be like? Answer in 20 words or less.",
        4: "If a fear had to visit you in your sleep, what form would it take? Answer in 20 words or less.",
        5: "What would a fear-themed amusement park ride, made just for you, be like? Answer in 20 words or less."
    },
    'personalities': {
        1: "What is your favorite subject? Answer in 20 words or less.",
        2: "What's the kind of problem that makes you forget to eat or sleep? Answer in 20 words or less.",
        3: "If you had unlimited resources, what would you work on? Answer in 20 words or less.",
        4: "Describe your work in a way a 5-year-old can understand. Answer in 20 words or less.",
        5: "What's a question that got you hooked? Answer in 20 words or less."
    },
    'personas': {
        1: "Tell me about your work. Answer in 20 words or less.",
        2: "What's something you've always been drawn to? Answer in 20 words or less.",
        3: "If someone gave you the keys to everything - money, power, influence - what's the first thing you'd do? Answer in 20 words or less.",
        4: "Describe your work in a way a 5-year-old can understand. Answer in 20 words or less.",
        5: "If you could teach everyone one thing, what would it be? Answer in 20 words or less."
    },
    'moods': {
        1: "How are you feeling today? Answer in 20 words or less.",
        2: "What's one sentence that would sum up your current vibe? Answer in 20 words or less.",
        3: "Tell me about your day. Answer in 20 words or less.",
        4: "What would a song that describes your current mood be called? Answer in 20 words or less.",
        5: "Where would you go if you could go anywhere right now? Answer in 20 words or less."
    },
    'places': {
        1: "What is your favorite place? Answer in 20 words or less.",
        2: "If you could instantly master any skill, what would it be? Answer in 20 words or less.",
        3: "If you could build a shrine to one place, what would you put inside it? Answer in 20 words or less.",
        4: "If someone filmed a movie about your life, where would the opening scene be set? Answer in 20 words or less.",
        5: "Where were you the last time you felt completely at peace? Answer in 20 words or less."
    }
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate steered outputs for concepts")
    parser.add_argument("--model_set", type=str, default='phi')
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--concepts_to_steer", type=str, default='all')
    args = parser.parse_args()

    MODEL_TYPE = args.model_set
    MODEL_VERSION, MODEL_SIZE = resolve_model_args(MODEL_TYPE, args.model_version, args.model_size)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    fnames = {
        'fears': 'data/fears/fears.txt',
        'personalities': 'data/personalities/personalities.txt',
        'moods': 'data/moods/moods.txt',
        'places': 'data/places/places.txt',
        'personas': 'data/personas/personas.txt'
    }
    lowers = {
        'fears': True,
        'personalities': True,
        'moods': True,
        'places': False,
        'personas': False
    }

    # Steering coefficients per model
    METHOD = 'rfm'
    if MODEL_TYPE == 'mistral':
        if MODEL_VERSION == 'Small-Instruct-2409':
            COEFS = [0.08, .09, .1, .11, .12, .13, .14, .15]
        elif MODEL_VERSION == 'Large-Instruct-2407-4bit':
            COEFS = [.075, .1, .125, .15, .175, .2, .225, .25]
    elif MODEL_TYPE == 'falcon':
        if MODEL_VERSION == '3':
            if MODEL_SIZE == '10B':
                COEFS = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
            elif MODEL_SIZE == '3B':
                COEFS = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]
    elif MODEL_TYPE == 'phi':
        if MODEL_VERSION == '3-medium-4k-instruct':
            COEFS = [2.0, 2.25, 2.5, 2.75, 3.0]
        elif MODEL_VERSION == '4':
            COEFS = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    elif MODEL_TYPE == 'llama':
        if MODEL_SIZE == '8B':
            COEFS = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        else:
            COEFS = [0.4, 0.41, 0.42, 0.43, 0.44, 0.45]
    else:
        raise ValueError(f"Model type {MODEL_TYPE} not supported")

    llm = select_llm(MODEL_TYPE, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)

    PROMPT_VERSIONS = [1, 2, 3, 4, 5]
    number_of_concepts_to_steer = 120

    for VERSION in PROMPT_VERSIONS:
        VERSION_LABEL = f'_v{VERSION}' if VERSION >= 2 else ''

        if args.concepts_to_steer == 'all':
            concepts_to_steer = ['personalities', 'moods', 'places', 'personas', 'fears']
        else:
            concepts_to_steer = [args.concepts_to_steer]

        for concept_label in concepts_to_steer:
            fname = fnames[concept_label]
            concepts = read_file(fname, lower=lowers[concept_label])

            if number_of_concepts_to_steer < len(concepts):
                import random
                random.seed(0)
                subconcepts_to_steer = random.sample(concepts, number_of_concepts_to_steer)
            else:
                subconcepts_to_steer = concepts

            os.makedirs('cached_outputs', exist_ok=True)
            out_file = f"cached_outputs/{METHOD}_{concept_label}_steered_500_concepts_{MODEL_TYPE}_{MODEL_VERSION}_{MODEL_SIZE}_english_only{VERSION_LABEL}.pkl"

            # Load existing cache if present
            if os.path.exists(out_file):
                try:
                    loaded = pickle.load(open(out_file, "rb"))
                except Exception:
                    loaded = None
                if isinstance(loaded, dict):
                    all_outputs = loaded
                elif isinstance(loaded, list):
                    all_outputs = {k: v for (k, v) in loaded}
                else:
                    all_outputs = {}
            else:
                all_outputs = {}
            print(f"Writing/merging outputs to file: {out_file}")

            for concept in tqdm(subconcepts_to_steer):
                print(f"===== CONCEPT={concept} =====")

                if concept in all_outputs:
                    print(f"Skipping concept already generated: {concept}")
                    continue

                prompt = PROMPTS[concept_label][VERSION]
                outputs = generate(concept, llm, prompt, image=None, coefs=COEFS,
                                   control_method=METHOD, max_tokens=50, gen_orig=False)
                all_outputs[concept] = outputs
                empty_cache()

                # Cache incrementally for robustness
                try:
                    with open(out_file, "wb") as file:
                        pickle.dump(all_outputs, file)
                        file.flush()
                        os.fsync(file.fileno())
                except Exception as e:
                    print(f"Warning: failed to write cache for concept '{concept}': {e}")

            with open(out_file, "wb") as file:
                pickle.dump(all_outputs, file)
            gc.collect()


if __name__ == "__main__":
    main()
