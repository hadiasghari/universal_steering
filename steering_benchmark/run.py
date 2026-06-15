"""
Steering Benchmark: Direction Computation

This script computes steering directions for various concepts (personalities, moods,
fears, places, personas) using different LLM models. Directions are saved to disk
for later use in steering generation.

Usage:
    python -m steering_benchmark.run --model_set phi --model_version 3-medium-4k-instruct
"""

import gpu_setup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import os
import torch
import numpy as np
from neural_controllers import NeuralController
from tqdm import tqdm
import gc
import utils
from steering_benchmark.model_loading import LLM, select_llm, resolve_model_args

SEED = 0
torch.manual_seed(SEED)
#torch.cuda.manual_seed(SEED)
np.random.seed(SEED)


def compute_save_directions(llm, dataset, concept, control_method='rfm'):
    """
    Compute and save steering directions for a concept.

    Args:
        llm: The language model
        dataset: Training data with inputs and labels
        concept: The concept name to steer towards
        control_method: Algorithm for direction computation ('rfm', 'linear', etc.)
    """
    controller = NeuralController(
        llm,
        llm.tokenizer,
        rfm_iters=8,
        control_method=control_method,
        n_components=1,
        batch_size=8,
    )
    controller.compute_directions(dataset[concept]['train']['inputs'], dataset[concept]['train']['labels'])
    controller.save(concept=concept, model_name=llm.name, path='directions/')


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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compute steering directions for concepts")
    parser.add_argument("--model_set", type=str, default='phi')
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--model_size", type=str, default=None)
    parser.add_argument("--concepts_to_steer", type=str, default='all')
    args = parser.parse_args()

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

    MODEL_SET = args.model_set
    MODEL_VERSION, MODEL_SIZE = resolve_model_args(MODEL_SET, args.model_version, args.model_size)

    llm = select_llm(MODEL_SET, MODEL_VERSION=MODEL_VERSION, MODEL_SIZE=MODEL_SIZE)
    METHOD = 'rfm'

    if args.concepts_to_steer == 'all':
        concepts_to_steer = ['fears', 'personas', 'places', 'personalities', 'moods']
    else:
        concepts_to_steer = [args.concepts_to_steer]
    number_of_concepts_to_steer = 120

    for concept_label in concepts_to_steer:
        fname = fnames[concept_label]
        concepts = read_file(fname, lower=lowers[concept_label])

        if number_of_concepts_to_steer < len(concepts):
            import random
            random.seed(0)
            subconcepts_to_steer = random.sample(concepts, number_of_concepts_to_steer)
        else:
            subconcepts_to_steer = concepts

        for concept in tqdm(subconcepts_to_steer):
            directions_file = f'directions/{METHOD}_{concept}_{llm.name}.pkl'
            if os.path.exists(directions_file):
                print(f"Skipping {concept} because directions file already exists")
                continue
            else:
                print(f"Computing directions to file: {directions_file}")

            print(f"===== CONCEPT={concept} =====")

            if concept_label == 'fears':
                dataset = utils.pca_fears_dataset(llm, concept)
            elif concept_label == 'personalities':
                dataset = utils.pca_personalities_dataset(llm, concept)
            elif concept_label == 'personas':
                dataset = utils.pca_persona_dataset(llm, concept)
            elif concept_label == 'moods':
                dataset = utils.pca_mood_dataset(llm, concept)
            elif concept_label == 'places':
                dataset = utils.pca_places_dataset(llm, concept)

            compute_save_directions(llm, dataset, concept, control_method=METHOD)
            del dataset
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    main()
