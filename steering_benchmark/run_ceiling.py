"""Prompted-ceiling control (Method 'Controls' bullet): can the model express each
concept when explicitly asked, with NO steering? One generation + one judgment per
concept. Establishes that steering failures are steering failures, not expression
failures.

Usage: python -m steering_benchmark.run_ceiling --model_set llama --model_size 8B
Writes csvs/ceiling_{model}.csv (concept, score) and prints per-supersense summary.
"""
import argparse
import csv
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import torch

CEILING_PROMPT = "Tell me about '{c}'. Answer in 20 words or less."

JUDGE_TEMPLATE = """You are evaluating whether a language model can express a target concept. You will be given:

1. A target concept (it may be of any kind: a feeling, a place, a profession, an object, an idea, etc.).
2. The model's response to the request: "Tell me about '<concept>'. Answer in 20 words or less."

Your task is to assess whether the response expresses the target concept.

Evaluation criteria:
1. The response counts as expressing the concept if it clearly centers on it, describes it, names it, OR manifestly embodies it.
2. Generic content that would fit almost any concept of that kind does NOT count.
3. Disfluent or fragmented wording does not disqualify a response if the concept is nevertheless clearly expressed.
4. A response that misunderstands the word (e.g., describes a different sense or a different thing entirely) scores 0.

Output format:
1. Score (0-1): 1 indicates the concept is expressed and 0 indicates it is not.
2. Explanation: A concise justification for the score.

Here is the concept: {concept} and here is the response: {response}"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_set", type=str, default='llama')
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument("--model_size", type=str, default=None)
    args = parser.parse_args()

    from steering_benchmark.model_loading import select_llm, resolve_model_args
    MV, MS = resolve_model_args(args.model_set, args.model_version, args.model_size)
    llm = select_llm(args.model_set, MODEL_VERSION=MV, MODEL_SIZE=MS)
    model, tokenizer = llm.language_model.eval(), llm.tokenizer

    df = pd.read_csv('bbxdata/bbx_wordnet_ds.csv')
    tag = f"{args.model_set}_{MV}_{MS}"
    gen_cache = f"cached_outputs/ceiling_generations_{tag}.pkl"

    # ---- generation (unsteered, greedy) ----
    gens = pickle.load(open(gen_cache, 'rb')) if os.path.exists(gen_cache) else {}
    for i, c in enumerate(df.concept):
        if c in gens:
            continue
        chat = [{"role": "user", "content": CEILING_PROMPT.format(c=c)}]
        ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True,
                                            return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        txt = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        gens[c] = txt
        if i % 25 == 0:
            print(f"gen {i}/600: {c} -> {txt[:60]}")
            pickle.dump(gens, open(gen_cache, 'wb'))
    pickle.dump(gens, open(gen_cache, 'wb'))
    print(f"generation done: {len(gens)}")

    # ---- judging (gpt-oss via ollama) ----
    import re
    from ollama import chat as ollama_chat
    judge_cache = f"cached_outputs/ceiling_judgments_{tag}.pkl"
    scores = pickle.load(open(judge_cache, 'rb')) if os.path.exists(judge_cache) else {}
    for i, c in enumerate(df.concept):
        if c in scores:
            continue
        prompt = JUDGE_TEMPLATE.format(concept=c, response=gens[c] or "None")
        out = ollama_chat(model='gpt-oss', messages=[{"role": "user", "content": prompt}],
                          think='low', options={'temperature': 0., 'num_predict': 500})
        m = re.search(r"Score:\s*\**\s*([01](?:\.\d+)?)", out.message.content or "")
        scores[c] = float(m.group(1)) if m else 0.0
        if i % 50 == 0:
            print(f"judge {i}/600: {c} = {scores[c]}")
            pickle.dump(scores, open(judge_cache, 'wb'))
    pickle.dump(scores, open(judge_cache, 'wb'))

    # ---- outputs ----
    os.makedirs('csvs', exist_ok=True)
    out_csv = f"csvs/ceiling_{tag}.csv"
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['concept', 'ceiling'])
        w.writerows(sorted(scores.items()))
    n = sum(1 for v in scores.values() if v >= 0.5)
    print(f"\nCEILING {tag}: {n}/600 expressible")
    df['ok'] = [scores[c] >= 0.5 for c in df.concept]
    print(df.groupby('supersense').ok.sum().sort_values().to_string())
    fails = [c for c in df.concept if scores[c] < 0.5]
    print("\nfailing concepts:", fails)


if __name__ == '__main__':
    main()
