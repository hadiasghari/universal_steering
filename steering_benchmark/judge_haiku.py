"""
Claude-Haiku judge for steered generations (HA).

Mirrors parse_results.py's judging flow (same eval prompt templates, same
response parsing, same Score-regex and max-over-coefs aggregation) but uses
the Anthropic API instead of local gpt-oss/gpt4. Usable standalone on any
cached_outputs directory:

    python -m steering_benchmark.judge_haiku --dir cached_outputs4d \
        --classes personalities,moods,places --versions 1,4

Results: prints per-class "Steered: X out of N" + totals, and writes
{dir}/haiku_judgments{_vV}.json with per-concept, per-coef scores.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse, json, os, pickle, re, time
from concurrent.futures import ThreadPoolExecutor

import anthropic
from dotenv import load_dotenv

from steering_benchmark.parse_results import load_prompt, parse_response

JUDGE_MODEL = "claude-haiku-4-5-20251001"
MODEL_NAME_DEFAULT = "llama_3.1_8B"   # filename fragment
PARSE_MODEL = "llama"                 # key into parse_results.ASSISTANT_TAGS


def judge_one(client, prompt, retries=5):
    if "gpt-oss" in JUDGE_MODEL:  # local ollama backend
        from ollama import chat
        for attempt in range(retries):
            try:
                r = chat(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}])
                content = r["message"]["content"]
                m = re.search(r"Score:\s*\**\s*([01](?:\.\d+)?)", content)
                return (int(float(m.group(1)) >= 0.5) if m else 0), content
            except Exception as e:
                print(f"ollama error (attempt {attempt}): {e}"); time.sleep(2 ** attempt)
        return 0, "JUDGE_FAILED"
    # Claude 5 family: temperature deprecated, thinking blocks precede text -> bigger budget
    is_c5 = "sonnet-5" in JUDGE_MODEL or "fable" in JUDGE_MODEL
    kw = {"max_tokens": 1500} if is_c5 else {"max_tokens": 200, "temperature": 0}
    for attempt in range(retries):
        try:
            msg = client.messages.create(
                model=JUDGE_MODEL, **kw,
                messages=[{"role": "user", "content": prompt}])
            content = next((b.text for b in msg.content if getattr(b, "type", "") == "text"), "")
            # judges occasionally emit graded scores ("Score: 0.7"); parse as float, threshold at 0.5
            m = re.search(r"Score:\s*\**\s*([01](?:\.\d+)?)", content)
            return (int(float(m.group(1)) >= 0.5) if m else 0), content
        except anthropic.RateLimitError:
            time.sleep(2 ** attempt)
        except anthropic.APIError as e:
            print(f"API error (attempt {attempt}): {e}")
            time.sleep(2 ** attempt)
    return 0, "JUDGE_FAILED"


def main():
    global JUDGE_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--classes", default="personalities,moods,places")
    ap.add_argument("--versions", default="1,4")
    ap.add_argument("--model_name", default=MODEL_NAME_DEFAULT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--judge_model", default=JUDGE_MODEL)
    args = ap.parse_args()
    JUDGE_MODEL = args.judge_model
    judge_slug = "haiku" if "haiku" in JUDGE_MODEL else JUDGE_MODEL.split("-")[1]

    load_dotenv(Path(__file__).parent.parent / ".env")
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY from env

    for version in [int(v) for v in args.versions.split(",")]:
        vlabel = "" if version == 1 else f"_v{version}"
        all_judg, grand_steered, grand_n = {}, 0, 0
        for label in args.classes.split(","):
            fp = f"{args.dir}/rfm_{label}_steered_500_concepts_{args.model_name}_english_only{vlabel}.pkl"
            if not os.path.exists(fp):
                print(f"missing: {fp}"); continue
            results = pickle.load(open(fp, "rb"))
            template = load_prompt(label, version)

            tasks = []   # (concept, coef, judge_prompt)
            for concept, responses in results.items():
                for resp in responses:
                    parsed = parse_response(resp, PARSE_MODEL) or "None"
                    tasks.append((concept, resp[0],
                                  template.format(personality=concept, parsed_response=parsed)))

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                scores = list(ex.map(lambda t: judge_one(client, t[2])[0], tasks))

            per_concept = {}
            judgments = {}
            for (concept, coef, _), s in zip(tasks, scores):
                per_concept[concept] = max(per_concept.get(concept, 0), s)
                judgments.setdefault(concept, {})[str(coef)] = s
            steered = sum(per_concept.values())
            print(f"[v{version}] {label}: Steered {steered} out of {len(per_concept)}")
            all_judg[label] = judgments
            grand_steered += steered; grand_n += len(per_concept)

        print(f"[v{version}] TOTAL: {grand_steered}/{grand_n} "
              f"({100*grand_steered/max(grand_n,1):.1f}%)  judge={JUDGE_MODEL}")
        out = f"{args.dir}/{judge_slug}_judgments{vlabel}.json"
        json.dump(all_judg, open(out, "w"), indent=1)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
