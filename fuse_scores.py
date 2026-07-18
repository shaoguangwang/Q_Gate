"""Q-Gate | Stage 2: Query-Modulated Gating and Score Fusion.

Reads the three raw expert-stream scores (score_logic, score_relevance,
score_narrative) from an input JSON, normalizes each stream with a unified
Min-Max + masked temperature Softmax pipeline, asks the query-aware gating
module for a weight triple [w_logic, w_relevance, w_narrative], and writes the
fused score_fusion distribution back to a new JSON.

Stream mapping (code name = component = expert model -> gating weight):
    score_logic     = Visual Grounding    = YOLO-World                 -> w_logic
    score_relevance = Global Matching      = BLIP (image-text matching) -> w_relevance
    score_narrative = Contextual Alignment = Sentence-BERT              -> w_narrative

Gating strategies:
    llm       Query-Modulated Gating (default; GPT-4o or any local backend).
    equal     Uniform 1/3 weights (static-fusion baseline).

Usage (see README for the full pipeline):
    export OPENAI_API_KEY="..."          # required for --strategy llm
    export OPENAI_BASE_URL="..."         # optional; defaults to the OpenAI API
    python fuse_scores.py \
        --input_json <narrative.json> \
        --output_dir <out_dir> \
        --strategy llm --backend gpt4 --temperature 0.5
"""

import json
import os
import argparse
from tqdm import tqdm
import numpy as np
import ast
import re
from typing import Dict

# Import the LLM interface lazily so the offline 'equal' strategy works without a backend.
try:
    from qgate.interface_llm import QGateGrounder
except ImportError:
    QGateGrounder = None

# ==========================================
# 1. Normalization core
# ==========================================

def min_max_scale(scores: np.ndarray) -> np.ndarray:
    """Linearly map scores to [0, 1] to remove scale discrepancies between the
    three streams (YOLO-World vs. BLIP vs. Sentence-BERT output different ranges)."""
    if len(scores) == 0:
        return scores
    min_val = np.min(scores)
    max_val = np.max(scores)
    if max_val - min_val == 0:
        # All scores identical -> map to 0 to avoid injecting noise.
        return np.zeros_like(scores)
    return (scores - min_val) / (max_val - min_val)

def masked_temperature_softmax(scores: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    """Masked temperature Softmax.
    1. Mask: positions with a (near-)zero raw score keep zero probability. This
       strictly preserves the sparsity of streams such as narrative, whose score
       is exactly zero for frames without subtitles.
    2. Temperature: T < 1 sharpens the distribution, T > 1 smooths it.
    """
    if len(scores) == 0:
        return scores

    # Only strictly positive entries receive probability mass. Relevance cosine
    # values can be negative, but Min-Max scaling maps them to >= 0 beforehand.
    non_zero_indices = scores > 0.00001

    if not np.any(non_zero_indices):
        return np.zeros_like(scores)

    valid_scores = scores[non_zero_indices]

    # Guard against a non-positive temperature (would divide by zero -> NaN).
    temperature = max(float(temperature), 1e-6)

    # Subtract max for numerical stability before exponentiating.
    exp_scores = np.exp((valid_scores - np.max(valid_scores)) / temperature)
    softmax_probs = exp_scores / np.sum(exp_scores)

    final_scores = np.zeros_like(scores)
    final_scores[non_zero_indices] = softmax_probs

    return final_scores

def preprocess_score_stream(raw_scores: list, temperature: float) -> np.ndarray:
    """Standard preprocessing pipeline: list -> numpy -> Min-Max -> masked Softmax."""
    scores_np = np.array(raw_scores, dtype=np.float32)
    if len(scores_np) == 0:
        return scores_np

    # 1. Align ranges to [0, 1].
    scaled_scores = min_max_scale(scores_np)

    # 2. Reshape the distribution (sharpen). Lower temperature -> sharper peaks.
    processed_scores = masked_temperature_softmax(scaled_scores, temperature)

    return processed_scores

# ==========================================
# 2. Weight strategies
# ==========================================

# --- LLM-based Query-Modulated Gating (default backend: GPT-4o) ---
def get_llm_weights(question: str, options: str, grounder) -> Dict[str, float]:
    """Prompt an LLM to route attention weights across the three expert streams.

    The candidate options are appended to the query (the options-aware setting).
    """
    if grounder is None:
        raise ValueError("LLM Backend not initialized. Check your imports.")

    prompt_template = """You are an expert Strategy Planner for a Video QA system. Your task is to assign optimal weights [w_logic, w_relevance, w_narrative] to three retrieval experts based on the user's query.

    User Query: "{question_text}"
    Candidate Options: "{options_text}\"""" + """

    ### 1. The Experts & Their Origins:
       **w_relevance (The Vision-Language Model / BLIP):**
        *   Mechanism: Computes global similarity between the query and the whole image frame.
        *   Role: The MOST reliable expert for overall scene understanding.
        *   Strength: Best for general scene understanding, atmosphere, and "fuzzy" matching. **This is your DEFAULT reliable expert.**
       **w_logic (The Object Detector / YOLO):**
        *   Mechanism: Use YOLO to detect specific objects (e.g., "3 cars", "red tie") and their spatial relations.
        *   Strength: Crucial ONLY when the question asks for **visual details** like counting, specific colors of small objects, or spatial positions (left/right).
       **w_narrative (The Subtitle Searcher / SBERT):**
        *   Mechanism: Matches query text with the video's subtitles/transcript.
        *   Strength: Essential for **audio/textual** cues.

    ### 2. Strategy Rules (CRITICAL):

    **Rule A: When to BOOST `w_narrative` (Set >= 0.5)**
       **Explicit Quote**: Query contains "subtitle says", "caption mentions", "phrase '...'", "after mentioning".
       **Audio/Speech**: Query asks "What did he say?", "What is the song?".
       **Cross-Modal Locating**: Query asks "When [text phrase] appears, what happens visually?" OR "What subtitle appears when [visual event] happens?".
       **Abstract/Reasoning**: Query asks "Why", "Reason", "Main idea", "History", or "According to the video" (referring to spoken content).

    **Rule B: When to BOOST `w_logic` (Set >= 0.4)**
       **Visual Detail**: Query asks "How many...", "What color is the...", "What is holding...", "Who is standing to the left".
       *Note*: If Rule A and B conflict (e.g., "What color is the shirt when he says 'Hello'"), prioritize **Narrative** to locate the time first, then Logic.

    **Rule C: The Default State (`w_relevance`)**
       For all other general visual questions (e.g., "What is the man doing?", "Describe the scene"), **`w_relevance` should be the highest (>= 0.5)** to ensure stability.

    ### 3. Output Format:
       Return ONLY a JSON object. No reasoning text.
       Weights must sum to 1.0.
       STRICTLY use double quotes ("") for keys, like {{"w_logic": 0.2}}. Do not use single quotes.

    Example 1 (Subtitle Anchor): "When the subtitle mentions 'freedom', what is she doing?"
    -> {{"w_logic": 0.2, "w_relevance": 0.1, "w_narrative": 0.7}}

    Example 2 (Visual Detail): "How many red apples are on the table?"
    -> {{"w_logic": 0.6, "w_relevance": 0.3, "w_narrative": 0.1}}

    Example 3 (General Action): "What is the boy in the blue shirt doing?"
    -> {{"w_logic": 0.2, "w_relevance": 0.7, "w_narrative": 0.1}}

    Your JSON Response:
    """
    prompt = prompt_template.format(question_text=question, options_text=options)

    response_str = ""
    try:
        response_str = grounder.VLM_model_interface.inference_text_only(
            query=prompt, temperature=0.0, max_tokens=100
        )

        # 1. Extract the first {...} block, guarding against surrounding prose or
        #    Markdown fences (e.g. ```json).
        match = re.search(r'\{.*?\}', response_str, re.DOTALL)
        raw_json_str = match.group(0) if match else response_str

        # 2. Repair a common open-model quirk: single quotes instead of double.
        cleaned_json_str = raw_json_str.replace("'", '"')

        # 3. Parse.
        weights = None
        try:
            weights = json.loads(cleaned_json_str)
        except json.JSONDecodeError:
            try:
                # Fallback: parse a Python-dict-formatted string via ast.
                weights = ast.literal_eval(raw_json_str)
            except Exception:
                pass

        # 4. Validate and normalize (guard against float drift). Require all three
        #    keys so a partial LLM response can't KeyError downstream.
        required = {"w_logic", "w_relevance", "w_narrative"}
        if isinstance(weights, dict) and required <= weights.keys():
            triple = {k: float(weights[k]) for k in required}
            total = sum(triple.values())
            if total > 0:
                return {k: v / total for k, v in triple.items()}
        else:
            raise ValueError("Parsed result is not a valid dictionary or missing keys.")

    except Exception as e:
        # Surface the raw model output for debugging, then fall back to equal weights.
        print(f"\nLLM Error: {e} | Raw Response: {response_str}\n")

    return {"w_logic": 0.333, "w_relevance": 0.333, "w_narrative": 0.334}

# ==========================================
# 3. Main pipeline
# ==========================================

def fuse_scores(input_json: str, output_dir: str, strategy: str, backend: str = 'gpt4',
                temperature: float = 0.5):
    """Unified fusion entry point.

    strategy:    'llm' | 'equal'
    temperature: masked-Softmax sharpness (lower is sharper; paper uses 0.5).
    """
    grounder = None
    if strategy == 'llm':
        if QGateGrounder is None:
            raise ImportError(
                "The 'llm' gating strategy needs the qgate LLM backend, whose imports "
                "failed. Install the full dependencies first: pip install -r requirements.txt "
                "(the offline demo uses --strategy equal instead)."
            )
        print(f"Initializing LLM Backend: {backend}...")
        grounder = QGateGrounder(backend=backend)

    print(f"Loading data from: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Sanity check: the grounding producer writes the raw grounding stream as
    # 'score_list', but fusion reads 'score_logic'. If the input carries 'score_list'
    # without 'score_logic', the merge step was skipped and every record would fuse
    # to an empty distribution — warn loudly instead of degrading silently.
    if data and not any('score_logic' in e for e in data) and any('score_list' in e for e in data):
        print("WARNING: input has 'score_list' but no 'score_logic'. Copy each record's "
              "score_list into a 'score_logic' field first (see README, Score streams).")

    updated_data = []

    for entry in tqdm(data, desc=f"Fusing Scores ({strategy})"):
        # 1. Fetch raw scores.
        s_logic_raw = entry.get('score_logic', [])
        s_relev_raw = entry.get('score_relevance', [])
        s_narra_raw = entry.get('score_narrative', [])

        # Length / alignment check.
        n = len(s_logic_raw)
        if n == 0 or len(s_relev_raw) != n or len(s_narra_raw) != n:
            # Missing or misaligned data -> default and skip.
            entry['score_fusion'] = []
            entry.update({"w_logic": 0, "w_relevance": 0, "w_narrative": 0})
            updated_data.append(entry)
            continue

        # 2. Preprocess: Min-Max -> masked temperature Softmax.
        s_logic_norm = preprocess_score_stream(s_logic_raw, temperature)
        s_relev_norm = preprocess_score_stream(s_relev_raw, temperature)
        s_narra_norm = preprocess_score_stream(s_narra_raw, temperature)

        # 3. Compute weights.
        weights = {"w_logic": 0.333, "w_relevance": 0.333, "w_narrative": 0.334}

        if strategy == 'equal':
            pass  # Keep uniform weights.
        elif strategy == 'llm':
            options_str = "\n".join(entry.get('options', [])) if isinstance(entry.get('options'), list) else entry.get('options', "")
            weights = get_llm_weights(entry.get('question', ''), options_str, grounder)

        # If a raw stream is entirely zero (e.g. a subtitle-less video -> narrative
        # is all-zero), zero that stream's weight and renormalize the rest to sum 1,
        # so gating weight is not wasted on a dead stream. When all three streams are
        # active this branch is skipped and the weighting is unchanged.
        active = {'w_logic': s_logic_norm.sum() > 0, 'w_relevance': s_relev_norm.sum() > 0, 'w_narrative': s_narra_norm.sum() > 0}
        if not all(active.values()):
            for _k in list(weights.keys()):
                if not active.get(_k, True):
                    weights[_k] = 0.0
            _wsum = sum(weights.values())
            if _wsum > 0:
                weights = {_k: _v / _wsum for _k, _v in weights.items()}
            else:
                # The gater put all weight on dead stream(s). Fall back to a uniform
                # split over the streams that do carry signal, so an extreme gating
                # decision never collapses the fusion to an all-zero distribution.
                n_active = sum(active.values())
                if n_active:
                    weights = {_k: (1.0 / n_active if active.get(_k) else 0.0) for _k in weights}

        # 4. Weighted fusion.
        score_fusion = (weights['w_logic'] * s_logic_norm +
                        weights['w_relevance'] * s_relev_norm +
                        weights['w_narrative'] * s_narra_norm)

        # Final normalization (guard against float drift).
        if score_fusion.sum() > 0:
            score_fusion /= score_fusion.sum()

        # 5. Store results.
        entry['score_fusion'] = score_fusion.tolist()
        entry['w_logic'] = weights['w_logic']
        entry['w_relevance'] = weights['w_relevance']
        entry['w_narrative'] = weights['w_narrative']

        updated_data.append(entry)

    # Save.
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_json))[0]
    out_name = f"{base_name}_fusion_{strategy}_T{temperature}.json"
    out_path = os.path.join(output_dir, out_name)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, indent=4, ensure_ascii=False)
    print(f"Done! Saved to: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q-Gate | Query-modulated gating and score fusion.")
    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--strategy", type=str, required=True, choices=['llm', 'equal'])
    parser.add_argument("--backend", type=str, default='gpt4', help="LLM backend if strategy is llm")
    parser.add_argument("--temperature", type=float, default=0.5, help="Softmax temperature for score sharpening (lower is sharper)")

    args = parser.parse_args()

    fuse_scores(
        input_json=args.input_json,
        output_dir=args.output_dir,
        strategy=args.strategy,
        backend=args.backend,
        temperature=args.temperature,
    )
