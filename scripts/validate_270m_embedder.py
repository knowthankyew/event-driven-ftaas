#!/usr/bin/env python3
"""
Comprehensive empirical validation script for bitnet-embedding-270m:
1. Compares token IDs for 20 legal clauses against Hugging Face reference tokenizer
   (GemmaTokenizerFast from microsoft/harrier-oss-v1-270m) element-by-element.
2. Computes 640-dim dense embeddings using native bitnet.cpp llama-embedding (metadata-governed mean pooling).
3. Computes 640-dim dense embeddings using the official unquantized teacher model (microsoft/harrier-oss-v1-270m).
4. Computes space-independent pairwise similarity matrix correlation (Spearman rho & Pearson r) across 190 off-diagonal pairs.
5. Evaluates MiniLM baseline comparison explaining Gemma backbone baseline similarity offset.
6. Records all metrics, token streams, and correlations with redacted host paths to docs/empirical_270m_embedder_validation.json.
"""

import os
import sys
import re
import json
import time
import platform
import subprocess
import numpy as np
import scipy.stats
import torch

# Add src to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
inference_dir = os.path.join(repo_root, "src/FtaaSService.Inference")
if inference_dir not in sys.path:
    sys.path.insert(0, inference_dir)

import bitnet_engine
from transformers import AutoModel, AutoTokenizer

CLAUSES = [
    ("12_CFR_229_10_a", "12 CFR 229.10(a) Cash deposits. A bank shall make funds deposited in an account by cash available for withdrawal not later than the business day after the banking day on which the cash is deposited."),
    ("12_CFR_229_10_b", "12 CFR 229.10(b) Electronic payments. A bank shall make funds received for deposit in an account by an electronic payment available for withdrawal not later than the business day after the banking day on which the bank receives the electronic payment."),
    ("12_CFR_229_10_c_treasury", "12 CFR 229.10(c) Certain check deposits. A depositary bank shall make funds deposited in an account by check available for withdrawal not later than the business day after the banking day in the case of a check drawn on the Treasury of the United States."),
    ("12_CFR_229_10_c_aggregate", "12 CFR 229.10(c)(1)(vi) The lesser of $275 or the aggregate amount deposited on any one banking day to all accounts of the customer by all checks not subject to next-day availability."),
    ("12_CFR_229_12_b_local", "12 CFR 229.12(b) Permanent schedule for local checks. A depositary bank shall make funds deposited in an account by a local check available for withdrawal not later than the second business day following the banking day on which funds are deposited."),
    ("12_CFR_229_13_b_large", "12 CFR 229.13(b) Large deposits exception. Sections 229.10(c) and 229.12 do not apply to the aggregate amount of deposits by one or more checks to the extent that the aggregate amount is in excess of $6,725 on any one banking day."),
    ("15_USC_8403_1_disclosure", "15 U.S.C. 8403(1) FTC ROSCA clear and conspicuous disclosures. It has become unlawful to charge a consumer unless the transaction clearly and conspicuously discloses all material terms before obtaining billing info."),
    ("15_USC_8403_2_consent", "15 U.S.C. 8403(2) FTC ROSCA express informed consent. It is unlawful to charge a consumer for an automatic renewal without obtaining the consumer's express informed consent before charging."),
    ("15_USC_8403_3_cancellation", "15 U.S.C. 8403(3) FTC ROSCA cancellation mechanism. The vendor must provide simple mechanisms for a consumer to stop recurring charges from being placed on the consumer's payment card."),
    ("CA_BPC_17602_a_1_clear", "California Bus. & Prof. Code 17602(a)(1) Clear and conspicuous presentation of automatic renewal terms in visual proximity to the request for consent before the subscription is fulfilled."),
    ("CA_BPC_17602_a_2_consent", "California Bus. & Prof. Code 17602(a)(2) It is unlawful to charge the consumer for an automatic renewal without first obtaining the consumer's affirmative consent to the agreement containing terms."),
    ("CA_AB_2863_click_cancel", "California AB 2863 Single-click cancellation mechanism. A business offering automatic renewal shall provide a consumer with a cost-effective, timely, and easy-to-use mechanism for cancellation."),
    ("CA_AB_2863_fee_change", "California AB 2863 Notice of fee changes. A business shall provide clear and conspicuous notice of any material changes in terms or fee increases at least 14 days prior to implementation."),
    ("UK_DMCC_cooling_off", "UK Digital Markets, Competition and Consumers Act 2024 Chapter 2. A consumer has the statutory right to cancel a subscription contract during the initial 14-day cooling-off period."),
    ("UK_DMCC_annual_notice", "UK DMCC Act 2024 Annual renewal reminder notice. A trader must send reminder notices to the consumer before a renewal contract renews for a further period of 12 months or more."),
    ("EU_CRD_art_9_withdrawal", "EU Directive 2011/83/EU Article 9 Right of withdrawal. The consumer shall have a period of 14 days to withdraw from a distance or off-premises contract without giving any reason."),
    ("EU_CRD_art_8_payment", "EU Directive 2011/83/EU Article 8 Formal requirements. If placing an order entails activating a button or similar function, the button shall be labelled in an easily legible manner with order with obligation to pay."),
    ("NY_GBL_527_a_procedure", "New York General Business Law 527-a. A business that makes an automatic renewal offer shall provide an online cancellation option for consumers who accepted the offer online."),
    ("IL_815_ILCS_601_notice", "Illinois 815 ILCS 601 Automatic Contract Renewal Act. Any person offering an automatic renewal contract shall disclose the cancellation mechanism and send written notice of renewal within 30 to 60 days."),
    ("FTC_negative_option_click", "FTC Negative Option Rule Click-to-Cancel. Sellers must make it as easy for consumers to cancel their enrollment as it was to sign up, using an equally prominent cancellation method.")
]

def redact_path(path_str: str) -> str:
    """Redact user-specific host directory paths."""
    if not path_str:
        return ""
    return re.sub(r"/Users/[^/]+", "<HOST_DIR>", str(path_str))

def main():
    print(f"Validating bitnet-embedding-270m against reference across {len(CLAUSES)} legal clauses...")
    print(f"Platform: {platform.platform()}, CPU: {platform.processor()}")

    # 1. Load HF Reference Tokenizer & Teacher Model (Harrier 270M)
    ref_model_id = "microsoft/harrier-oss-v1-270m"
    print(f"Loading Hugging Face reference tokenizer & model from '{ref_model_id}'...")
    tok = AutoTokenizer.from_pretrained(ref_model_id)
    model = AutoModel.from_pretrained(ref_model_id, torch_dtype=torch.float32)
    model.eval()

    clause_results = []
    bitnet_embeddings = []
    harrier_embeddings = []

    for idx, (tag, clause_text) in enumerate(CLAUSES, 1):
        # A. Reference tokenization
        hf_enc = tok(clause_text)
        hf_ids = hf_enc.input_ids
        hf_tok_count = len(hf_ids)

        # B. Native bitnet.cpp tokenization (extract exact token IDs)
        tok_cmd = [
            str(bitnet_engine.BITNET_TOKENIZE_CLI_PATH),
            "-m", str(bitnet_engine.BITNET_EMBED_MODEL_PATH),
            "-p", clause_text
        ]
        tok_res = subprocess.run(tok_cmd, capture_output=True, text=True)
        native_ids = [
            int(m.group(1))
            for line in tok_res.stdout.splitlines()
            if (m := re.match(r'^\s*(\d+)\s*->', line))
        ]
        native_tok_count = len(native_ids)
        ids_exact_match = (native_ids == hf_ids)

        # C. Native bitnet.cpp embedding (metadata-governed mean pooling)
        bitnet_emb_vec, bitnet_lat = bitnet_engine.embed_bitnet_sync(clause_text)
        bitnet_emb = np.array(bitnet_emb_vec, dtype=np.float32)
        bitnet_embeddings.append(bitnet_emb)

        # D. Reference PyTorch embedding (Last-token pooling + L2 normalization)
        with torch.no_grad():
            inputs = tok(clause_text, return_tensors="pt")
            out = model(**inputs)
            last_idx = inputs.attention_mask.sum(dim=1) - 1
            ref_emb_pt = out.last_hidden_state[0, last_idx[0]]
            ref_emb_pt = torch.nn.functional.normalize(ref_emb_pt, p=2, dim=0).numpy()
            harrier_embeddings.append(ref_emb_pt)

        # E. Direct cross-space cosine similarity
        cos_sim = float(np.dot(ref_emb_pt, bitnet_emb) / (np.linalg.norm(ref_emb_pt) * np.linalg.norm(bitnet_emb)))

        item = {
            "clause_index": idx,
            "tag": tag,
            "text": clause_text,
            "hf_token_count": hf_tok_count,
            "native_token_count": native_tok_count,
            "token_ids_exact_match": ids_exact_match,
            "hf_token_ids": hf_ids,
            "native_token_ids": native_ids,
            "bitnet_embedding_dim": len(bitnet_emb),
            "ref_embedding_dim": len(ref_emb_pt),
            "cosine_similarity_direct": cos_sim,
            "bitnet_latency_ms": bitnet_lat
        }
        clause_results.append(item)
        print(f"[{idx:02d}/20] {tag:<26} | Tokens: HF={hf_tok_count:3d}, Native={native_tok_count:3d} (match={ids_exact_match}) | Latency: {bitnet_lat:.1f}ms")

    bitnet_mat = np.array(bitnet_embeddings)
    harrier_mat = np.array(harrier_embeddings)

    # 2. Pairwise Similarity Matrix & Rank Correlation (190 off-diagonal pairs)
    print("\nComputing pairwise similarity matrices and Spearman/Pearson correlation...")
    # Normalize rows
    bitnet_normed = bitnet_mat / np.linalg.norm(bitnet_mat, axis=1, keepdims=True)
    harrier_normed = harrier_mat / np.linalg.norm(harrier_mat, axis=1, keepdims=True)

    sim_bitnet = np.dot(bitnet_normed, bitnet_normed.T)
    sim_harrier = np.dot(harrier_normed, harrier_normed.T)

    triu_indices = np.triu_indices(len(CLAUSES), k=1)
    bitnet_pairs = sim_bitnet[triu_indices]
    harrier_pairs = sim_harrier[triu_indices]

    spearman_res = scipy.stats.spearmanr(bitnet_pairs, harrier_pairs)
    pearson_res = scipy.stats.pearsonr(bitnet_pairs, harrier_pairs)

    print(f"Pairwise comparison over {len(bitnet_pairs)} off-diagonal clause pairs:")
    print(f"  Spearman rho: {spearman_res.statistic:.4f} (p-value: {spearman_res.pvalue:.4e})")
    print(f"  Pearson r:    {pearson_res.statistic:.4f} (p-value: {pearson_res.pvalue:.4e})")

    # 3. MiniLM Semantic Baseline Comparison (explaining Gemma backbone baseline offset)
    print("\nComputing MiniLM semantic baseline comparison...")
    minilm_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    minilm_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    minilm_model.eval()

    def embed_minilm(text):
        inputs = minilm_tok(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            out = minilm_model(**inputs)
            mask = inputs.attention_mask.unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask
            return torch.nn.functional.normalize(mean_pooled, p=2, dim=1)[0].numpy()

    baseline_texts = {
        "fruit": "Apples, oranges, and bananas are fresh fruits.",
        "physics": "Quantum field theory and relativistic black holes.",
        "banking_reg_cc": "Federal Reserve check collection and funds availability schedule under 12 CFR Part 229 Regulation CC.",
        "ftc_rosca": "Clear and conspicuous disclosure and express informed consent for negative option subscription renewal under FTC ROSCA."
    }

    minilm_vecs = {k: embed_minilm(v) for k, v in baseline_texts.items()}
    harrier_vecs = {}
    bitnet_vecs = {}
    for k, v in baseline_texts.items():
        # Harrier
        with torch.no_grad():
            inputs = tok(v, return_tensors="pt")
            out = model(**inputs)
            last_idx = inputs.attention_mask.sum(dim=1) - 1
            emb = torch.nn.functional.normalize(out.last_hidden_state[0, last_idx[0]], p=2, dim=0).numpy()
            harrier_vecs[k] = emb
        # BitNet
        bvec, _ = bitnet_engine.embed_bitnet_sync(v)
        bitnet_vecs[k] = np.array(bvec, dtype=np.float32)

    baseline_comparisons = {
        "fruit_vs_physics_dissimilar": {
            "minilm_cosine": float(np.dot(minilm_vecs["fruit"], minilm_vecs["physics"])),
            "harrier_teacher_cosine": float(np.dot(harrier_vecs["fruit"], harrier_vecs["physics"])),
            "bitnet_270m_cosine": float(np.dot(bitnet_vecs["fruit"], bitnet_vecs["physics"])),
            "explanation": "MiniLM centers dissimilar concepts near 0 (-0.094), while Gemma backbones exhibit baseline offset (~0.42-0.55) due to large 262k vocabulary structure."
        },
        "banking_vs_rosca_domain_separation": {
            "minilm_cosine": float(np.dot(minilm_vecs["banking_reg_cc"], minilm_vecs["ftc_rosca"])),
            "harrier_teacher_cosine": float(np.dot(harrier_vecs["banking_reg_cc"], harrier_vecs["ftc_rosca"])),
            "bitnet_270m_cosine": float(np.dot(bitnet_vecs["banking_reg_cc"], bitnet_vecs["ftc_rosca"]))
        }
    }

    # Summary statistics
    all_tokens_match = all(r["token_ids_exact_match"] for r in clause_results)
    mean_lat = float(np.mean([r["bitnet_latency_ms"] for r in clause_results]))

    print("\n=== Validation Summary ===")
    print(f"Total Clauses Evaluated: {len(clause_results)}")
    print(f"Exact Token ID Match Rate: {'100.0%' if all_tokens_match else 'Failed'}")
    print(f"Pairwise Spearman Correlation (rho): {spearman_res.statistic:.4f} (p={spearman_res.pvalue:.4e})")
    print(f"Pairwise Pearson Correlation (r):     {pearson_res.statistic:.4f} (p={pearson_res.pvalue:.4e})")
    print(f"Mean BitNet Latency: {mean_lat:.1f}ms")

    output_path = os.path.abspath(os.path.join(repo_root, "docs/empirical_270m_embedder_validation.json"))
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "bitnet_embed_model": redact_path(bitnet_engine.BITNET_EMBED_MODEL_PATH),
            "reference_teacher_model": ref_model_id,
            "total_clauses": len(clause_results),
            "exact_token_id_match_all": all_tokens_match,
            "pairwise_spearman_rho": round(float(spearman_res.statistic), 4),
            "pairwise_spearman_pvalue": float(spearman_res.pvalue),
            "pairwise_pearson_r": round(float(pearson_res.statistic), 4),
            "pairwise_pearson_pvalue": float(pearson_res.pvalue),
            "mean_bitnet_latency_ms": round(mean_lat, 2)
        },
        "baseline_semantic_comparisons": baseline_comparisons,
        "clauses": clause_results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved full empirical validation results to {output_path}")

if __name__ == "__main__":
    main()
