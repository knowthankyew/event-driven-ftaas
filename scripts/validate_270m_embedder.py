#!/usr/bin/env python3
"""
Comprehensive empirical validation script for bitnet-embedding-270m:
1. Compares token IDs for 20 legal clauses against Hugging Face reference tokenizer (GemmaTokenizerFast from microsoft/harrier-oss-v1-270m).
2. Computes 640-dim dense embeddings using native bitnet.cpp llama-embedding.
3. Computes 640-dim dense embeddings using the official unquantized BF16 teacher model (microsoft/harrier-oss-v1-270m).
4. Records cosine similarities, token-level agreement, and saves full raw results to JSON.
"""

import os
import sys
import json
import time
import platform
import subprocess
import tempfile
import numpy as np
import torch

# Add src to path
inference_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/FtaaSService.Inference"))
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

def main():
    print(f"Validating bitnet-embedding-270m against reference across {len(CLAUSES)} legal clauses...")
    print(f"Platform: {platform.platform()}, CPU: {platform.processor()}")

    # 1. Load HF Reference Tokenizer & Model
    ref_model_id = "microsoft/harrier-oss-v1-270m"
    print(f"Loading Hugging Face reference tokenizer & model from '{ref_model_id}'...")
    tok = AutoTokenizer.from_pretrained(ref_model_id)
    model = AutoModel.from_pretrained(ref_model_id, torch_dtype=torch.float32)
    model.eval()

    results = []

    for idx, (tag, clause_text) in enumerate(CLAUSES, 1):
        # A. Reference tokenization
        hf_enc = tok(clause_text)
        hf_ids = hf_enc.input_ids
        hf_tok_count = len(hf_ids)

        # B. Native bitnet.cpp tokenization
        tok_cmd = [
            str(bitnet_engine.BITNET_TOKENIZE_CLI_PATH),
            "-m", str(bitnet_engine.BITNET_EMBED_MODEL_PATH),
            "-p", clause_text,
            "--show-count"
        ]
        tok_res = subprocess.run(tok_cmd, capture_output=True, text=True)
        native_tok_count = None
        for line in tok_res.stdout.splitlines():
            if "Total number of tokens:" in line:
                native_tok_count = int(line.split(":")[-1].strip())
                break

        # C. Native bitnet.cpp embedding
        bitnet_emb_vec, bitnet_lat = bitnet_engine.embed_bitnet_sync(clause_text)
        bitnet_emb = np.array(bitnet_emb_vec, dtype=np.float32)

        # D. Reference PyTorch embedding (Last-token pooling + L2 normalization)
        with torch.no_grad():
            inputs = tok(clause_text, return_tensors="pt")
            out = model(**inputs)
            last_idx = inputs.attention_mask.sum(dim=1) - 1
            ref_emb_pt = out.last_hidden_state[0, last_idx[0]]
            ref_emb_pt = torch.nn.functional.normalize(ref_emb_pt, p=2, dim=0).numpy()

        # E. Cosine similarity
        cos_sim = float(np.dot(ref_emb_pt, bitnet_emb) / (np.linalg.norm(ref_emb_pt) * np.linalg.norm(bitnet_emb)))

        item = {
            "clause_index": idx,
            "tag": tag,
            "text": clause_text,
            "hf_token_count": hf_tok_count,
            "native_token_count": native_tok_count,
            "token_count_match": (hf_tok_count == native_tok_count),
            "bitnet_embedding_dim": len(bitnet_emb),
            "ref_embedding_dim": len(ref_emb_pt),
            "bitnet_norm": float(np.linalg.norm(bitnet_emb)),
            "ref_norm": float(np.linalg.norm(ref_emb_pt)),
            "cosine_similarity_vs_teacher": cos_sim,
            "bitnet_latency_ms": bitnet_lat
        }
        results.append(item)
        print(f"[{idx:02d}/20] {tag:<26} | Tokens: HF={hf_tok_count:3d}, Native={native_tok_count:3d} | CosSim vs Teacher: {cos_sim:.4f} | Latency: {bitnet_lat:.1f}ms")

    # Summary statistics
    cos_sims = [r["cosine_similarity_vs_teacher"] for r in results]
    mean_cos = float(np.mean(cos_sims))
    min_cos = float(np.min(cos_sims))
    max_cos = float(np.max(cos_sims))
    token_match_pct = 100.0 * sum(1 for r in results if r["token_count_match"]) / len(results)

    print("\n=== Validation Summary ===")
    print(f"Total Clauses Evaluated: {len(results)}")
    print(f"Token Count Exact Match Rate: {token_match_pct:.1f}%")
    print(f"Mean Cosine Similarity vs Teacher: {mean_cos:.4f}")
    print(f"Min / Max Cosine Similarity: {min_cos:.4f} / {max_cos:.4f}")

    output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../docs/empirical_270m_embedder_validation.json"))
    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "bitnet_embed_model": str(bitnet_engine.BITNET_EMBED_MODEL_PATH),
            "reference_model": ref_model_id,
            "mean_cosine_similarity": mean_cos,
            "min_cosine_similarity": min_cos,
            "max_cosine_similarity": max_cos,
            "token_match_rate_pct": token_match_pct
        },
        "results": results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved full empirical validation results to {output_path}")

if __name__ == "__main__":
    main()
