import sys
import torch
# Monkeypatch torch.compile to identity on Python 3.12 where Dynamo is unsupported
torch.compile = lambda fn, *args, **kwargs: fn

from transformers import AutoModelForCausalLM, AutoTokenizer

statute_text = """12 CFR Part 229 - Availability of Funds and Collection of Checks (Regulation CC)
Authority: 12 U.S.C. 4001-4010, 12 U.S.C. 5001-5018.
Source: 53 FR 19433, May 27, 1988; as amended at 89 FR 54997, July 3, 2024 (effective July 1, 2025).

Subpart B - Availability of Funds and Disclosure of Schedules
Section 229.10 - Next-day availability.

(a) Cash deposits. (1) A bank shall make funds deposited in an account by cash available for withdrawal not later than the business day after the banking day on which the cash is deposited.
(b) Electronic payments. (1) A bank shall make funds received for deposit in an account by an electronic payment available for withdrawal not later than the business day after the banking day on which the bank receives the electronic payment.
(c) Certain check deposits. (1) General rule. A depositary bank shall make funds deposited in an account by check available for withdrawal not later than the business day after the banking day on which the funds are deposited, in the case of:
(i) A check drawn on the Treasury of the United States and deposited in an account held by a payee of the check;
(ii) A U.S. Postal Service money order deposited in person to an employee of the depositary bank and held by a payee of the money order;
(iii) A check drawn on a Federal Reserve Bank or Federal Home Loan Bank and deposited in person to an employee of the depositary bank and held by a payee of the check;
(iv) A check drawn by a State or a unit of general local government and deposited in person to an employee of the depositary bank;
(v) A cashier check, certified check, or teller check deposited in person to an employee of the depositary bank and held by a payee of the check; and
(vi) The lesser of $275 or the aggregate amount deposited on any one banking day to all accounts of the customer by all checks not subject to next-day availability under paragraphs (c)(1)(i) through (v) of this section.

Section 229.12 - Availability schedule.
(b) Permanent schedule. (1) Local checks. A depositary bank shall make funds deposited in an account by a local check available for withdrawal not later than the second business day following the banking day on which funds are deposited.

Section 229.13 - Exceptions.
(b) Large deposits. Sections 229.10(c) and 229.12 do not apply to the aggregate amount of deposits by one or more checks to the extent that the aggregate amount is in excess of $6,725 on any one banking day.

Question: What is the statutory dollar threshold for large deposits under Section 229.13?"""

model_id = "microsoft/BitNet-b1.58-2B-4T"
print("Loading HF tokenizer & model...")
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, low_cpu_mem_usage=True)
model.eval()

# Chat templated prompt (HF format)
messages = [{"role": "user", "content": statute_text}]
prompt_hf = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tok(prompt_hf, return_tensors="pt")
print(f"HF Input token count: {inputs.input_ids.shape[1]}")

# 1. Greedy decoding (do_sample=False, temp 0)
print("\n--- HF PyTorch Test 1: Greedy Decoding (do_sample=False, no repetition penalty) ---")
with torch.no_grad():
    out1 = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        pad_token_id=tok.eos_token_id
    )
gen_tokens1 = out1[0][inputs.input_ids.shape[1]:]
print("Generated Tokens:", gen_tokens1.tolist())
print("Decoded Output:\n", repr(tok.decode(gen_tokens1, skip_special_tokens=False)))

# 2. Greedy with repetition penalty 1.1
print("\n--- HF PyTorch Test 2: Greedy Decoding with repetition_penalty=1.1 ---")
with torch.no_grad():
    out2 = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id
    )
gen_tokens2 = out2[0][inputs.input_ids.shape[1]:]
print("Generated Tokens:", gen_tokens2.tolist())
print("Decoded Output:\n", repr(tok.decode(gen_tokens2, skip_special_tokens=False)))

# 3. Sampling at temp 0.5 with repetition penalty 1.1
print("\n--- HF PyTorch Test 3: Sampled Decoding (temp=0.5, repetition_penalty=1.1) ---")
with torch.no_grad():
    out3 = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=True,
        temperature=0.5,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id
    )
gen_tokens3 = out3[0][inputs.input_ids.shape[1]:]
print("Generated Tokens:", gen_tokens3.tolist())
print("Decoded Output:\n", repr(tok.decode(gen_tokens3, skip_special_tokens=False)))
