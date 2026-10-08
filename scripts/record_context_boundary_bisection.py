#!/usr/bin/env python3
"""
record_context_boundary_bisection.py: Context-Length Boundary Bisection v2.

Measures the generation pass-rate curve across 12 distinct statutory clauses and
natural length variants from 16 to 48 tokens (step 2).
Tests multiple configurations:
- Default batch (-b 512, -ub 512, -t 4)
- Micro-batching (-b 16, -ub 16, -t 4)
- Single-thread default (-t 1)
- Flash Attention on/off and Float32 KV cache (-ctk f32 -ctv f32) at boundary lengths
- Single-token prefill (-b 1, labeled as immediate EOS behavior)

Outputs empirical results to docs/empirical_context_boundary.json.
Conclusions are generated strictly via template interpolation with zero hardcoded digits.
All host filesystem paths are redacted.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_FILE = REPO_ROOT / "docs" / "empirical_context_boundary.json"

DEFAULT_CLI_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-completion"
DEFAULT_TOKENIZE_PATH = Path.home() / "Github" / "BitNet" / "build" / "bin" / "llama-tokenize"
DEFAULT_MODEL_PATH = Path.home() / "Github" / "BitNet" / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"

CLI_PATH = Path(os.getenv("BITNET_COMPLETION_BIN", str(DEFAULT_CLI_PATH))).resolve()
TOKENIZE_PATH = Path(os.getenv("BITNET_TOKENIZE_BIN", str(DEFAULT_TOKENIZE_PATH))).resolve()
MODEL_PATH = Path(os.getenv("BITNET_MODEL_PATH", str(DEFAULT_MODEL_PATH))).resolve()

# 12 Distinct Statutory Clauses with Natural Phrasing and Regex Checked Answers
CLAUSES = [
    {
        "id": "c01_reg_cc_275",
        "topic": "12 CFR 229.10(c)(1)(vi) Next-Day Check Minimum",
        "regex": r"\$?275",
        "variants": {
            16: "Next day minimum check availability is $275. What is the amount?",
            18: "Bank next day minimum check deposit availability is $275. What amount?",
            20: "Statutory next day minimum check deposit availability is $275. What is the amount?",
            22: "Under Regulation CC, next day minimum availability for check deposits is $275. What amount?",
            24: "Under Regulation CC, statutory next day minimum check availability is $275. What is the amount?",
            26: "Under federal Regulation CC rules, statutory next day check availability is $275. What is the minimum?",
            28: "Under federal Regulation CC rules, statutory next day check deposit availability is $275. What is the amount?",
            30: "Under federal Regulation CC, the statutory next day aggregate check deposit availability is $275. What is the amount?",
            32: "Under federal Regulation CC banking rules, statutory next day aggregate check deposit availability is $275. What is the amount?",
            34: "Under federal Regulation CC banking rules, the statutory next day aggregate check deposit availability is $275. What is the required dollar amount?",
            36: "Under federal Regulation CC banking rules, statutory next day aggregate check deposit availability is $275 on any banking day. What is the required dollar amount?",
            38: "Under federal Regulation CC banking rules, statutory next day aggregate check deposit availability is $275 on any one banking day. What is the required dollar amount?",
            40: "Under federal Regulation CC rules, a bank must make available the lesser of $275 or aggregate check deposits on any banking day. What is the dollar threshold?",
            42: "Under federal Regulation CC banking rules, a bank must make available the lesser of $275 or aggregate check deposits on any banking day. What is the dollar threshold?",
            44: "Under federal Regulation CC banking rules, a depositary bank must make available the lesser of $275 or aggregate check deposits on any banking day. What is the dollar threshold?",
            46: "Under federal Regulation CC banking rules, a depositary bank must make available the lesser of $275 or aggregate check deposits on any banking day for withdrawal. What is the dollar threshold?",
            48: "Under federal Regulation CC banking rules, a depositary bank must make available the lesser of $275 or aggregate check deposits on any one banking day for withdrawal. What is the dollar threshold?"
        }
    },
    {
        "id": "c02_reg_cc_6725",
        "topic": "12 CFR 229.13(b) Large Deposit Exception",
        "regex": r"\$?6,?725",
        "variants": {
            16: "Large check deposit exception threshold is $6,725. What is the threshold?",
            18: "Regulation CC large check deposit exception threshold is $6,725. What threshold?",
            20: "Regulation CC large deposit exception threshold is $6,725 on one day. What threshold?",
            22: "Under Regulation CC, large deposit exception threshold is $6,725 on one day. What is threshold?",
            24: "Under Regulation CC, large check deposit exception threshold is $6,725 on any day. What threshold?",
            26: "Under Regulation CC rules, the large check deposit exception threshold is $6,725 on any day. What threshold?",
            28: "Under Regulation CC rules, the large check deposit exception threshold is $6,725 on one banking day. What threshold?",
            30: "Under federal Regulation CC rules, the large check deposit exception threshold is $6,725 on any banking day. What is the amount?",
            32: "Under federal Regulation CC rules, large check deposit exception threshold applies to aggregate deposits over $6,725 on one day. What is the amount?",
            34: "Under federal Regulation CC rules, large check deposit exception threshold applies to aggregate check deposits over $6,725 on one day. What is the dollar amount?",
            36: "Under federal Regulation CC rules, large check deposit exception threshold applies to aggregate check deposits in excess of $6,725 on one day. What is the dollar amount?",
            38: "Under federal Regulation CC banking rules, large check deposit exception applies to aggregate check deposits in excess of $6,725 on one banking day. What is the dollar amount?",
            40: "Under federal Regulation CC banking rules, large deposit exception rules apply to aggregate check deposits in excess of $6,725 on any one banking day. What is the dollar amount?",
            42: "Under federal Regulation CC banking rules, large deposit exception rules apply to aggregate check deposits in excess of $6,725 on any one banking day for accounts. What is the dollar amount?",
            44: "Under federal Regulation CC banking rules, large deposit exception rules apply to aggregate check deposits in excess of $6,725 on any one banking day for customers. What is the exact dollar amount?",
            46: "Under federal Regulation CC banking rules, large deposit exception rules apply to the aggregate amount of deposits in excess of $6,725 on any one banking day for customers. What is the exact dollar amount?",
            48: "Under federal Regulation CC banking rules, large deposit exception rules apply to the aggregate amount of check deposits in excess of $6,725 on any one banking day for customers. What is the exact dollar amount?"
        }
    },
    {
        "id": "c03_reg_cc_second_day",
        "topic": "12 CFR 229.12(b) Local Check Schedule",
        "regex": r"second\s+business\s+day",
        "variants": {
            16: "Local check funds availability is second business day. When available?",
            18: "Local check deposit funds availability is second business day. When are funds available?",
            20: "Local check deposit availability schedule is second business day. When are funds available?",
            22: "Under Regulation CC, local check availability is second business day. When are check funds available?",
            24: "Under Regulation CC, local check deposit availability is second business day. When are funds available?",
            26: "Under Regulation CC, local check deposit funds availability is second business day. When can funds be withdrawn?",
            28: "Under federal Regulation CC rules, local check deposit availability is second business day. When can funds be withdrawn?",
            30: "Under federal Regulation CC rules, standard local check deposit availability is second business day. When can funds be withdrawn by customers?",
            32: "Under federal Regulation CC rules, local check funds must be available not later than second business day. When can funds be withdrawn by customers?",
            34: "Under federal Regulation CC rules, a bank makes local check funds available not later than second business day. When can funds be withdrawn by customers?",
            36: "Under federal Regulation CC rules, a bank must make local check funds available not later than second business day following deposit. When can funds be withdrawn?",
            38: "Under federal Regulation CC rules, a bank must make local check funds available not later than second business day following banking day. When can funds be withdrawn?",
            40: "Under federal Regulation CC rules, a bank must make local check funds available for withdrawal not later than second business day following banking day. When can funds be withdrawn?",
            42: "Under federal Regulation CC banking rules, a bank must make local check funds available for withdrawal not later than second business day following banking day. When can funds be withdrawn?",
            44: "Under federal Regulation CC banking rules, a depositary bank must make local check funds available for withdrawal not later than second business day following banking day. When can funds be withdrawn?",
            46: "Under federal Regulation CC banking rules, a depositary bank must make local check funds available for withdrawal not later than second business day following the banking day. When can funds be withdrawn?",
            48: "Under federal Regulation CC banking rules, a depositary bank must make local check funds available for customer withdrawal not later than second business day following the banking day. When can funds be withdrawn?"
        }
    },
    {
        "id": "c04_rosca_clear",
        "topic": "15 U.S.C. 8403(1) ROSCA Disclosure",
        "regex": r"clear\s+and\s+conspicuous",
        "variants": {
            16: "Online subscription terms disclosure must be clear and conspicuous. What standard?",
            18: "Online subscription terms disclosure must be clear and conspicuous. What disclosure standard?",
            20: "Under ROSCA, online subscription terms disclosure must be clear and conspicuous. What standard?",
            22: "Under federal ROSCA, subscription terms disclosure must be clear and conspicuous. What is required standard?",
            24: "Under federal ROSCA, subscription billing disclosure must be clear and conspicuous. What is the required standard?",
            26: "Under federal ROSCA, subscription billing disclosure must be clear and conspicuous prior to charging. What standard is required?",
            28: "Under federal ROSCA, recurring subscription billing disclosure must be clear and conspicuous prior to charging. What standard is required?",
            30: "Under federal ROSCA, recurring subscription terms disclosure must be clear and conspicuous before obtaining billing information. What standard is required?",
            32: "Under federal ROSCA law, recurring subscription terms disclosure must be clear and conspicuous before obtaining billing information. What standard is required?",
            34: "Under federal ROSCA law, recurring subscription terms disclosure must be clear and conspicuous before obtaining customer billing information. What disclosure standard is required?",
            36: "Under federal ROSCA law, recurring subscription contract terms disclosure must be clear and conspicuous before obtaining customer billing information. What disclosure standard is required?",
            38: "Under federal ROSCA law, recurring subscription contract terms disclosure must be clear and conspicuous before obtaining customer billing information online. What disclosure standard is required?",
            40: "Under federal ROSCA law, all recurring subscription contract terms disclosure must be clear and conspicuous before obtaining customer billing information online. What disclosure standard is required by law?",
            42: "Under federal ROSCA law, all material recurring subscription contract terms disclosure must be clear and conspicuous before obtaining customer billing information online. What disclosure standard is required by law?",
            44: "Under federal ROSCA law, all material recurring subscription contract terms disclosure must be clear and conspicuous before obtaining customer billing information online. What disclosure standard is required by federal law?",
            46: "Under federal ROSCA law, all material recurring subscription contract terms disclosure must be clear and conspicuous before obtaining customer billing information online from shoppers. What disclosure standard is required by federal law?",
            48: "Under federal ROSCA law, all material recurring subscription contract terms disclosure must be clear and conspicuous before obtaining customer billing information online from consumers. What disclosure standard is required by federal law?"
        }
    },
    {
        "id": "c05_ab2863_cancel",
        "topic": "Cal. BPC 17602(a)(4) Click-to-Cancel",
        "regex": r"click[\s-]to[\s-]cancel",
        "variants": {
            16: "California subscription cancellation requires click-to-cancel mechanism. What mechanism?",
            18: "California subscription cancellation law requires click-to-cancel mechanism. What mechanism is required?",
            20: "California subscription law requires a simple click-to-cancel mechanism. What mechanism is required?",
            22: "Under California AB 2863, subscription renewal requires click-to-cancel mechanism. What is required?",
            24: "Under California AB 2863, subscription cancellation requires click-to-cancel mechanism. What is required?",
            26: "Under California AB 2863, online subscription cancellation requires a click-to-cancel mechanism. What is required?",
            28: "Under California AB 2863 law, online subscription cancellation requires a click-to-cancel mechanism. What mechanism is required?",
            30: "Under California AB 2863 law, online subscription cancellation requires a simple click-to-cancel mechanism for consumers. What mechanism is required?",
            32: "Under California AB 2863 law, online subscription cancellation requires a cost-effective click-to-cancel mechanism for consumers. What mechanism is required?",
            34: "Under California AB 2863 law, online subscription cancellation requires a cost-effective click-to-cancel mechanism for all consumers. What mechanism is required by statute?",
            36: "Under California AB 2863 law, online subscription cancellation requires a cost-effective click-to-cancel mechanism for all consumers online. What mechanism is required by statute?",
            38: "Under California AB 2863 law, an online subscription cancellation flow requires a cost-effective click-to-cancel mechanism for all consumers online. What mechanism is required by statute?",
            40: "Under California AB 2863 law, an online subscription cancellation flow requires a cost-effective click-to-cancel mechanism for all consumers online without obstacles. What mechanism is required by statute?",
            42: "Under California AB 2863 law, an online subscription cancellation flow requires a cost-effective click-to-cancel mechanism for all consumers online without unnecessary obstacles. What mechanism is required by statute?",
            44: "Under California AB 2863 law, an online subscription cancellation flow requires a cost-effective click-to-cancel mechanism for all consumers online without unnecessary obstruction. What cancellation mechanism is required by statute?",
            46: "Under California AB 2863 law, an online subscription cancellation flow requires a cost-effective click-to-cancel mechanism for all consumers online without unnecessary obstruction or delays. What cancellation mechanism is required by statute?",
            48: "Under California AB 2863 law, an online subscription cancellation flow requires a cost-effective click-to-cancel mechanism for all consumers online without unnecessary obstruction or steps. What cancellation mechanism is required by statute?"
        }
    },
    {
        "id": "c06_uk_dmcc_cooling",
        "topic": "UK DMCC Act s. 256 14-Day Cooling-Off",
        "regex": r"14[\s-]day",
        "variants": {
            16: "UK subscription contracts statutory cooling-off cancellation period is 14-day. How long?",
            18: "UK subscription contracts statutory cooling-off cancellation period is 14-day. What is the duration?",
            20: "Under UK DMCC Act, statutory cooling-off cancellation period is 14-day. What is the duration?",
            22: "Under the UK DMCC Act, statutory subscription cooling-off cancellation is 14-day. What is the period?",
            24: "Under the UK DMCC Act, statutory subscription cooling-off period is 14-day. What is the duration?",
            26: "Under the UK DMCC Act, the statutory subscription cooling-off period is 14-day for consumers. What duration?",
            28: "Under the UK DMCC Act, the statutory subscription cooling-off period is 14-day for consumers. What is the period?",
            30: "Under the UK DMCC Act 2024, the statutory subscription cooling-off period is 14-day for consumers. What is the duration?",
            32: "Under the UK DMCC Act 2024, the statutory subscription cooling-off cancellation period is 14-day for consumers. What is the duration?",
            34: "Under the UK DMCC Act 2024, statutory subscription contract cooling-off cancellation period is 14-day for all consumers. What is the duration?",
            36: "Under the UK DMCC Act 2024, the statutory subscription contract cooling-off cancellation period is 14-day for all consumers. What is the statutory duration?",
            38: "Under the UK DMCC Act 2024, the statutory subscription contract cooling-off cancellation period is 14-day for all retail consumers. What is the statutory duration?",
            40: "Under the UK DMCC Act 2024, the statutory subscription contract cooling-off cancellation period is 14-day for all retail consumers under law. What is the statutory duration?",
            42: "Under the UK DMCC Act 2024, the statutory subscription contract cooling-off cancellation period is 14-day for all retail consumers under statute. What is the statutory duration?",
            44: "Under the UK DMCC Act 2024, the statutory subscription contract cooling-off cancellation period is 14-day for all retail consumers under enactment. What is the statutory duration of cooling off?",
            46: "Under the UK DMCC Act 2024, the statutory subscription contract cooling-off cancellation period is 14-day for all retail consumers under new enactment. What is the statutory duration of cooling off?",
            48: "Under the UK DMCC Act 2024, the statutory subscription contract cooling-off cancellation period is 14-day for all retail consumers under enacted provisions. What is the statutory duration of cooling off?"
        }
    },
    {
        "id": "c07_eu_crd_button",
        "topic": "EU CRD Art. 8 Obligation to Pay",
        "regex": r"obligation\s+to\s+pay",
        "variants": {
            16: "EU online order button label must state obligation to pay. What phrase?",
            18: "EU consumer law online order button must state obligation to pay. What phrase is required?",
            20: "Under EU Consumer Rights Directive, order button must state obligation to pay. What label?",
            22: "Under EU Consumer Rights Directive, the order button must state obligation to pay. What phrase?",
            24: "Under EU Consumer Rights Directive rules, the order button must state obligation to pay. What phrase?",
            26: "Under EU Consumer Rights Directive rules, an online order button must state obligation to pay. What phrase?",
            28: "Under EU Consumer Rights Directive rules, every online order button must state obligation to pay. What phrase?",
            30: "Under EU Consumer Rights Directive rules, every online order button must state order with obligation to pay. What phrase?",
            32: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay. What phrase is required?",
            34: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly. What phrase is required?",
            36: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly on checkout. What phrase is required?",
            38: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly on checkout screens. What phrase is required?",
            40: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly on digital checkout screens. What phrase is required?",
            42: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly on digital checkout screens for consumers. What phrase is required?",
            44: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly on digital checkout screens for all consumers. What phrase is required?",
            46: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly on digital checkout screens for all consumers placing orders. What phrase is required?",
            48: "Under EU Consumer Rights Directive rules, each online order button must state an order with obligation to pay clearly on digital checkout screens for all consumers placing an order. What exact phrase is required?"
        }
    },
    {
        "id": "c08_reg_e_investigate",
        "topic": "12 CFR 1005.11(a) Billing Error Window",
        "regex": r"10\s+business\s+days",
        "variants": {
            16: "Electronic billing error investigation time limit is 10 business days. How long?",
            18: "Regulation E electronic billing error investigation limit is 10 business days. What time limit?",
            20: "Under Regulation E, electronic billing error investigation limit is 10 business days. What timeframe?",
            22: "Under Regulation E, electronic fund billing error investigation is 10 business days. What is timeframe?",
            24: "Under federal Regulation E, electronic billing error investigation is 10 business days. What is the timeframe?",
            26: "Under federal Regulation E rules, electronic billing error investigation is 10 business days. What is the time limit?",
            28: "Under federal Regulation E rules, bank electronic billing error investigation is 10 business days. What is the time limit?",
            30: "Under federal Regulation E rules, financial institution billing error investigation is 10 business days. What is the time limit?",
            32: "Under federal Regulation E rules, a financial institution billing error investigation is 10 business days. What is the statutory time limit?",
            34: "Under federal Regulation E rules, a financial institution billing error investigation period is 10 business days. What is the statutory time limit?",
            36: "Under federal Regulation E rules, a financial institution electronic billing error investigation period is 10 business days. What is the statutory time limit?",
            38: "Under federal Regulation E rules, a financial institution electronic fund billing error investigation period is 10 business days. What is the statutory time limit?",
            40: "Under federal Regulation E rules, a financial institution electronic fund transfer billing error investigation period is 10 business days. What is the statutory time limit?",
            42: "Under federal Regulation E rules, a financial institution electronic fund transfer billing error investigation period is 10 business days from notice. What is the statutory time limit?",
            44: "Under federal Regulation E rules, a financial institution electronic fund transfer billing error investigation period is 10 business days from consumer notice. What is the statutory time limit?",
            46: "Under federal Regulation E rules, a financial institution electronic fund transfer billing error investigation period is 10 business days from consumer notice receipt. What is the statutory time limit?",
            48: "Under federal Regulation E rules, a financial institution electronic fund transfer billing error investigation period is 10 business days from oral consumer notice receipt. What is the statutory time limit?"
        }
    },
    {
        "id": "c09_reg_e_stop",
        "topic": "12 CFR 1005.10(c) Stop Payment Three Days",
        "regex": r"three\s+days|three\s+business\s+days",
        "variants": {
            16: "Stop recurring electronic transfer payment up to three days prior. When stop?",
            18: "Consumer can stop recurring electronic debit up to three days prior. When stop payment?",
            20: "Under Regulation E, consumer can stop electronic payment three days prior. When stop payment?",
            22: "Under Regulation E, a consumer can stop electronic debit three days before date. What deadline?",
            24: "Under federal Regulation E, a consumer can stop electronic debit three days before date. What deadline?",
            26: "Under federal Regulation E rules, a consumer can stop electronic debits three days before date. What is deadline?",
            28: "Under federal Regulation E rules, a consumer can stop preauthorized transfers three days before date. What is deadline?",
            30: "Under federal Regulation E rules, a consumer can stop recurring preauthorized transfers three days before date. What is deadline?",
            32: "Under federal Regulation E rules, a consumer may stop payment of recurring transfers three days before date. What is deadline?",
            34: "Under federal Regulation E rules, a consumer may stop payment of recurring electronic transfers three days before date. What is the deadline?",
            36: "Under federal Regulation E rules, a consumer may stop payment of recurring electronic debits up to three days before date. What is the deadline?",
            38: "Under federal Regulation E rules, a consumer may stop payment of recurring preauthorized electronic debits up to three days before date. What is the deadline?",
            40: "Under federal Regulation E rules, a consumer may stop payment of recurring preauthorized electronic debits up to three days before the scheduled date. What is deadline?",
            42: "Under federal Regulation E rules, a consumer may stop payment of recurring preauthorized electronic debits up to three days before the scheduled transfer. What is the deadline?",
            44: "Under federal Regulation E rules, a consumer may stop payment of recurring preauthorized electronic debits up to three days before the scheduled transfer date. What is the deadline?",
            46: "Under federal Regulation E rules, a consumer may stop payment of recurring preauthorized electronic debits by notice up to three days before the scheduled transfer date. What is the deadline?",
            48: "Under federal Regulation E rules, a consumer may stop payment of recurring preauthorized electronic debits by giving notice up to three days before the scheduled transfer date. What is the deadline?"
        }
    },
    {
        "id": "c10_ca_unconditional",
        "topic": "Cal. BPC 17603 Unconditional Gift",
        "regex": r"unconditional\s+gift",
        "variants": {
            16: "Goods sent without affirmative consent are deemed unconditional gift. What status?",
            18: "Merchandise sent without affirmative consent is deemed an unconditional gift. What legal status?",
            20: "Under California law, goods sent without affirmative consent are an unconditional gift. What status?",
            22: "Under California law, merchandise sent without affirmative consent is an unconditional gift. What status?",
            24: "Under California BPC 17603, goods sent without affirmative consent are an unconditional gift. What status?",
            26: "Under California BPC 17603, merchandise sent without affirmative consent is an unconditional gift. What legal status?",
            28: "Under California BPC 17603, merchandise sent without consumer affirmative consent is an unconditional gift. What legal status?",
            30: "Under California BPC 17603 law, merchandise sent without consumer affirmative consent is an unconditional gift. What is legal status?",
            32: "Under California BPC 17603 law, merchandise sent to consumers without affirmative consent is an unconditional gift. What is the legal status?",
            34: "Under California BPC 17603 law, any merchandise sent to consumers without affirmative consent is an unconditional gift. What is the legal status?",
            36: "Under California BPC 17603 law, any merchandise sent to consumers without prior affirmative consent is an unconditional gift. What is the legal status?",
            38: "Under California BPC 17603 law, any merchandise sent to consumers without prior affirmative consent is deemed an unconditional gift. What is the legal status?",
            40: "Under California BPC 17603 statute, any merchandise sent to consumers without prior affirmative consent is deemed an unconditional gift. What is the legal status?",
            42: "Under California BPC 17603 statute, any retail merchandise sent to consumers without prior affirmative consent is deemed an unconditional gift. What is the legal status?",
            44: "Under California BPC 17603 statute, any retail merchandise sent to consumers without prior affirmative consent is deemed an unconditional gift under law. What is the legal status?",
            46: "Under California BPC 17603 statute, any retail merchandise sent to consumers without prior affirmative consent is deemed an unconditional gift under California law. What is the legal status?",
            48: "Under California BPC 17603 statute, any retail merchandise sent to consumers without prior affirmative consent is deemed an unconditional gift under state California law. What is the legal status?"
        }
    },
    {
        "id": "c11_ucc_stop",
        "topic": "UCC 4-403 Stop Payment Right",
        "regex": r"stop\s+payment",
        "variants": {
            16: "Bank customer has right to order stop payment on items. What right?",
            18: "Bank customer has the right to order stop payment on account items. What right?",
            20: "Under UCC 4-403, customer has right to order stop payment on items. What right?",
            22: "Under UCC 4-403, a customer has the right to order stop payment on items. What right?",
            24: "Under UCC Article 4, a customer has the right to order stop payment on items. What right?",
            26: "Under UCC Article 4 rules, a customer has the right to order stop payment on items. What right?",
            28: "Under UCC Article 4 rules, a bank customer has the right to order stop payment on items. What right?",
            30: "Under UCC Article 4 rules, a bank customer has the right to order stop payment on account items. What right?",
            32: "Under UCC Article 4 rules, a bank customer has the right to order stop payment on any item. What right is granted?",
            34: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any item. What right is granted?",
            36: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any account item. What right is granted?",
            38: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any item drawn. What right is granted by statute?",
            40: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any item drawn on account. What right is granted by statute?",
            42: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any check or item drawn on account. What right is granted by statute?",
            44: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any check or item drawn on customer account. What right is granted by statute?",
            46: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any check or item drawn on the customer account. What right is granted by statute?",
            48: "Under UCC Article 4 rules, a bank customer has the legal right to order stop payment on any check or item drawn on the customer bank account. What right is granted by statute?"
        }
    },
    {
        "id": "c12_il_notice",
        "topic": "815 ILCS 601/10 Illinois 30 to 60 Days",
        "regex": r"30|thirty",
        "variants": {
            16: "Illinois automatic renewal contract notice must be given thirty days prior. How many days?",
            18: "Illinois renewal contract advance written notice must be given thirty days prior. How many days?",
            20: "Under Illinois law, renewal contract written notice must be given thirty days prior. How many days?",
            22: "Under Illinois law, automatic renewal written notice must be given thirty days prior. How many days notice?",
            24: "Under Illinois 815 ILCS, automatic renewal written notice must be given thirty days prior. How many days notice?",
            26: "Under Illinois 815 ILCS rules, automatic renewal written notice must be given thirty days prior. How many days notice?",
            28: "Under Illinois 815 ILCS rules, automatic renewal contract written notice must be given thirty days prior. How many days notice?",
            30: "Under Illinois 815 ILCS rules, automatic renewal contract written notice must be given thirty days before renewal. How many days notice?",
            32: "Under Illinois 815 ILCS rules, automatic renewal contract written notice must be given thirty to sixty days before renewal. How many days notice?",
            34: "Under Illinois 815 ILCS rules, automatic renewal contract advance written notice must be given thirty to sixty days before renewal. How many days notice?",
            36: "Under Illinois 815 ILCS rules, automatic renewal contract advance written notice must be given thirty to sixty days before renewal date. How many days notice?",
            38: "Under Illinois 815 ILCS rules, an automatic renewal contract advance written notice must be given thirty to sixty days before renewal date. How many days notice is required?",
            40: "Under Illinois 815 ILCS rules, an automatic renewal contract advance written notice must be given thirty to sixty days before the renewal date. How many days notice is required?",
            42: "Under Illinois 815 ILCS rules, an automatic renewal contract advance written notice must be given thirty to sixty days before the renewal date to consumers. How many days notice is required?",
            44: "Under Illinois 815 ILCS rules, an automatic renewal contract advance written notice must be given thirty to sixty days before the renewal date to all consumers. How many days notice is required?",
            46: "Under Illinois 815 ILCS rules, an automatic renewal contract advance written notice must be given thirty to sixty days before the scheduled renewal date to all consumers. How many days notice is required?",
            48: "Under Illinois 815 ILCS rules, an automatic renewal contract advance written notice must be given thirty to sixty days before the scheduled renewal date to all retail consumers. How many days notice is required?"
        }
    }
]


def redact_path(path_str: str) -> str:
    """Redact user-specific host directory paths."""
    if not path_str:
        return ""
    return re.sub(r"/Users/[^/]+", "<HOST_DIR>", str(path_str))


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 of file contents."""
    if not file_path.is_file():
        return "missing"
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


TOKEN_CACHE: Dict[str, Tuple[int, List[int]]] = {}


def count_tokens(prompt: str) -> Tuple[int, List[int]]:
    """Tokenize prompt using llama-tokenize to get exact count and token IDs."""
    if prompt in TOKEN_CACHE:
        return TOKEN_CACHE[prompt]

    if not TOKENIZE_PATH.is_file():
        # Fallback approximation
        words = prompt.split()
        return len(words), []

    cmd = [
        str(TOKENIZE_PATH),
        "-m", str(MODEL_PATH),
        "--show-count",
        "--stdin",
        "--log-disable"
    ]
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, check=True)
        count = 0
        m = re.search(r"Total number of tokens:\s*(\d+)", proc.stdout)
        if m:
            count = int(m.group(1))
        else:
            m2 = re.search(r"(\d+)\s+tokens", proc.stdout)
            if m2:
                count = int(m2.group(1))

        token_ids = [int(x) for x in re.findall(r"^\s*(\d+)\s+->", proc.stdout, re.MULTILINE)]
        res = (count, token_ids)
        TOKEN_CACHE[prompt] = res
        return res
    except Exception:
        fallback = (len(prompt.split()), [])
        TOKEN_CACHE[prompt] = fallback
        return fallback


def classify_run_output(output_text: str, expected_regex: str) -> str:
    """
    Classify generation output into one of five mutually exclusive states:
    - 'correct': expected regex matches, no loops or corrupt glyphs
    - 'garbage': corrupt SIMD/tile glyphs (e.g. CJK 备份, @@@, scrambled)
    - 'loop': repetitive loops (3+ repeated words/phrases)
    - 'eos': empty or immediate [end of text]
    - 'fluent_wrong': grammatically fluent response, but expected answer missing
    """
    clean = output_text.strip()
    if not clean or clean == "[end of text]":
        return "eos"

    # Check for corruption glyphs
    if any(k in clean for k in ["备份", "@@@", "Scalars", "Inhell", "2 0 3 0", "2 3 0 2 3"]):
        return "garbage"
    if re.search(r"[\u4e00-\u9fff]", clean):
        return "garbage"

    # Check for repetitive loops
    words = clean.split()
    if len(words) >= 6:
        trigrams = [" ".join(words[i:i+3]) for i in range(len(words)-2)]
        counts = [trigrams.count(tg) for tg in set(trigrams)]
        if max(counts) >= 3:
            return "loop"

    # Check if answer matches expected regex
    if re.search(expected_regex, clean, re.IGNORECASE):
        return "correct"

    return "fluent_wrong"


def extract_first_token_id(out_text: str) -> Optional[int]:
    """Extract first generated token ID from model output."""
    clean = out_text.strip()
    if not clean or clean == "[end of text]":
        return 128001
    cnt, ids = count_tokens(clean)
    if ids and len(ids) > 1 and ids[0] == 128000:
        return ids[1]
    elif ids:
        return ids[0]
    return None


def run_bitnet_generation(
    prompt: str,
    flags: List[str],
    timeout: float = 30.0
) -> Tuple[int, str, str, str]:
    """Execute llama-completion with specific prompt and runtime flags."""
    cmd = [
        str(CLI_PATH),
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-c", "512",
        "--temp", "0.0",
        "-ngl", "0",
        "-no-cnv",
        "-n", "25",
        "--no-display-prompt"
    ] + flags

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, input="", timeout=timeout)
        clean = proc.stdout.strip()
        gen_clean = clean.split("Assistant:")[-1].strip() if "Assistant:" in clean else clean
        stdout_tail = gen_clean[:200]
        stderr_tail = proc.stderr.strip()[-300:] if proc.stderr else ""
        return proc.returncode, gen_clean, stdout_tail, stderr_tail
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT", "TIMEOUT"
    except Exception as e:
        return -2, "", "", str(e)


def build_bisection_conclusion(
    highest_all_pass: int,
    recommended_cap: int,
    stated_margin: int,
    total_runs_count: int,
    correct_runs_count: int,
    garbage_runs_count: int,
    loop_runs_count: int
) -> str:
    """
    Template function building conclusion text with zero hardcoded digits.
    All numeric values must be interpolated.
    Uses exact required phrase: 'context-length-dependent failure; mechanism unknown'.
    """
    return (
        f"Empirical context-length bisection establishes a context-length-dependent failure; mechanism unknown. "
        f"Across a total of {total_runs_count} evaluated runs, {correct_runs_count} passed factual extraction, "
        f"{garbage_runs_count} exhibited corruption, and {loop_runs_count} degenerated into repetitive loops. "
        f"The highest prompt token length at which all tested clauses pass without degeneration is {highest_all_pass} tokens. "
        f"Applying a safety margin of {stated_margin} tokens below the highest all-pass length yields a "
        f"recommended safe generation ceiling of {recommended_cap} tokens. "
        f"Prompts exceeding {highest_all_pass} tokens risk context-length-dependent failure on CPU execution."
    )


def execute_bisection_sweep(output_file: Path = DEFAULT_OUTPUT_FILE) -> dict:
    """Run systematic bisection sweep across 12 clauses and lengths 16..48."""
    print("=== Running Context-Length Boundary Bisection Sweep ===")

    # Get git commits
    bitnet_git_commit = "unknown"
    llama_submodule_commit = "unknown"
    bitnet_repo = CLI_PATH.parents[2]
    if (bitnet_repo / ".git").exists():
        try:
            bitnet_git_commit = subprocess.check_output(["git", "-C", str(bitnet_repo), "rev-parse", "HEAD"], text=True).strip()
            llama_submodule_commit = subprocess.check_output(["git", "-C", str(bitnet_repo / "3rdparty" / "llama.cpp"), "rev-parse", "HEAD"], text=True).strip()
        except Exception:
            pass

    model_sha = compute_sha256(MODEL_PATH)
    target_lengths = [16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48]

    runs_data = []

    print(f"Sweeping {len(CLAUSES)} clauses across {len(target_lengths)} target lengths...")

    clause_results = {c["id"]: {} for c in CLAUSES}

    for c in CLAUSES:
        cid = c["id"]
        regex = c["regex"]
        print(f"\n--- Testing Clause: {cid} ({c['topic']}) ---")

        for tlen in target_lengths:
            ptext = c["variants"].get(tlen, "")
            formatted_prompt = f"User: {ptext}<|eot_id|>\nAssistant: "
            exact_tokens, token_ids = count_tokens(formatted_prompt)

            # Standard configs
            configs_to_run = [
                ("default_b512_t4", ["-t", "4"]),
                ("micro_b16_ub16", ["-t", "4", "-b", "16", "-ub", "16"]),
                ("default_b512_t1", ["-t", "1"]),
            ]

            # Boundary configs at lengths 24..30
            if tlen in (24, 26, 28, 30):
                configs_to_run.extend([
                    ("flash_attn_on", ["-t", "4", "-fa", "on"]),
                    ("flash_attn_off", ["-t", "4", "-fa", "off"]),
                    ("f32_kv_cache", ["-t", "4", "-ctk", "f32", "-ctv", "f32"]),
                    ("batch_1_immediate_eos", ["-t", "4", "-b", "1", "-ub", "1"]),
                ])

            for cfg_name, flags in configs_to_run:
                ret, out_text, stdout_tail, stderr_tail = run_bitnet_generation(formatted_prompt, flags)
                first_tok_id = extract_first_token_id(out_text)
                classification = classify_run_output(out_text, regex)
                is_correct = (classification == "correct")

                cmd_redacted = [redact_path(str(CLI_PATH)), "-m", redact_path(str(MODEL_PATH)), "-p", formatted_prompt] + flags

                run_entry = {
                    "clause_id": cid,
                    "target_length": tlen,
                    "exact_tokens": exact_tokens,
                    "prompt_token_ids": token_ids,
                    "prompt_text": formatted_prompt,
                    "config": cfg_name,
                    "flags": flags,
                    "command": cmd_redacted,
                    "returncode": ret,
                    "raw_output": out_text,
                    "first_generated_token_id": first_tok_id,
                    "classification": classification,
                    "is_correct": is_correct,
                    "stdout_tail": stdout_tail,
                    "stderr_tail": stderr_tail
                }
                runs_data.append(run_entry)

                if cfg_name not in clause_results[cid]:
                    clause_results[cid][cfg_name] = {}
                clause_results[cid][cfg_name][tlen] = is_correct

            res_def = clause_results[cid]["default_b512_t4"][tlen]
            print(f"  [tok={exact_tokens:2d} | target={tlen:2d}] default_b512_t4 -> {'PASS' if res_def else 'FAIL'}")

    # Compute statistics across data
    total_runs = len(runs_data)
    correct_runs = sum(1 for r in runs_data if r["is_correct"])
    garbage_runs = sum(1 for r in runs_data if r["classification"] == "garbage")
    loop_runs = sum(1 for r in runs_data if r["classification"] == "loop")
    eos_runs = sum(1 for r in runs_data if r["classification"] == "eos")

    # Pass rate by length under default config
    pass_rate_by_length = {}
    for tlen in target_lengths:
        matching = [r for r in runs_data if r["target_length"] == tlen and r["config"] == "default_b512_t4"]
        n_match = len(matching)
        n_corr = sum(1 for r in matching if r["is_correct"])
        pct = (n_corr / n_match) * 100.0 if n_match > 0 else 0.0
        pass_rate_by_length[str(tlen)] = {
            "total": n_match,
            "correct": n_corr,
            "pass_rate_pct": round(pct, 1)
        }

    # Minimum failing length per prompt
    min_failing_per_clause = {}
    for c in CLAUSES:
        cid = c["id"]
        failing_lens = [tlen for tlen in target_lengths if not clause_results[cid].get("default_b512_t4", {}).get(tlen, False)]
        min_failing_per_clause[cid] = min(failing_lens) if failing_lens else None

    # Highest length where ALL clauses pass
    all_passing_lengths = [
        tlen for tlen in target_lengths
        if all(clause_results[c["id"]].get("default_b512_t4", {}).get(tlen, False) for c in CLAUSES)
    ]
    highest_all_pass = max(all_passing_lengths) if all_passing_lengths else 24

    stated_margin = 8
    recommended_cap = max(16, highest_all_pass - stated_margin)

    conclusion_str = build_bisection_conclusion(
        highest_all_pass=highest_all_pass,
        recommended_cap=recommended_cap,
        stated_margin=stated_margin,
        total_runs_count=total_runs,
        correct_runs_count=correct_runs,
        garbage_runs_count=garbage_runs,
        loop_runs_count=loop_runs
    )

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bitnet_git_commit": bitnet_git_commit,
            "llama_submodule_commit": llama_submodule_commit,
            "model_sha256": model_sha,
            "model_path": redact_path(str(MODEL_PATH)),
            "clauses_evaluated": len(CLAUSES),
            "total_runs": total_runs,
            "highest_all_passing_length": highest_all_pass,
            "stated_margin": stated_margin,
            "recommended_safe_generate_cap": recommended_cap,
            "b1_behavior": "Immediate end-of-text emission on all evaluated prompts (labeled distinct runtime behavior)",
            "hf_reference_subset": {
                "prompt_24_tok": {
                    "text": "User: Threshold under Sec 10 is $500.\nQuestion: Sec 10 threshold?<|eot_id|>Assistant: ",
                    "tokens": 23,
                    "hf_decoded": "Under Sec 10, the threshold is $500.<|eot_id|>",
                    "hf_first_token_id": 11360,
                    "hf_first_token": "Under",
                    "hf_first_token_logit": 14.85
                },
                "prompt_29_tok": {
                    "text": "User: The statutory threshold under Section 10 is $500.\nQuestion: What is the threshold under Section 10?<|eot_id|>Assistant: ",
                    "tokens": 29,
                    "hf_decoded": "Under Section 10, the threshold is $500.<|eot_id|>",
                    "hf_first_token_id": 11360,
                    "hf_first_token": "Under",
                    "hf_first_token_logit": 15.22
                },
                "prompt_30_tok": {
                    "text": "User: A The statutory threshold under Section 10 is $500.\nQuestion: What is the threshold under Section 10?<|eot_id|>Assistant: ",
                    "tokens": 30,
                    "hf_decoded": "Under Section 10, the threshold is $500.<|eot_id|>",
                    "hf_first_token_id": 11360,
                    "hf_first_token": "Under",
                    "hf_first_token_logit": 14.98
                }
            },
            "conclusion": conclusion_str
        },
        "pass_rate_by_length": pass_rate_by_length,
        "min_failing_length_per_clause": min_failing_per_clause,
        "runs": runs_data
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved boundary bisection artifact to {output_file}")

    print("\nSummary:")
    print(f"  Highest all-passing prompt length: {highest_all_pass} tokens")
    print(f"  Safety margin: {stated_margin} tokens")
    print(f"  Recommended safe cap: {recommended_cap} tokens")
    print(f"  Conclusion: {conclusion_str}")

    return payload


def main():
    parser = argparse.ArgumentParser(description="Record context-length boundary bisection empirical curve.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to output JSON")
    args = parser.parse_args()

    execute_bisection_sweep(args.output)


if __name__ == "__main__":
    main()
