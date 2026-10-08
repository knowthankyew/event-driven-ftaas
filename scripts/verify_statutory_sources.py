#!/usr/bin/env python3
"""
verify_statutory_sources.py: Primary Source Verification Script v2.

Fetches statutory texts via GET (using eCFR versioned full-text API, legislation.gov.uk XML,
and statutory sources) and verifies authentic content against declared source facts.
Records raw response byte SHA-256 and extracted text SHA-256 into docs/verified_statutory_sources.json
and writes excerpt snapshots into docs/source_snapshots/.

State transitions:
- 'verified': HTTP 2xx, content_sha256 and extracted_text_sha256 computed, and all source_facts found.
- 'mismatch': HTTP 2xx, but required source_facts or verbatim text absent.
- 'unreachable': Non-2xx HTTP code or network exception (records exception class and message; no guessed cause).
- 'unverified': Initial or unattempted state.
"""

import argparse
import gzip
import hashlib
import json
import os
import re
import ssl
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Optional, Tuple

try:
    import certifi
    SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
except Exception:
    SSL_CONTEXT = ssl.create_default_context()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_FILE = REPO_ROOT / "docs" / "verified_statutory_sources.json"
DEFAULT_SNAPSHOT_DIR = REPO_ROOT / "docs" / "source_snapshots"

# 40 Authentic Statutory Sources with Primary URLs, Facts, and External Status Evidence
PRIMARY_SOURCES = [
    # 1. Regulation CC (12 CFR Part 229) - In Force
    {
        "id": "reg_cc_229_10_a",
        "statute": "12 CFR § 229.10(a)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "1988-09-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.10",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.10#p-229.10(a)",
        "citation": "12 CFR 229.10(a) Cash deposits",
        "chunk_type": "paraphrase",
        "source_facts": ["cash available for withdrawal", "business day after the banking day", "in person to an employee"],
        "legal_notes": None
    },
    {
        "id": "reg_cc_229_10_b",
        "statute": "12 CFR § 229.10(b)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "1988-09-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.10",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.10#p-229.10(b)",
        "citation": "12 CFR 229.10(b) Electronic payments",
        "chunk_type": "paraphrase",
        "source_facts": ["electronic payment", "business day after the banking day", "received"],
        "legal_notes": None
    },
    {
        "id": "reg_cc_229_10_c_treasury",
        "statute": "12 CFR § 229.10(c)(1)(i)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "1988-09-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.10",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.10#p-229.10(c)(1)(i)",
        "citation": "12 CFR 229.10(c)(1)(i) U.S. Treasury checks",
        "chunk_type": "paraphrase",
        "source_facts": ["Treasury of the United States", "payee", "business day"],
        "legal_notes": None
    },
    {
        "id": "reg_cc_229_10_c_cashier",
        "statute": "12 CFR § 229.10(c)(1)(v)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "1988-09-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.10",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.10#p-229.10(c)(1)(v)",
        "citation": "12 CFR 229.10(c)(1)(v) Cashier and teller checks",
        "chunk_type": "paraphrase",
        "source_facts": ["cashier", "teller", "certified", "business day"],
        "legal_notes": None
    },
    {
        "id": "reg_cc_229_10_c_aggregate",
        "statute": "12 CFR § 229.10(c)(1)(vi)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2025-07-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.10",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.10#p-229.10(c)(1)(vi)",
        "citation": "12 CFR 229.10(c)(1)(vi) $275 next-day availability minimum (amended 89 FR 54997)",
        "chunk_type": "paraphrase",
        "source_facts": ["$275", "lesser of", "aggregate amount deposited on any one banking day"],
        "legal_notes": "Statutory amount adjusted to $275 effective July 1, 2025 pursuant to 89 FR 54997."
    },
    {
        "id": "reg_cc_229_12_b",
        "statute": "12 CFR § 229.12(b)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "1990-09-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.12",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.12#p-229.12(b)",
        "citation": "12 CFR 229.12(b) Local check availability schedule",
        "chunk_type": "paraphrase",
        "source_facts": ["local check", "second business day following the banking day"],
        "legal_notes": None
    },
    {
        "id": "reg_cc_229_13_b",
        "statute": "12 CFR § 229.13(b)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2025-07-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.13",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.13#p-229.13(b)",
        "citation": "12 CFR 229.13(b) $6,725 large deposit exception threshold (amended 89 FR 54997)",
        "chunk_type": "paraphrase",
        "source_facts": ["$6,725", "large deposits", "aggregate amount"],
        "legal_notes": "Threshold adjusted to $6,725 effective July 1, 2025 pursuant to 89 FR 54997."
    },
    {
        "id": "reg_cc_229_13_d",
        "statute": "12 CFR § 229.13(d)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "1988-09-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.13",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.13#p-229.13(d)",
        "citation": "12 CFR 229.13(d) Repeated overdrafts exception",
        "chunk_type": "paraphrase",
        "source_facts": ["repeatedly overdrawn", "six months", "$6,725"],
        "legal_notes": None
    },
    {
        "id": "reg_cc_229_19_b",
        "statute": "12 CFR § 229.19(b)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "1988-09-01",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=229&section=229.19",
        "human_url": "https://www.ecfr.gov/current/title-12/chapter-II/subchapter-A/part-229/subpart-B/section-229.19#p-229.19(b)",
        "citation": "12 CFR 229.19(b) ATM exclusion from in-person employee deposit",
        "chunk_type": "paraphrase",
        "source_facts": ["atm", "business day", "availability"],
        "legal_notes": None
    },

    # 2. FTC ROSCA (15 U.S.C. § 8403) - In Force
    {
        "id": "rosca_8403_1",
        "statute": "15 U.S.C. § 8403(1)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2010-12-29",
        "source_url": "https://www.law.cornell.edu/uscode/text/15/8403",
        "human_url": "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section8403",
        "citation": "15 U.S.C. 8403(1) Clear and conspicuous disclosure of negative option terms",
        "chunk_type": "paraphrase",
        "source_facts": ["clear and conspicuous", "terms"],
        "legal_notes": None
    },
    {
        "id": "rosca_8403_2",
        "statute": "15 U.S.C. § 8403(2)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2010-12-29",
        "source_url": "https://www.law.cornell.edu/uscode/text/15/8403",
        "human_url": "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section8403",
        "citation": "15 U.S.C. 8403(2) Express informed consent before charging",
        "chunk_type": "paraphrase",
        "source_facts": ["express informed consent"],
        "legal_notes": None
    },
    {
        "id": "rosca_8403_3",
        "statute": "15 U.S.C. § 8403(3)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2010-12-29",
        "source_url": "https://www.law.cornell.edu/uscode/text/15/8403",
        "human_url": "https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title15-section8403",
        "citation": "15 U.S.C. 8403(3) Simple mechanism for cancellation",
        "chunk_type": "paraphrase",
        "source_facts": ["simple mechanism", "recurring"],
        "legal_notes": None
    },

    # 3. California Automatic Renewal Law (Bus. & Prof. Code §§ 17600–17606 & AB 2863) - In Force
    {
        "id": "ca_bpc_17602_a_1",
        "statute": "Cal. Bus. & Prof. Code § 17602(a)(1)",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2010-12-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "human_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "citation": "Cal. BPC 17602(a)(1) Presentation in visual proximity to consent",
        "chunk_type": "paraphrase",
        "source_facts": ["visual proximity", "affirmative consent", "terms"],
        "legal_notes": None
    },
    {
        "id": "ca_bpc_17602_a_2",
        "statute": "Cal. Bus. & Prof. Code § 17602(a)(2)",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2010-12-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "human_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "citation": "Cal. BPC 17602(a)(2) Affirmative consent required prior to charge",
        "chunk_type": "paraphrase",
        "source_facts": ["affirmative consent", "automatic renewal"],
        "legal_notes": None
    },
    {
        "id": "ca_bpc_17602_a_3",
        "statute": "Cal. Bus. & Prof. Code § 17602(a)(3)",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2010-12-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "human_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "citation": "Cal. BPC 17602(a)(3) Post-sale acknowledgment capable of retention",
        "chunk_type": "paraphrase",
        "source_facts": ["acknowledgment", "cancellation policy", "retention"],
        "legal_notes": None
    },
    {
        "id": "ca_bpc_17602_b",
        "statute": "Cal. Bus. & Prof. Code § 17602(b)",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2010-12-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "human_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17602",
        "citation": "Cal. BPC 17602(b) Clear and conspicuous notice of material change",
        "chunk_type": "paraphrase",
        "source_facts": ["material change", "clear and conspicuous"],
        "legal_notes": None
    },
    {
        "id": "ca_ab_2863_click_cancel",
        "statute": "Cal. Bus. & Prof. Code § 17602(a)(4)",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2025-07-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2863",
        "human_url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2863",
        "citation": "Cal. AB 2863 click-to-cancel statutory online mechanism (Cal. BPC 17602(a)(4))",
        "chunk_type": "paraphrase",
        "source_facts": ["click-to-cancel", "online mechanism", "cancel"],
        "legal_notes": "Added by Stats. 2024, Ch. 789 (AB 2863), effective July 1, 2025."
    },
    {
        "id": "ca_ab_2863_annual_notice",
        "statute": "Cal. Bus. & Prof. Code § 17602(d)",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2025-07-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2863",
        "human_url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2863",
        "citation": "Cal. AB 2863 annual reminder notice between 15 and 45 days prior (Cal. BPC 17602(d))",
        "chunk_type": "paraphrase",
        "source_facts": ["notice", "annual", "renewal"],
        "legal_notes": "Added by AB 2863, effective July 1, 2025."
    },
    {
        "id": "ca_ab_2863_free_trial",
        "statute": "Cal. Bus. & Prof. Code § 17602(c)",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2025-07-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2863",
        "human_url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2863",
        "citation": "Cal. AB 2863 free trial expiration notice (Cal. BPC 17602(c))",
        "chunk_type": "paraphrase",
        "source_facts": ["free gift", "trial", "notice"],
        "legal_notes": "Added by AB 2863, effective July 1, 2025."
    },
    {
        "id": "ca_bpc_17603_unconditional_gift",
        "statute": "Cal. Bus. & Prof. Code § 17603",
        "jurisdiction": "US_CA",
        "status": "in_force",
        "effective_from": "2010-12-01",
        "source_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17603",
        "human_url": "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml?lawCode=BPC&sectionNum=17603",
        "citation": "Cal. BPC 17603 Goods sent without affirmative consent deemed unconditional gifts",
        "chunk_type": "paraphrase",
        "source_facts": ["unconditional gift", "affirmative consent"],
        "legal_notes": None
    },

    # 4. FTC Negative Option Rule (16 CFR Part 425) - Vacated by 8th Cir.
    {
        "id": "ftc_negative_option_equal",
        "statute": "16 CFR § 425.5",
        "jurisdiction": "US_FEDERAL",
        "status": "vacated_by_court_order",
        "effective_from": "2024-10-16 (promulgated); vacated 2025-07-08",
        "source_url": "https://www.ftc.gov/legal-library/browse/rules/negative-option-rule",
        "human_url": "https://www.ftc.gov/legal-library/browse/rules/negative-option-rule",
        "citation": "16 CFR 425.5 Symmetric cancellation method (Vacated by 8th Cir. in Custom Communications, Inc. v. FTC, No. 24-3137)",
        "chunk_type": "paraphrase",
        "source_facts": ["negative option", "cancel", "easy"],
        "legal_notes": "Vacated nationwide by Eighth Circuit on July 8, 2025 (Custom Communications, Inc. v. FTC, No. 24-3137, consolidated with No. 24-3388). 1973 rule remains in effect; FTC issued ANPRM in March 2026.",
        "status_evidence": {
            "url": "https://ecf.ca8.uscourts.gov/opndir/25/07/243137P.pdf",
            "retrieved_at": "2026-10-08T11:54:00Z",
            "evidence_sha256": "4b684206584cba36b94e3575916323630f9a2b5358046bcf3a1eecce70d069b2",
            "note": "Per curiam opinion in Custom Communications, Inc. v. FTC, No. 24-3137 (8th Cir. July 8, 2025) vacating rule amendments in their entirety."
        }
    },
    {
        "id": "ftc_negative_option_save",
        "statute": "16 CFR § 425.6",
        "jurisdiction": "US_FEDERAL",
        "status": "vacated_by_court_order",
        "effective_from": "2024-10-16 (promulgated); vacated 2025-07-08",
        "source_url": "https://www.ftc.gov/legal-library/browse/rules/negative-option-rule",
        "human_url": "https://www.ftc.gov/legal-library/browse/rules/negative-option-rule",
        "citation": "16 CFR 425.6 Save attempts restriction (Vacated by 8th Cir. in Custom Communications, Inc. v. FTC, No. 24-3137)",
        "chunk_type": "paraphrase",
        "source_facts": ["save", "negative option", "cancel"],
        "legal_notes": "Vacated nationwide by Eighth Circuit on July 8, 2025 (Custom Communications, Inc. v. FTC, No. 24-3137).",
        "status_evidence": {
            "url": "https://ecf.ca8.uscourts.gov/opndir/25/07/243137P.pdf",
            "retrieved_at": "2026-10-08T11:54:00Z",
            "evidence_sha256": "4b684206584cba36b94e3575916323630f9a2b5358046bcf3a1eecce70d069b2",
            "note": "Per curiam opinion in Custom Communications, Inc. v. FTC, No. 24-3137 (8th Cir. July 8, 2025) vacating rule amendments."
        }
    },
    {
        "id": "ftc_negative_option_misrep",
        "statute": "16 CFR § 425.3",
        "jurisdiction": "US_FEDERAL",
        "status": "vacated_by_court_order",
        "effective_from": "2024-10-16 (promulgated); vacated 2025-07-08",
        "source_url": "https://www.ftc.gov/legal-library/browse/rules/negative-option-rule",
        "human_url": "https://www.ftc.gov/legal-library/browse/rules/negative-option-rule",
        "citation": "16 CFR 425.3 Prohibition of misrepresentations (Vacated by 8th Cir. in Custom Communications, Inc. v. FTC, No. 24-3137)",
        "chunk_type": "paraphrase",
        "source_facts": ["misrepresent", "negative option"],
        "legal_notes": "Vacated nationwide by Eighth Circuit on July 8, 2025 (Custom Communications, Inc. v. FTC, No. 24-3137).",
        "status_evidence": {
            "url": "https://ecf.ca8.uscourts.gov/opndir/25/07/243137P.pdf",
            "retrieved_at": "2026-10-08T11:54:00Z",
            "evidence_sha256": "4b684206584cba36b94e3575916323630f9a2b5358046bcf3a1eecce70d069b2",
            "note": "Per curiam opinion in Custom Communications, Inc. v. FTC, No. 24-3137 (8th Cir. July 8, 2025) vacating rule amendments."
        }
    },

    # 5. UK Digital Markets, Competition and Consumers Act 2024 (DMCC Part 4, Chapter 2) - Enacted, Not In Force
    {
        "id": "uk_dmcc_precontract_info",
        "statute": "UK DMCC Act 2024 s. 255",
        "jurisdiction": "UK",
        "status": "enacted_not_in_force",
        "effective_from": "Enacted 2024-05-24; commencement scheduled January 2027",
        "source_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/255/data.xml",
        "human_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/255",
        "citation": "UK DMCC Act 2024 s. 255 Pre-contract information for subscription contracts (Enacted, not yet commenced)",
        "chunk_type": "paraphrase",
        "source_facts": ["pre-contract information", "subscription contract", "trader"],
        "legal_notes": "Enacted May 24, 2024. UK DBT announced January 2027 commencement for subscription regime.",
        "status_evidence": {
            "url": "https://www.legislation.gov.uk/ukpga/2024/13/contents",
            "retrieved_at": "2026-10-08T11:54:00Z",
            "evidence_sha256": "81f185db2d6e382d6b38c2ea30588820c22fa59dfc35b62b7ae1e44f80c6f2a6",
            "note": "Official statutory text enacted May 24, 2024; subscription provisions uncommenced."
        },
        "expected_commencement": {
            "date": "2027-01",
            "source": "UK Department for Business and Trade announcement (August 2026)"
        }
    },
    {
        "id": "uk_dmcc_cooling_off",
        "statute": "UK DMCC Act 2024 s. 256",
        "jurisdiction": "UK",
        "status": "enacted_not_in_force",
        "effective_from": "Enacted 2024-05-24; commencement scheduled January 2027",
        "source_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/256/data.xml",
        "human_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/256",
        "citation": "UK DMCC Act 2024 s. 256 14-day cooling-off rights to cancel subscription contracts (Enacted, not yet commenced)",
        "chunk_type": "paraphrase",
        "source_facts": ["cooling-off", "14 days", "cancel"],
        "legal_notes": "Enacted May 24, 2024. UK DBT announced January 2027 commencement for subscription regime.",
        "status_evidence": {
            "url": "https://www.legislation.gov.uk/ukpga/2024/13/contents",
            "retrieved_at": "2026-10-08T11:54:00Z",
            "evidence_sha256": "81f185db2d6e382d6b38c2ea30588820c22fa59dfc35b62b7ae1e44f80c6f2a6",
            "note": "Official statutory text enacted May 24, 2024."
        },
        "expected_commencement": {
            "date": "2027-01",
            "source": "UK Department for Business and Trade announcement (August 2026)"
        }
    },
    {
        "id": "uk_dmcc_renewal_notice",
        "statute": "UK DMCC Act 2024 s. 257",
        "jurisdiction": "UK",
        "status": "enacted_not_in_force",
        "effective_from": "Enacted 2024-05-24; commencement scheduled January 2027",
        "source_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/257/data.xml",
        "human_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/257",
        "citation": "UK DMCC Act 2024 s. 257 Pre-renewal reminder notices (Enacted, not yet commenced)",
        "chunk_type": "paraphrase",
        "source_facts": ["reminder notice", "renewal", "subscription contract"],
        "legal_notes": "Enacted May 24, 2024. UK DBT announced January 2027 commencement for subscription regime.",
        "status_evidence": {
            "url": "https://www.legislation.gov.uk/ukpga/2024/13/contents",
            "retrieved_at": "2026-10-08T11:54:00Z",
            "evidence_sha256": "81f185db2d6e382d6b38c2ea30588820c22fa59dfc35b62b7ae1e44f80c6f2a6",
            "note": "Official statutory text enacted May 24, 2024."
        },
        "expected_commencement": {
            "date": "2027-01",
            "source": "UK Department for Business and Trade announcement (August 2026)"
        }
    },
    {
        "id": "uk_dmcc_exit_steps",
        "statute": "UK DMCC Act 2024 s. 258",
        "jurisdiction": "UK",
        "status": "enacted_not_in_force",
        "effective_from": "Enacted 2024-05-24; commencement scheduled January 2027",
        "source_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/258/data.xml",
        "human_url": "https://www.legislation.gov.uk/ukpga/2024/13/section/258",
        "citation": "UK DMCC Act 2024 s. 258 Single straightforward exit communication (Enacted, not yet commenced)",
        "chunk_type": "paraphrase",
        "source_facts": ["straightforward", "exit", "cancel"],
        "legal_notes": "Enacted May 24, 2024. UK DBT announced January 2027 commencement for subscription regime.",
        "status_evidence": {
            "url": "https://www.legislation.gov.uk/ukpga/2024/13/contents",
            "retrieved_at": "2026-10-08T11:54:00Z",
            "evidence_sha256": "81f185db2d6e382d6b38c2ea30588820c22fa59dfc35b62b7ae1e44f80c6f2a6",
            "note": "Official statutory text enacted May 24, 2024."
        },
        "expected_commencement": {
            "date": "2027-01",
            "source": "UK Department for Business and Trade announcement (August 2026)"
        }
    },

    # 6. EU Consumer Rights Directive (Directive 2011/83/EU) - In Force
    {
        "id": "eu_crd_art_8_button",
        "statute": "EU Directive 2011/83/EU Art. 8",
        "jurisdiction": "EU",
        "status": "in_force",
        "effective_from": "2014-06-13",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32011L0083",
        "human_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32011L0083",
        "citation": "EU Directive 2011/83/EU Art. 8 Order with obligation to pay button label mandate",
        "chunk_type": "paraphrase",
        "source_facts": ["order with obligation to pay", "button", "pay"],
        "legal_notes": None
    },
    {
        "id": "eu_crd_art_9_withdrawal",
        "statute": "EU Directive 2011/83/EU Art. 9",
        "jurisdiction": "EU",
        "status": "in_force",
        "effective_from": "2014-06-13",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32011L0083",
        "human_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32011L0083",
        "citation": "EU Directive 2011/83/EU Art. 9 14 calendar days statutory right of withdrawal",
        "chunk_type": "paraphrase",
        "source_facts": ["14 days", "withdrawal", "contract"],
        "legal_notes": None
    },
    {
        "id": "eu_crd_art_14_refund",
        "statute": "EU Directive 2011/83/EU Art. 14",
        "jurisdiction": "EU",
        "status": "in_force",
        "effective_from": "2014-06-13",
        "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32011L0083",
        "human_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32011L0083",
        "citation": "EU Directive 2011/83/EU Art. 14 14-day reimbursement obligation",
        "chunk_type": "paraphrase",
        "source_facts": ["14 days", "reimburse", "payments"],
        "legal_notes": None
    },

    # 7. State Automatic Renewal Laws (NY, IL, CO, VA, DC) - In Force
    {
        "id": "ny_gbl_527_a",
        "statute": "N.Y. Gen. Bus. Law § 527-a",
        "jurisdiction": "US_NY",
        "status": "in_force",
        "effective_from": "2021-02-09",
        "source_url": "https://www.nysenate.gov/legislation/laws/GBS/527-A",
        "human_url": "https://www.nysenate.gov/legislation/laws/GBS/527-A",
        "citation": "N.Y. GBL 527-a Mandatory online cancellation option",
        "chunk_type": "paraphrase",
        "source_facts": ["online cancellation", "consumer", "automatic renewal"],
        "legal_notes": None
    },
    {
        "id": "il_815_ilcs_601",
        "statute": "815 Ill. Comp. Stat. 601/10",
        "jurisdiction": "US_IL",
        "status": "in_force",
        "effective_from": "2000-01-01",
        "source_url": "https://www.ilga.gov/legislation/ilcs/ilcs3.asp?ActID=2334",
        "human_url": "https://www.ilga.gov/legislation/ilcs/ilcs3.asp?ActID=2334",
        "citation": "815 ILCS 601/10 30 to 60 days advance written reminder notice",
        "chunk_type": "paraphrase",
        "source_facts": ["30", "60", "written notice", "renewal"],
        "legal_notes": None
    },
    {
        "id": "co_hb_21_1239",
        "statute": "Colo. Rev. Stat. § 6-1-732",
        "jurisdiction": "US_CO",
        "status": "in_force",
        "effective_from": "2022-01-01",
        "source_url": "https://leg.colorado.gov/bills/hb21-1239",
        "human_url": "https://leg.colorado.gov/bills/hb21-1239",
        "citation": "Colo. Rev. Stat. 6-1-732 Colorado subscription disclosures and online link",
        "chunk_type": "paraphrase",
        "source_facts": ["Colorado", "subscription", "notice"],
        "legal_notes": None
    },
    {
        "id": "va_code_59_1_207",
        "statute": "Va. Code § 59.1-207.46",
        "jurisdiction": "US_VA",
        "status": "in_force",
        "effective_from": "2019-07-01",
        "source_url": "https://law.lis.virginia.gov/vacode/title59.1/chapter17.7/section59.1-207.46/",
        "human_url": "https://law.lis.virginia.gov/vacode/title59.1/chapter17.7/section59.1-207.46/",
        "citation": "Va. Code 59.1-207.46 Express verifiable consent and cancellation instructions",
        "chunk_type": "paraphrase",
        "source_facts": ["verifiable consent", "automatic renewal", "cancellation"],
        "legal_notes": None
    },
    {
        "id": "dc_code_28_3872",
        "statute": "D.C. Code § 28-3872",
        "jurisdiction": "US_DC",
        "status": "in_force",
        "effective_from": "2019-03-13",
        "source_url": "https://code.dccouncil.gov/us/dc/council/code/sections/28-3872",
        "human_url": "https://code.dccouncil.gov/us/dc/council/code/sections/28-3872",
        "citation": "D.C. Code 28-3872 Notice between 30 and 60 days prior to renewal",
        "chunk_type": "paraphrase",
        "source_facts": ["30", "60 days", "renewal notice"],
        "legal_notes": None
    },

    # 8. Federal Banking / Electronic Funds Transfer Act (Reg E - 12 CFR Part 1005) - In Force
    {
        "id": "reg_e_1005_10",
        "statute": "12 CFR § 1005.10(b)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2011-12-30",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=1005&section=1005.10",
        "human_url": "https://www.consumerfinance.gov/rules-policy/regulations/1005/10/#b",
        "citation": "12 CFR 1005.10(b) Signed written authorization for recurring electronic debits",
        "chunk_type": "paraphrase",
        "source_facts": ["preauthorized electronic fund transfers", "signed or similarly authenticated"],
        "legal_notes": None
    },
    {
        "id": "reg_e_1005_10_c",
        "statute": "12 CFR § 1005.10(c)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2011-12-30",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=1005&section=1005.10",
        "human_url": "https://www.consumerfinance.gov/rules-policy/regulations/1005/10/#c",
        "citation": "12 CFR 1005.10(c) Right to stop payment up to three days before scheduled date",
        "chunk_type": "paraphrase",
        "source_facts": ["stop payment", "three business days", "preauthorized"],
        "legal_notes": None
    },
    {
        "id": "reg_e_1005_11",
        "statute": "12 CFR § 1005.11(a)",
        "jurisdiction": "US_FEDERAL",
        "status": "in_force",
        "effective_from": "2011-12-30",
        "source_url": "https://www.ecfr.gov/api/versioner/v1/full/2025-07-01/title-12.xml?part=1005&section=1005.11",
        "human_url": "https://www.consumerfinance.gov/rules-policy/regulations/1005/11/#a",
        "citation": "12 CFR 1005.11(a) 10 business day billing error investigation period",
        "chunk_type": "paraphrase",
        "source_facts": ["10 business days", "error", "investigat"],
        "legal_notes": None
    },

    # 9. Uniform Commercial Code (UCC Article 4) - In Force
    {
        "id": "ucc_4_403",
        "statute": "UCC § 4-403",
        "jurisdiction": "US_STATE_MODEL",
        "status": "in_force",
        "effective_from": "1990-01-01",
        "source_url": "https://www.law.cornell.edu/ucc/4/4-403",
        "human_url": "https://www.law.cornell.edu/ucc/4/4-403",
        "citation": "UCC 4-403 Customer right to order stop payment on items",
        "chunk_type": "paraphrase",
        "source_facts": ["stop payment", "customer", "bank"],
        "legal_notes": None
    },
    {
        "id": "ucc_4_406",
        "statute": "UCC § 4-406",
        "jurisdiction": "US_STATE_MODEL",
        "status": "in_force",
        "effective_from": "1990-01-01",
        "source_url": "https://www.law.cornell.edu/ucc/4/4-406",
        "human_url": "https://www.law.cornell.edu/ucc/4/4-406",
        "citation": "UCC 4-406 Customer duty to discover and report unauthorized signatures",
        "chunk_type": "paraphrase",
        "source_facts": ["customer", "unauthorized signature", "statement"],
        "legal_notes": None
    }
]


def default_http_get_fetcher(url: str, timeout: float = 10.0) -> Tuple[int, bytes, str, dict]:
    """
    Default HTTP GET fetcher sending User-Agent and compression headers.
    Returns (status_code, raw_response_bytes, final_url, headers).
    Raises exception on network or protocol failure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
        status_code = resp.status
        final_url = resp.geturl()
        raw_bytes = resp.read()
        headers_dict = dict(resp.headers)
        return status_code, raw_bytes, final_url, headers_dict


def extract_text_from_bytes(raw_bytes: bytes, headers: Optional[dict] = None) -> str:
    """Decompress and extract text from HTML or XML response bytes."""
    data = raw_bytes
    if headers and headers.get("Content-Encoding") == "gzip" or (len(data) > 2 and data[:2] == b"\x1f\x8b"):
        try:
            data = gzip.decompress(data)
        except Exception:
            pass

    text = data.decode("utf-8", errors="replace")

    # If XML, extract text nodes
    if "<DIV" in text or "<Legislation" in text or "<?xml" in text:
        try:
            root = ET.fromstring(text)
            extracted = " ".join("".join(root.itertext()).split())
            if extracted:
                return extracted
        except Exception:
            pass

    # Clean HTML tags and normalize whitespace
    cleaned = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return cleaned


def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase for containment checks."""
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_single_source(
    source: dict,
    fetcher: Callable[[str], Tuple[int, bytes, str, dict]],
    snapshot_dir: Optional[Path] = None,
) -> dict:
    """
    Verify a single statutory source against authoritative GET response.
    Returns record dictionary with strict fields.
    """
    source_id = source["id"]
    url = source["source_url"]
    attempt_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    record = {
        "id": source_id,
        "statute": source["statute"],
        "jurisdiction": source["jurisdiction"],
        "status": source["status"],
        "effective_from": source["effective_from"],
        "source_url": url,
        "human_url": source.get("human_url", url),
        "citation": source["citation"],
        "chunk_type": source.get("chunk_type", "paraphrase"),
        "source_facts": source.get("source_facts", []),
        "last_attempted_on": attempt_time,
        "final_url": None,
        "http_status": None,
        "content_sha256": None,
        "extracted_text_sha256": None,
        "state": "unverified",
        "verified_on": None,
        "failure_error": None,
        "legal_notes": source.get("legal_notes")
    }

    if "status_evidence" in source:
        record["status_evidence"] = source["status_evidence"]
    if "expected_commencement" in source:
        record["expected_commencement"] = source["expected_commencement"]

    try:
        status_code, raw_bytes, final_url, headers = fetcher(url)
        record["http_status"] = status_code
        record["final_url"] = final_url

        if status_code < 200 or status_code >= 300:
            record["state"] = "unreachable"
            record["failure_error"] = {
                "class": "HTTPError",
                "message": f"HTTP status code {status_code}"
            }
            return record

        record["content_sha256"] = hashlib.sha256(raw_bytes).hexdigest()

        # Extract text
        extracted_text = extract_text_from_bytes(raw_bytes, headers)
        text_sha = hashlib.sha256(extracted_text.encode("utf-8")).hexdigest()
        record["extracted_text_sha256"] = text_sha

        # Save snapshot excerpt if snapshot_dir provided
        if snapshot_dir:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            snapshot_file = snapshot_dir / f"{text_sha}.txt"
            if not snapshot_file.is_file():
                header = (
                    f"# Source Attribution: {source_id} - {source['statute']}\n"
                    f"# Source URL: {url}\n"
                    f"# Extracted SHA-256: {text_sha}\n"
                    f"# Retrieved At: {attempt_time}\n\n"
                )
                # Store excerpt (up to 4000 characters) to respect reuse terms
                excerpt = extracted_text[:4000]
                snapshot_file.write_text(header + excerpt, encoding="utf-8")

        # Verify facts or verbatim containment
        chunk_type = source.get("chunk_type", "paraphrase")
        source_facts = source.get("source_facts", [])
        norm_extracted = normalize_text(extracted_text)

        facts_present = True
        missing_facts = []

        if chunk_type == "verbatim":
            verbatim_text = source.get("text", "")
            if normalize_text(verbatim_text) not in norm_extracted:
                facts_present = False
                missing_facts.append("verbatim_containment")
        else:
            for fact in source_facts:
                if normalize_text(fact) not in norm_extracted:
                    facts_present = False
                    missing_facts.append(fact)

        if facts_present and len(source_facts) > 0:
            record["state"] = "verified"
            record["verified_on"] = attempt_time
        else:
            record["state"] = "mismatch"
            record["failure_error"] = {
                "class": "ContentMismatchError",
                "message": f"Missing required source facts: {missing_facts}"
            }

    except urllib.error.HTTPError as e:
        record["http_status"] = e.code
        record["state"] = "unreachable"
        record["failure_error"] = {
            "class": "HTTPError",
            "message": f"HTTP Error {e.code}: {e.reason}"
        }
    except urllib.error.URLError as e:
        record["http_status"] = None
        record["state"] = "unreachable"
        record["failure_error"] = {
            "class": "URLError",
            "message": str(e.reason)
        }
    except Exception as e:
        record["http_status"] = None
        record["state"] = "unreachable"
        record["failure_error"] = {
            "class": type(e).__name__,
            "message": str(e)
        }

    return record


def verify_all_sources(
    sources: Optional[list] = None,
    fetcher: Optional[Callable] = None,
    snapshot_dir: Optional[Path] = DEFAULT_SNAPSHOT_DIR,
    output_file: Optional[Path] = None,
) -> dict:
    """Verify all sources, compute status breakdown, and optionally save JSON artifact."""
    target_sources = sources if sources is not None else PRIMARY_SOURCES
    actual_fetcher = fetcher if fetcher is not None else default_http_get_fetcher

    records = []
    print(f"=== Verifying {len(target_sources)} Statutory Primary Sources via GET ===")

    for idx, s in enumerate(target_sources, 1):
        rec = verify_single_source(s, actual_fetcher, snapshot_dir=snapshot_dir)
        records.append(rec)
        print(f"[{idx:2d}/{len(target_sources)}] {s['statute']:<28} | State: {rec['state']:<12} | HTTP: {rec['http_status']}")

    status_counts = {"verified": 0, "unreachable": 0, "mismatch": 0, "unverified": 0}
    legal_status_counts = {}
    for r in records:
        st = r["state"]
        status_counts[st] = status_counts.get(st, 0) + 1
        lst = r["status"]
        legal_status_counts[lst] = legal_status_counts.get(lst, 0) + 1

    payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_statutory_sources": len(records),
            "verification_state_breakdown": status_counts,
            "status_breakdown": legal_status_counts,
            "verification_tool": "scripts/verify_statutory_sources.py",
            "verification_version": "2.0.0"
        },
        "sources": records
    }

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved verified sources artifact to {output_file}")

    print(f"Verification summary: {status_counts}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Verify statutory primary sources via GET requests.")
    parser.add_argument("--run", action="store_true", help="Run live verification against configured endpoints")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE, help="Path to output JSON")
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT_DIR, help="Path to snapshot directory")

    args = parser.parse_args()
    verify_all_sources(snapshot_dir=args.snapshots, output_file=args.output)


if __name__ == "__main__":
    main()
