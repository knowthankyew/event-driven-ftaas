#!/usr/bin/env python3
"""
Seed sample domain-specific dataset for FTaaS testing.
Generates disjoint training and validation JSONL datasets for Financial Earnings Sentiment & Extraction.
"""
import json
import os
from pathlib import Path

TRAIN_DATA = [
    {
        "prompt": "Analyze earnings report snippet: Q3 gross margin expanded 340 bps YoY to 43.1%, driven by favorable product mix and lower ocean freight costs. Operating income rose 18%.",
        "completion": "SENTIMENT: Positive | METRICS: Gross Margin +340bps (43.1%), Operating Income +18% | SUMMARY: Strong margin expansion propelled by freight tailwinds and product mix."
    },
    {
        "prompt": "Analyze earnings report snippet: Enterprise ARR grew 28% to $142M, while net revenue retention softened to 104% due to elongation in deal close cycles across EMEA.",
        "completion": "SENTIMENT: Neutral | METRICS: ARR +28% ($142M), NRR 104% | SUMMARY: Top-line growth remains solid but sales cycle friction in EMEA impacts retention."
    },
    {
        "prompt": "Analyze earnings report snippet: Operating expenses surged 42% due to aggressive R&D hiring, outpacing 9% revenue growth and compressing operating margin to negative 6.2%.",
        "completion": "SENTIMENT: Negative | METRICS: OpEx +42%, Revenue +9%, Operating Margin -6.2% | SUMMARY: Uncontrolled expense expansion severely pressured operating profitability."
    },
    {
        "prompt": "Analyze earnings report snippet: Free cash flow surged to $88M from $24M a year ago, reflecting disciplined working capital management and inventory normalization.",
        "completion": "SENTIMENT: Positive | METRICS: Free Cash Flow $88M (+266% YoY) | SUMMARY: Substantial cash generation supported by optimized working capital."
    },
    {
        "prompt": "Analyze earnings report snippet: Management trimmed full-year revenue guidance by $40M citing foreign exchange headwinds and reduced discretionary IT spending.",
        "completion": "SENTIMENT: Negative | METRICS: Full-Year Guidance -$40M | SUMMARY: Macro headwinds and IT budget tightening forced downside guidance revision."
    },
    {
        "prompt": "Analyze earnings report snippet: Customer count increased 14% to 12,400 with 68 customers now contributing over $1M in annualized billings.",
        "completion": "SENTIMENT: Positive | METRICS: Customers +14% (12,400), >$1M Customers: 68 | SUMMARY: Continued upward migration into large enterprise accounts."
    },
    {
        "prompt": "Analyze earnings report snippet: Supply chain bottlenecks delayed hardware shipments, resulting in a $15M revenue deferral into the subsequent fiscal quarter.",
        "completion": "SENTIMENT: Negative | METRICS: Revenue Deferral $15M | SUMMARY: Logistics disruption caused temporary top-line slippage into the following quarter."
    },
    {
        "prompt": "Analyze earnings report snippet: Cloud infrastructure segment grew 39% YoY to $310M, representing 52% of total consolidated revenue for the first time.",
        "completion": "SENTIMENT: Positive | METRICS: Cloud Revenue +39% ($310M, 52% share) | SUMMARY: Cloud transition milestone achieved with rapid high-margin revenue acceleration."
    },
    {
        "prompt": "Analyze earnings report snippet: Gross retention remained stable at 96%, though customer acquisition cost (CAC) increased 19% reflecting competitive bidding in digital channels.",
        "completion": "SENTIMENT: Neutral | METRICS: Gross Retention 96%, CAC +19% | SUMMARY: Churn remains tightly contained despite marketing efficiency headwinds."
    },
    {
        "prompt": "Analyze earnings report snippet: Restructuring charges of $22M were recognized as part of a 7% global workforce reduction designed to optimize long-term run-rate margins.",
        "completion": "SENTIMENT: Neutral | METRICS: Restructuring Charge $22M, Headcount -7% | SUMMARY: Short-term restructuring expenses undertaken to right-size operational cost structure."
    },
    {
        "prompt": "Analyze earnings report snippet: Subscription revenue grew 24% to $185M, accounting for 88% of total revenue compared to 81% in the prior year period.",
        "completion": "SENTIMENT: Positive | METRICS: Subscription Revenue +24% ($185M, 88% mix) | SUMMARY: High-margin recurring revenue stream continues to expand as a percentage of total mix."
    },
    {
        "prompt": "Analyze earnings report snippet: Days Sales Outstanding (DSO) deteriorated from 48 days to 63 days as key enterprise clients pushed payment terms past 90 days.",
        "completion": "SENTIMENT: Negative | METRICS: DSO increased from 48 to 63 days | SUMMARY: Working capital efficiency degraded due to enterprise receivables stretching."
    },
    {
        "prompt": "Analyze earnings report snippet: Cash reserves stood at $410M with zero outstanding debt under the revolving credit facility, leaving significant strategic flexibility.",
        "completion": "SENTIMENT: Positive | METRICS: Cash $410M, Debt: $0 | SUMMARY: Pristine balance sheet provides defensive buffer and capital allocation flexibility."
    },
    {
        "prompt": "Analyze earnings report snippet: Retail foot traffic declined 11% YoY while average basket size rose 6%, partially offsetting store volume weakness.",
        "completion": "SENTIMENT: Neutral | METRICS: Foot Traffic -11%, Basket Size +6% | SUMMARY: Pricing power and product bundling cushioned the impact of lower physical store traffic."
    },
    {
        "prompt": "Analyze earnings report snippet: Regulatory compliance costs rose $12M following new data residency laws in the EU and APAC regions.",
        "completion": "SENTIMENT: Negative | METRICS: Compliance Costs +$12M | SUMMARY: Jurisdictional regulatory compliance created unbudgeted structural overhead."
    },
    {
        "prompt": "Analyze earnings report snippet: Adjusted EBITDA reached $54M, surpassing consensus expectations of $47M by 15%, on disciplined operational execution.",
        "completion": "SENTIMENT: Positive | METRICS: Adj. EBITDA $54M (+15% vs consensus $47M) | SUMMARY: Substantial EBITDA beat driven by lean cost controls."
    },
    {
        "prompt": "Analyze earnings report snippet: Raw materials inflation pushed Cost of Goods Sold (COGS) up 16% YoY, depressing unit gross profit by 210 bps.",
        "completion": "SENTIMENT: Negative | METRICS: COGS +16%, Gross Profit -210bps | SUMMARY: Commodity input costs outpaced price adjustments, hurting unit economics."
    },
    {
        "prompt": "Analyze earnings report snippet: The Board authorized an accelerated $250M share repurchase program, signaling strong confidence in multi-year cash flow generation.",
        "completion": "SENTIMENT: Positive | METRICS: Share Repurchase $250M | SUMMARY: Proactive capital return program reflects management conviction in underlying cash generation."
    },
    {
        "prompt": "Analyze earnings report snippet: Services revenue contracted 5% as consulting engagements were postponed, though software license renewals remained resilient at 98%.",
        "completion": "SENTIMENT: Neutral | METRICS: Services -5%, Renewals 98% | SUMMARY: Discretionary professional services softened while core software commitments held firm."
    },
    {
        "prompt": "Analyze earnings report snippet: Net debt to EBITDA ratio improved from 2.8x to 1.9x following successful debt paydown of $130M from operating cash.",
        "completion": "SENTIMENT: Positive | METRICS: Net Debt/EBITDA 1.9x (from 2.8x), Debt Repayment $130M | SUMMARY: Meaningful deleveraging strengthens financial durability."
    }
]

# Strictly disjoint held-out validation dataset (never seen during training)
VAL_DATA = [
    {
        "prompt": "Analyze earnings report snippet: GAAP operating margin improved 480 bps to 22.4%, supported by automation in customer support and lower cloud egress fees.",
        "completion": "SENTIMENT: Positive | METRICS: Operating Margin +480bps (22.4%) | SUMMARY: Structural cost optimizations in support and cloud infrastructure drove substantial margin expansion."
    },
    {
        "prompt": "Analyze earnings report snippet: Revenue from the manufacturing vertical plunged 33% YoY following automotive plant shutdowns and project freezes.",
        "completion": "SENTIMENT: Negative | METRICS: Manufacturing Revenue -33% YoY | SUMMARY: Sector-wide industrial disruption caused steep revenue erosion in the manufacturing segment."
    },
    {
        "prompt": "Analyze earnings report snippet: Headcount grew 3% to 4,200 while average revenue per employee reached an all-time high of $340K.",
        "completion": "SENTIMENT: Positive | METRICS: Headcount +3% (4,200), Rev/Employee $340K | SUMMARY: High organizational leverage achieved with modest headcount expansion alongside record productivity."
    },
    {
        "prompt": "Analyze earnings report snippet: Average contract duration shortened from 36 months to 21 months as enterprise clients sought annual renewal flexibility.",
        "completion": "SENTIMENT: Neutral | METRICS: Contract Duration 21 months (vs 36 months prior) | SUMMARY: Enterprise commitment durations compressed to achieve budget flexibility without contract churn."
    },
    {
        "prompt": "Analyze earnings report snippet: Inventory write-downs of $18M were incurred due to obsolescence of legacy gen-1 telematics hardware.",
        "completion": "SENTIMENT: Negative | METRICS: Inventory Write-Down $18M | SUMMARY: Accelerated depreciation of legacy hardware models created one-time gross margin pressure."
    }
]

def seed(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_file = output_dir / "sample-financial-sentiment.jsonl"
    val_file = output_dir / "sample-financial-sentiment-val.jsonl"
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in TRAIN_DATA:
            f.write(json.dumps(item) + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for item in VAL_DATA:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Successfully seeded {len(TRAIN_DATA)} training records to {train_file}")
    print(f"✅ Successfully seeded {len(VAL_DATA)} disjoint validation records to {val_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent / "data" / "datasets"
    seed(base_dir)
