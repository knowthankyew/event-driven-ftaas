# FTaaS Enterprise Studio — User Guide
### *A Practical Guide for Technical Leads, Product Managers & Operations Teams (No Coding Required)*

---

## 🌟 What is FTaaS Enterprise Studio?

Most off-the-shelf AI models (like generic ChatGPT or baseline open-source LLMs) generate **generic, one-size-fits-all responses**. They don't know your organization's internal acronyms, SLA requirements, or mandatory compliance phrasing (such as deposit hold policies or statutory notices).

**FTaaS (Fine-Tuning as a Service) Enterprise Studio** is a zero-code web application that enables non-technical business teams to:
1. **See the difference**: Compare a generic AI side-by-side with a customized, policy-enforced AI model tailored to your department.
2. **Train custom team adapters**: Define company Q&A policies in a spreadsheet-like editor or upload a CSV file—no Python, machine learning, or code required.
3. **Audit policy adherence**: Check whether required policy tags and disclaimers appear in the generated output.

---

## 📋 System Prerequisites

To launch the studio, you only need one software component installed on your computer:
- **.NET 10 SDK** (Available for free on macOS, Windows, and Linux from [dotnet.microsoft.com/download](https://dotnet.microsoft.com/download)).

*(Note: If you only want to explore the interface, test dataset creation, and evaluate policy formatting, **Docker is NOT required**. Docker and Python are only needed if you want to run the full, live GPU training pipeline on your machine. See [Modes of Operation](#-two-modes-of-operation-preview-vs-live-compute) below).*

---

## 🚀 How to Launch in 1 Step

Open your Terminal (macOS Terminal, iTerm, or PowerShell) in this repository directory and run:

```bash
./scripts/start-studio.sh
```

### What happens next?
1. The script verifies that .NET 10 is available.
2. It builds and starts the local web server in 2–3 seconds.
3. Your default web browser will automatically open to: **[http://localhost:5100](http://localhost:5100)**.
4. **To stop the studio when you're done:** Switch back to your terminal window and press `Ctrl + C`.

> [!TIP]
> **Want to watch a demo first?** Check out [`demo.mp4`](demo.mp4) for a pre-recorded video walkthrough, or generate a fresh video anytime with `./scripts/record-demo.sh`.

---

## ⚙️ Two Modes of Operation: Preview vs. Live Compute

To maintain complete transparency regarding model execution, the studio supports two distinct modes:

| Mode | What It Does | Hardware & Infra Needed | When to Use |
|---|---|---|---|
| **🟡 Interactive Preview Mode** *(Default)* | Displays pre-formatted, realistic domain examples to demonstrate formatting, disclaimer inclusion, and policy tagging without spinning up heavy machine learning compute. | Just `.NET 10 SDK` (No Docker, no Python, no GPU required) | Initial product demonstrations, dataset curation, and UI exploration. |
| **🟢 Live Compute Mode** | Dynamically routes prompts through PyTorch model weights on your Apple Silicon (MPS) or NVIDIA GPU, generating live tokens in real time. | Docker (RabbitMQ + MLflow) and Python ML Worker active | Full end-to-end local training and live inference testing. |

> [!NOTE]
> When running in **Preview Mode**, the studio prominently displays a yellow banner labeled *"🟡 Interactive Preview Mode Active"* above the comparison cards so there is never confusion about whether compute is simulated or live.

---

## 🧭 Visual Tour of the Studio

The web application is divided into 4 main tabs at the top of your screen:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚡ Comparison Arena   │   🛠️ Adapter Studio   │   🔄 Event Pipeline   │   📚 Adapter Library │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 1: ⚡ Comparison Arena (Side-by-Side Evaluation)

This is where you observe the business impact of team-specific customization:

1. **Select a Business Persona**:
   - **Fintech Support & Compliance**: Handles money transfers, deposit holds, and strict regulatory notices.
   - **Enterprise SaaS Customer Ops**: Handles enterprise SLAs, cancellation terms, and internal ticketing tags.
   - **Financial Earnings & SEC**: Extracts revenue metrics, margins, and basis points from earnings transcripts.
2. **Choose an Inquiry**:
   - Click one of the sample prompt chips (e.g. `ACH Deposit Clearance Delay` or `Contract Cancellation Past 30 Days`).
3. **Click "Compare Models"**:
   - **Left Panel (Generic AI)**: Shows how a generic foundation model responds. Notice it gives polite but vague answers, omits policy thresholds, and forgets mandatory legal disclaimers.
   - **Right Panel (Company Custom AI)**: Shows how a specialized model responds. Notice it cites exact policy clauses (e.g. *Regulation CC $10,000 threshold* or *Section 8.2 of Enterprise MSA*), adds automated ticket tags (`[ACH-HELD-VERIFY]`), and embeds required disclaimers.
4. **Team Policy & Disclaimer Match Inspector**:
   - Shows checkmarks (✓) for each company policy rule detected in the answer.

> [!IMPORTANT]
> **Policy Audit Disclaimer**: The compliance inspector is an automated pattern- and keyword-matching tool that checks whether your designated training rules appear in the model output. It is intended for policy demonstration and quality assurance; it does **not** constitute independent legal advice or statutory regulatory certification.

---

### Tab 2: 🛠️ No-Code Adapter Studio (Train Your Assistant)

Use this tab to create or modify custom rules for your team:

> [!CAUTION]
> **Data Privacy Notice & Guardrail Scope**: Do **NOT** paste real customer Personally Identifiable Information (PII), real credit card/bank account numbers, or real patient health records into sample training tables or upload files. Always use fictitious or sanitized data.
>
> *Note on Guardrail Capabilities*: The Studio and server ingestion gateway run an automated heuristic scanner to block delimited Social Security Numbers (`XXX-XX-XXXX`), contextual SSNs (`SSN: XXXXXXXXX`), and Luhn-valid credit card numbers across all columns (including unmapped metadata). However, this is a **best-effort safety net, not an exhaustive Data Loss Prevention (DLP) certification tool**. Raw, uncontextualized 9-digit numbers or non-standard formats will not be detected. Never treat a passing scan as proof that a dataset is safe; company data sanitization policies remain mandatory.

1. **Add Training Examples**:
   - Click **"Load Persona Policy Examples"** to pre-populate sample rules, or click **"+ Add Example Row"** to type your own.
   - **Inquiry (Prompt)**: The question an employee or customer asks (e.g., *"Can I cancel after 45 days?"*).
   - **Compliant Response (Completion)**: The exact answer, policy citation, and disclaimer you want the AI to give.
2. **Or Upload an Existing File**:
   - Drag and drop a `.csv` or `.jsonl` document from your computer into the upload box.
   - *Dataset Guidelines*: LoRA policy adapters typically only need 20 to 500 focused examples for high-quality convergence. The system enforces a **25 MB file size ceiling** and a **50,000 record maximum** to maintain optimal server performance and memory safety.
3. **Deploy Team Adapter**:
   - Give your model a name (e.g., `support-team-v1`).
   - Click **"Train & Deploy Team Adapter"**.
   - The platform accepts your job immediately without freezing your browser and switches you to the Event Pipeline view to monitor background processing.

---

### Tab 3: 🔄 Event Pipeline (How It Works Behind the Scenes)

If you need to explain the system to an executive, engineering director, or client, this tab provides an animated visual diagram:

1. **Business Lead (You)**: Submits 20–50 policy examples through the web interface.
2. **Ingestion Gateway (.NET 10)**: Instantly accepts the job in milliseconds so the web browser never hangs waiting on machine learning compute.
3. **Message Broker (RabbitMQ)**: Sends a lightweight pointer to the dataset, preventing system bottlenecks.
4. **ML Worker (PyTorch / LoRA)**: Trains a tiny **~1.8 megabyte adapter file** instead of duplicating a massive 7 gigabyte model.
   - *Training Time*: ~2–3 minutes on Apple Silicon (M1–M4) or an NVIDIA GPU (~10–15 minutes on standard CPU-only machines).
5. **Dynamic Serving Engine**: Keeps one core model warm in memory and mounts your team's adapter on the fly in milliseconds.

---

### Tab 4: 📚 Adapter Library (Active Models)

- Shows all trained company models stored on your machine.
- Displays adapter footprint (~1.8 MB), creation date, and status (*"Ready to Serve"*).
- Click **"⚡ Load in Arena"** on any model card to immediately test it in the Comparison Arena.

---

## ❓ Frequently Asked Questions (FAQ)

#### Q: Do I need Docker running to open the studio?
**A:** No. You can launch `./scripts/start-studio.sh` with only the .NET 10 SDK installed. The studio will start immediately in **Interactive Preview Mode**, allowing you to explore the interface, build datasets, and evaluate side-by-side policy formatting without spinning up Docker containers. To enable **Live Compute Mode** with live on-device GPU inference, start Docker and the Python services via `./scripts/dev-up.sh`.

#### Q: Why does the app emphasize "1.8 MB Adapter"?
**A:** Traditional AI fine-tuning requires copying and saving the entire model (often 7 to 70 Gigabytes), which requires expensive dedicated GPUs and massive disk storage. FTaaS uses **LoRA (Low-Rank Adaptation)**, which freezes the base model and only trains a tiny mathematical adapter layer (~1.8 Megabytes). This allows an enterprise to run hundreds of custom department adapters on a single shared model.

#### Q: How do I share this with a colleague on the same office network?
**A:** Find your machine's local network IP address (e.g., `192.168.1.50`). Your colleague can open `http://192.168.1.50:5100` on their laptop or tablet and interact with the studio with you in real time.

#### Q: What should I do if the port is already in use?
**A:** If you see a warning saying `Port 5100 is already in use`, an earlier instance of the studio is still running in another terminal window. Either return to that terminal and press `Ctrl + C`, or run `kill $(lsof -t -i :5100)`.
