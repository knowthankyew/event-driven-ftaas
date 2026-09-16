# FTaaS Enterprise Studio — User Guide
### *A Practical Guide for Technical Leads, Product Managers & Operations Teams (No Coding Required)*

---

## 🌟 What is FTaaS Enterprise Studio?

Most AI tools (like ChatGPT or raw open-source models) give **generic, off-the-shelf answers**. They don't know your company's internal acronyms, return policies, or mandatory legal disclosures (like FDIC insurance notices or SLA timelines).

**FTaaS (Fine-Tuning as a Service) Enterprise Studio** is an interactive platform that lets you:
1. **See the difference**: Compare generic AI side-by-side with a customized, compliance-enforced AI model tailored to a specific department.
2. **Train your own assistant**: Teach the AI new team policies in 2 minutes using a simple spreadsheet-like screen—no Python, machine learning knowledge, or terminal coding required.
3. **Verify compliance**: Automatically audit every answer against company rules and regulatory disclosures.

---

## 🚀 How to Launch in 1 Step

Open your Terminal (macOS Terminal, iTerm, or PowerShell) and run this single command:

```bash
./scripts/start-studio.sh
```

### What happens next?
- The platform compiles and starts automatically in 2–3 seconds.
- Your default web browser will immediately open to: **[http://localhost:5100](http://localhost:5100)**.
- **To stop the app when you're done:** Go back to the terminal window and press `Ctrl + C`.

---

## 🧭 Visual Tour of the Studio

The web application is divided into 4 main tabs at the top of your screen:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚡ Comparison Arena   │   🛠️ Adapter Studio   │   🔄 Event Pipeline   │   📚 Adapter Library │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Tab 1: ⚡ Comparison Arena (The "Proof of the Pudding")

This is where you see the tangible business value in real time.

1. **Pick a Business Persona**:
   - **Fintech Support & Compliance**: Handles money transfers, deposit holds, and strict regulatory notices.
   - **Enterprise SaaS Customer Ops**: Handles enterprise SLAs, cancellation terms, and internal ticketing tags.
   - **Financial Earnings & SEC**: Extracts revenue metrics, margins, and basis points from earnings transcripts.
2. **Choose a Test Inquiry**:
   - Click one of the quick scenario pills (e.g. `ACH Deposit Clearance Delay` or `Contract Cancellation Past 30 Days`).
3. **Click "Compare Live Models"**:
   - **Left Panel (Generic AI)**: Shows how a standard base model answers. Notice it gives generic apologies, lacks exact policy numbers, and forgets mandatory legal disclaimers.
   - **Right Panel (Company Custom AI)**: Shows how the specialized model answers. Notice it cites exact policies (e.g. *Regulation CC $10,000 threshold* or *Section 8.2 of Enterprise MSA*), adds automated ticket tags (`[ACH-HELD-VERIFY]`), and includes required regulatory disclaimers.
4. **Audit Checklist (Bottom Box)**:
   - Shows green checkmarks (✓) for each company compliance rule detected in the answer.

---

### Tab 2: 🛠️ No-Code Adapter Studio (Train Your Assistant)

Want to customize the AI for your own team or department?

1. **Add Training Examples**:
   - Click **"Load Persona Policy Examples"** to see sample data, or click **"+ Add Example Row"** to type your own.
   - **Inquiry (Prompt)**: The question an employee or customer asks (e.g., *"Can I get a refund after 45 days?"*).
   - **Compliant Response (Completion)**: The exact answer and policy disclaimers you want the AI to give.
2. **Or Upload an Existing File**:
   - Drag and drop a `.csv` or `.jsonl` document from your computer into the upload box.
3. **Give It a Name & Click "Train & Deploy Team Adapter"**:
   - Give your model a name like `billing-team-v1`.
   - Click the big purple button.
   - The system immediately accepts your job (`202 Accepted`) and automatically switches you to the Event Pipeline view to show the background progress!

---

### Tab 3: 🔄 Event Pipeline (How It Works Behind the Scenes)

If you need to explain the platform to an executive, engineering director, or client, this tab provides an animated visual diagram of the architecture:

1. **1. Business Lead (You)**: Submits 20–50 policy examples through the web interface.
2. **2. Ingestion Gateway (.NET 10)**: Instantly returns `202 Accepted`. This ensures your web browser never freezes while machine learning compute is running.
3. **3. Message Broker (RabbitMQ)**: Transmits a lightweight pointer to the dataset, preventing system bottlenecks.
4. **4. ML Worker (PyTorch / LoRA)**: Trains a tiny **1.8 megabyte adapter file** in 2–3 minutes instead of duplicating a massive 7 gigabyte model.
5. **5. Dynamic Serving Engine**: Keeps one core model warm in memory and swaps your team's adapter in and out on the fly.

---

### Tab 4: 📚 Adapter Library (Active Models)

- Shows all trained company models currently stored on the machine.
- Shows adapter size (~1.8 MB), creation date, and status (*"Ready to Serve"*).
- Click **"⚡ Load in Arena"** on any model card to immediately test it in the Comparison Arena!

---

## ❓ Frequently Asked Questions (FAQ)

#### Q: Do I need Docker or complex background servers running just to try the studio?
**A:** No! The studio includes an intelligent interactive preview mode. You can launch `./scripts/start-studio.sh` standalone, and it will immediately let you test personas, run side-by-side comparisons, and build datasets without needing any Docker containers running.

#### Q: Why does the app say "1.8 MB Adapter"?
**A:** Traditional AI fine-tuning requires copying and saving the entire model (often 7 to 70 Gigabytes), which costs thousands of dollars and huge amounts of disk space. FTaaS uses **LoRA (Low-Rank Adaptation)**, which only trains tiny mathematical adapter layers (~1.8 Megabytes). You can store 500 team adapters on a single thumb drive!

#### Q: How do I share this with a colleague on the same office Wi-Fi?
**A:** Find your computer's local IP address (e.g., `192.168.1.50`). Your colleague can open `http://192.168.1.50:5100` on their laptop or phone and use the studio with you in real time.

#### Q: What should I do if the port is already in use?
**A:** If you see an error saying `Port 5100 is already in use`, it means an earlier instance is still running in another terminal window. Simply close that terminal window or press `Ctrl + C` there.
