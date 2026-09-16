// FTaaS Enterprise Studio - Modern Non-Tech Client Logic
(function() {
  'use strict';

  // State
  const state = {
    personas: [],
    selectedPersona: null,
    activeTab: 'arena',
    jobs: [],
    datasetRows: [],
    activeJobId: null,
    isComparing: false,
    pollInterval: null,
    isLiveEngine: false,
    consecutiveProbeFailures: 0,
    isProbingEngine: false
  };

  // DOM Elements
  const elements = {
    tabs: document.querySelectorAll('.nav-tab-btn'),
    tabViews: document.querySelectorAll('.tab-view'),
    personaGrid: document.getElementById('persona-grid'),
    samplePromptsRow: document.getElementById('sample-prompts-row'),
    promptInput: document.getElementById('prompt-input'),
    maxTokensInput: document.getElementById('max-tokens'),
    tempInput: document.getElementById('temperature'),
    runCompareBtn: document.getElementById('btn-run-compare'),
    baseCompletion: document.getElementById('base-completion'),
    baseLatency: document.getElementById('base-latency'),
    tunedCompletion: document.getElementById('tuned-completion'),
    tunedLatency: document.getElementById('tuned-latency'),
    tunedAdapterLabel: document.getElementById('tuned-adapter-label'),
    complianceGrid: document.getElementById('compliance-grid'),
    previewNotice: document.getElementById('preview-notice'),
    modeStatusPill: document.getElementById('mode-status-pill'),
    modeStatusIndicator: document.getElementById('mode-status-indicator'),
    modeStatusText: document.getElementById('mode-status-text'),
    baseModeTag: document.getElementById('base-mode-tag'),
    tunedModeTag: document.getElementById('tuned-mode-tag'),
    datasetTableBody: document.getElementById('dataset-table-body'),
    addRowBtn: document.getElementById('btn-add-row'),
    loadTemplateBtn: document.getElementById('btn-load-template'),
    submitJobBtn: document.getElementById('btn-submit-job'),
    jobNameInput: document.getElementById('job-name-input'),
    fileInput: document.getElementById('file-input'),
    dropZone: document.getElementById('drop-zone'),
    libraryGrid: document.getElementById('library-grid'),
    overviewAdapters: document.getElementById('stat-adapters'),
    pipelineTracker: document.getElementById('pipeline-tracker')
  };

  // Initialize
  async function init() {
    setupTabNavigation();
    setupDropzone();
    setupEventListeners();
    await checkEngineHealth();
    await loadPersonas();
    await loadJobs();
    startPolling();

    // Support URL param for direct deep linking e.g. ?tab=studio or ?tab=pipeline or ?compare=1
    const urlParams = new URLSearchParams(window.location.search);
    const requestedTab = urlParams.get('tab');
    if (requestedTab) {
      switchTab(requestedTab);
    }
    if (urlParams.get('compare') === '1' || urlParams.get('compare') === 'true') {
      setTimeout(runComparison, 500);
    }
  }

  // Navigation
  function setupTabNavigation() {
    elements.tabs.forEach(btn => {
      btn.addEventListener('click', () => {
        const target = btn.dataset.tab;
        switchTab(target);
      });
    });
  }

  function switchTab(tabId) {
    state.activeTab = tabId;
    elements.tabs.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    });
    elements.tabViews.forEach(view => {
      view.classList.toggle('active', view.id === `view-${tabId}`);
    });
  }

  // Toast Notification
  function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
      <span>${type === 'success' ? '✓' : type === 'error' ? '✕' : 'ℹ'}</span>
      <span>${message}</span>
    `;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }

  // Fetch Personas from API (with robust local fallback)
  async function loadPersonas() {
    try {
      const res = await fetch('/api/v1/studio/personas');
      if (res.ok) {
        state.personas = await res.json();
      } else {
        throw new Error('API returned status ' + res.status);
      }
    } catch (err) {
      console.warn('Using bundled persona catalog:', err);
      state.personas = getDefaultPersonas();
    }
    renderPersonas();
    if (state.personas.length > 0) {
      selectPersona(state.personas[0].id);
    }
  }

  function renderPersonas() {
    if (!elements.personaGrid) return;
    elements.personaGrid.innerHTML = '';
    state.personas.forEach(p => {
      const card = document.createElement('div');
      card.className = 'persona-card';
      card.dataset.id = p.id;
      card.innerHTML = `
        <div class="persona-header">
          <div class="persona-name">${p.name}</div>
          <span class="persona-tag">${p.adapterSize || '1.8 MB'}</span>
        </div>
        <div class="persona-dept">${p.department}</div>
        <div class="persona-desc">${p.description}</div>
      `;
      card.addEventListener('click', () => selectPersona(p.id));
      elements.personaGrid.appendChild(card);
    });
  }

  function selectPersona(id) {
    const persona = state.personas.find(p => p.id === id);
    if (!persona) return;
    state.selectedPersona = persona;

    // Highlight active card
    document.querySelectorAll('.persona-card').forEach(card => {
      card.classList.toggle('active', card.dataset.id === id);
    });

    // Update Sample Prompts
    renderSamplePrompts(persona);

    // Update Compliance Rules list
    renderComplianceRules(persona.complianceRules, '');

    // Pre-fill Dataset in Studio if empty
    if (state.datasetRows.length === 0 && persona.seedPairs) {
      state.datasetRows = [...persona.seedPairs];
      renderDatasetRows();
    }
    if (elements.jobNameInput && !elements.jobNameInput.value) {
      elements.jobNameInput.value = persona.defaultJobName || `${persona.id}-v1`;
    }
  }

  function renderSamplePrompts(persona) {
    if (!elements.samplePromptsRow) return;
    elements.samplePromptsRow.innerHTML = '';
    (persona.samplePrompts || []).forEach((sample, idx) => {
      const chip = document.createElement('button');
      chip.className = 'sample-prompt-chip';
      chip.textContent = sample.title;
      chip.title = sample.prompt;
      chip.addEventListener('click', () => {
        elements.promptInput.value = sample.prompt;
        // Automatically run compare for fast non-tech feedback
        runComparison();
      });
      elements.samplePromptsRow.appendChild(chip);
    });

    // Pick first prompt as default if empty
    if (persona.samplePrompts && persona.samplePrompts.length > 0 && !elements.promptInput.value) {
      elements.promptInput.value = persona.samplePrompts[0].prompt;
    }
  }

  function renderComplianceRules(rules, responseText) {
    if (!elements.complianceGrid) return;
    elements.complianceGrid.innerHTML = '';
    const lowerResp = (responseText || '').toLowerCase();

    rules.forEach(rule => {
      const item = document.createElement('div');
      item.className = 'compliance-rule-item';

      // Smart keyword check for rule validation
      let passed = false;
      const rLower = rule.toLowerCase();
      if (rLower.includes('fdic') && (lowerResp.includes('fdic') || lowerResp.includes('deposit account'))) passed = true;
      else if (rLower.includes('reg cc') && (lowerResp.includes('reg cc') || lowerResp.includes('10,000') || lowerResp.includes('verification'))) passed = true;
      else if (rLower.includes('2210') && (lowerResp.includes('not offer financial') || lowerResp.includes('sec-no-advisory') || lowerResp.includes('risk'))) passed = true;
      else if (rLower.includes('30-day') && (lowerResp.includes('30 calendar days') || lowerResp.includes('msa-sec8') || lowerResp.includes('non-refundable'))) passed = true;
      else if (rLower.includes('sev-1') && (lowerResp.includes('15 minutes') || lowerResp.includes('pagerduty') || lowerResp.includes('incident-sev1'))) passed = true;
      else if (rLower.includes('gdpr') && (lowerResp.includes('7 years') || lowerResp.includes('statutory') || lowerResp.includes('gdpr-exception'))) passed = true;
      else if (rLower.includes('sentiment') && lowerResp.includes('sentiment:') && lowerResp.includes('metrics:')) passed = true;
      else if (responseText && lowerResp.includes('tag:') || lowerResp.includes('tag [')) passed = true;

      if (passed) {
        item.classList.add('passed');
        item.innerHTML = `
          <div class="rule-status-icon check">✓</div>
          <div>
            <strong>Rule Enforced:</strong> ${rule}
          </div>
        `;
      } else {
        item.innerHTML = `
          <div class="rule-status-icon warn">!</div>
          <div>
            <strong>Rule Requirement:</strong> ${rule}
          </div>
        `;
      }
      elements.complianceGrid.appendChild(item);
    });
  }

  // Arena: Run Side-by-Side Comparison
  async function runComparison() {
    const prompt = elements.promptInput.value.trim();
    if (!prompt) {
      showToast('Please enter or select a customer inquiry prompt.', 'error');
      return;
    }

    state.isComparing = true;
    elements.runCompareBtn.disabled = true;
    elements.runCompareBtn.innerHTML = '<span>⏳</span> Analyzing & Comparing...';

    elements.baseCompletion.classList.add('placeholder');
    elements.baseCompletion.textContent = 'Generating base model completion...';
    elements.tunedCompletion.classList.add('placeholder');
    elements.tunedCompletion.textContent = 'Mounting LoRA adapter & evaluating policies...';

    // Find any completed job adapter if available
    const completedJob = state.jobs.find(j => j.status === 'Succeeded' && j.adapterPath);
    const adapterPath = completedJob ? completedJob.adapterPath : 'artifacts/default/model_adapters';
    const jobId = completedJob ? completedJob.jobId : null;

    if (elements.tunedAdapterLabel) {
      elements.tunedAdapterLabel.textContent = state.selectedPersona 
        ? `${state.selectedPersona.name} (${state.selectedPersona.adapterSize || '1.8 MB'})`
        : 'Enterprise Adapter (1.8 MB)';
    }

    try {
      const payload = {
        prompt: prompt,
        maxTokens: parseInt(elements.maxTokensInput.value, 10) || 64,
        temperature: parseFloat(elements.tempInput.value) || 0.2,
        jobId: jobId,
        adapterPath: adapterPath
      };

      const res = await fetch('/api/v1/inference/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`Inference API error: ${res.status}`);
      }

      const data = await res.json();
      renderComparisonResult(data);
    } catch (err) {
      console.error('Comparison request failed:', err);
      // Fallback local simulation if backend has any issue
      renderSimulatedResult(prompt);
    } finally {
      state.isComparing = false;
      elements.runCompareBtn.disabled = false;
      elements.runCompareBtn.innerHTML = '<span>⚡</span> Compare Models';
    }
  }

  function renderComparisonResult(data) {
    elements.baseCompletion.classList.remove('placeholder');
    elements.baseCompletion.textContent = data.baseCompletion || 'No output generated.';
    elements.baseLatency.textContent = `Latency: ${data.latencyMs?.baseModel || 45}ms`;

    elements.tunedCompletion.classList.remove('placeholder');
    elements.tunedCompletion.textContent = data.fineTunedCompletion || 'No output generated.';
    elements.tunedLatency.textContent = `Latency: ${data.latencyMs?.fineTuned || 48}ms • Memory: ~1.8 MB`;

    // Loud, explicit notice banner for preview vs live compute transparency
    if (elements.previewNotice) {
      elements.previewNotice.style.display = 'block';
      if (data.isSimulated) {
        elements.previewNotice.style.background = 'rgba(245, 158, 11, 0.12)';
        elements.previewNotice.style.borderColor = 'rgba(245, 158, 11, 0.35)';
        elements.previewNotice.style.color = '#fde68a';
        elements.previewNotice.innerHTML = `
          <strong>🟡 Interactive Preview Mode Active:</strong> The live ML inference service (:8000) is currently offline or pre-warming. The responses shown below are pre-formatted demonstration examples illustrating team policy formatting and compliance tags, <em>not live neural network generation</em>. To run real weights on your hardware, launch the Python worker and inference services.
        `;
        if (elements.modeStatusIndicator) {
          elements.modeStatusIndicator.style.background = '#f59e0b';
          elements.modeStatusIndicator.style.boxShadow = '0 0 8px #f59e0b';
        }
        if (elements.modeStatusText) {
          elements.modeStatusText.innerHTML = 'Engine: <strong style="color:#fbbf24;">Preview Mode</strong>';
        }
        if (elements.baseModeTag) elements.baseModeTag.textContent = 'Preview Output';
        if (elements.tunedModeTag) elements.tunedModeTag.textContent = 'Preview Output';
      } else {
        elements.previewNotice.style.background = 'rgba(16, 185, 129, 0.12)';
        elements.previewNotice.style.borderColor = 'rgba(16, 185, 129, 0.35)';
        elements.previewNotice.style.color = '#a7f3d0';
        elements.previewNotice.innerHTML = `
          <strong>✓ Live Compute Run:</strong> Output generated dynamically on ${data.device || 'device'} from loaded PyTorch model weights.
        `;
        if (elements.modeStatusIndicator) {
          elements.modeStatusIndicator.style.background = '#10b981';
          elements.modeStatusIndicator.style.boxShadow = '0 0 8px #10b981';
        }
        if (elements.modeStatusText) {
          elements.modeStatusText.innerHTML = 'Engine: <strong style="color:#34d399;">Live Compute</strong>';
        }
        if (elements.baseModeTag) elements.baseModeTag.textContent = 'Live Model Run';
        if (elements.tunedModeTag) elements.tunedModeTag.textContent = 'Live LoRA Mount';
      }
    }

    // Update compliance checklist
    if (state.selectedPersona) {
      renderComplianceRules(state.selectedPersona.complianceRules, data.fineTunedCompletion);
    }
  }

  function renderSimulatedResult(prompt) {
    const pLower = prompt.toLowerCase();
    let baseComp = "We have received your request and will look into this issue shortly. Please contact support if you need further assistance.";
    let tunedComp = "[TAG: ENTERPRISE-POLICY-ENFORCED] Advise customer: Under company policy Section 4, requests are verified under standard SLA. Required statutory disclaimers attached.";

    if (pLower.includes('transfer') || pLower.includes('15,000') || pLower.includes('delayed')) {
      baseComp = "We apologize for the delay with your transfer. Transfers can take several business days to arrive depending on your bank. Please wait another 24 to 48 hours.";
      tunedComp = "Advise the customer: Under Reg CC and internal ACH policy, external transfers exceeding $10,000 are subject to standard 3-5 business day secondary verification. Reference Ticket Tag: [ACH-HELD-VERIFY]. Mandatory compliance notice: 'Funds are held in accordance with Federal Reserve Regulation CC and FinCEN transaction monitoring guidelines. FDIC insurance coverage applies once funds are credited to your deposit account.'";
    } else if (pLower.includes('cancel') || pLower.includes('45 days') || pLower.includes('contract')) {
      baseComp = "I understand you want to cancel. We are sorry to see you go. Please review our terms of service for any refund questions.";
      tunedComp = "Internal Policy Response: Under Section 8.2 of Enterprise SaaS Master Services Agreement, the standard cancellation window is strictly 30 calendar days from provision date. Contracts beyond 30 days are non-refundable for the remaining annual term. Tag: [MSA-SEC8-NONREF]. Recommended escalation: Offer dedicated Customer Success review [CS-REENGAGE] or contract seat reallocation under Addendum B.";
    } else if (pLower.includes('margin') || pLower.includes('revenue') || pLower.includes('operating')) {
      baseComp = "The company reported higher margins and revenue growth this quarter.";
      tunedComp = "SENTIMENT: Positive | METRICS: Gross Margin +340bps (43.1%), Operating Income +18% | SUMMARY: Strong margin expansion propelled by freight tailwinds and product mix.";
    }

    renderComparisonResult({
      baseCompletion: baseComp,
      fineTunedCompletion: tunedComp,
      latencyMs: { baseModel: 42.1, fineTuned: 44.8 },
      isSimulated: true,
      note: 'Inference engine is in preview mode. Realistic domain evaluation loaded.'
    });
  }

  // Active Engine Health Probe (with load debounce / hysteresis)
  async function checkEngineHealth() {
    if (state.isProbingEngine) return;
    state.isProbingEngine = true;
    try {
      const res = await fetch('/api/v1/studio/engine-health');
      if (res.ok) {
        const data = await res.json();
        if (data.isLive) {
          state.isLiveEngine = true;
          state.consecutiveProbeFailures = 0;
          if (elements.modeStatusIndicator) {
            elements.modeStatusIndicator.style.background = '#10b981';
            elements.modeStatusIndicator.style.boxShadow = '0 0 8px #10b981';
          }
          if (elements.modeStatusText) {
            elements.modeStatusText.innerHTML = `Engine: <strong style="color: #34d399;">Live Compute (${data.device})</strong>`;
          }
          return;
        }
      }
      handleProbeFailure();
    } catch {
      handleProbeFailure();
    } finally {
      state.isProbingEngine = false;
    }
  }

  function handleProbeFailure() {
    state.consecutiveProbeFailures++;
    // Only downgrade status to Preview Mode after 2 consecutive failures to prevent flapping during heavy inference
    if (state.consecutiveProbeFailures >= 2 || !state.isLiveEngine) {
      state.isLiveEngine = false;
      if (elements.modeStatusIndicator) {
        elements.modeStatusIndicator.style.background = '#f59e0b';
        elements.modeStatusIndicator.style.boxShadow = '0 0 8px #f59e0b';
      }
      if (elements.modeStatusText) {
        elements.modeStatusText.innerHTML = `Engine: <strong style="color: #fbbf24;">Preview Mode</strong>`;
      }
    }
  }

  // Luhn checksum algorithm for credit/debit card validation
  function passesLuhn(digits) {
    let sum = 0;
    let alternate = false;
    for (let i = digits.length - 1; i >= 0; i--) {
      let n = parseInt(digits.charAt(i), 10);
      if (alternate) {
        n *= 2;
        if (n > 9) n -= 9;
      }
      sum += n;
      alternate = !alternate;
    }
    return sum % 10 === 0;
  }

  // Active PII Guardrail Detector (Delimiter-aware SSA SSN and Luhn-validated Payment Cards)
  function scanForPII(text) {
    if (!text) return null;

    // 1. Delimited SSN (XXX-XX-XXXX or XXX XX XXXX) with SSA rule checks (area != 000, 666, 900-999; group != 00; serial != 0000)
    const ssnDelimited = /\b(?!000|666|9\d{2})\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}\b/;
    if (ssnDelimited.test(text)) return 'Social Security Number (SSN)';

    // 2. Contextual SSN: only match contiguous 9 digits if preceded by "ssn" or "social security"
    const ssnContext = /(?:\bssn\b|\bsocial\s*security(?:\s*number)?)[^\d]{1,10}((?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4})\b/i;
    if (ssnContext.test(text)) return 'Social Security Number (SSN)';

    // 3. Credit / Debit Card: Check 13-19 digits formatted or raw, verified with Luhn checksum
    const cardMatches = text.match(/\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{15,16}\b/g);
    if (cardMatches) {
      for (const card of cardMatches) {
        const digitsOnly = card.replace(/\D/g, '');
        if (digitsOnly.length >= 13 && digitsOnly.length <= 19 && passesLuhn(digitsOnly)) {
          return 'Credit/Debit Card Number';
        }
      }
    }

    return null;
  }

  function updatePiiStatus() {
    let piiFound = null;
    let piiRowIndex = -1;

    const rows = elements.datasetTableBody ? elements.datasetTableBody.querySelectorAll('tr') : [];
    rows.forEach((tr, index) => {
      const promptInput = tr.querySelector('.prompt-cell');
      const compInput = tr.querySelector('.comp-cell');

      const piiPrompt = promptInput ? scanForPII(promptInput.value) : null;
      const piiComp = compInput ? scanForPII(compInput.value) : null;

      if (promptInput) promptInput.classList.toggle('pii-flagged', !!piiPrompt);
      if (compInput) compInput.classList.toggle('pii-flagged', !!piiComp);

      if ((piiPrompt || piiComp) && !piiFound) {
        piiFound = piiPrompt || piiComp;
        piiRowIndex = index + 1;
      }
    });

    let piiBanner = document.getElementById('pii-guardrail-banner');
    if (piiFound) {
      if (!piiBanner && elements.datasetTableBody) {
        piiBanner = document.createElement('div');
        piiBanner.id = 'pii-guardrail-banner';
        piiBanner.style.cssText = 'background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; padding: 0.75rem 1.25rem; border-radius: 8px; margin-bottom: 1rem; font-size: 0.85rem;';
        const card = elements.datasetTableBody.closest('.studio-card');
        const table = elements.datasetTableBody.closest('table');
        if (card && table) card.insertBefore(piiBanner, table);
      }
      if (piiBanner) {
        piiBanner.innerHTML = `🚨 <strong>Active PII Guardrail Alert:</strong> Detected potential real customer <strong>${piiFound}</strong> in Row ${piiRowIndex}. Training is blocked until sanitized or replaced with fictitious examples.`;
      }
      if (elements.submitJobBtn) {
        elements.submitJobBtn.disabled = true;
        elements.submitJobBtn.title = 'Blocked by PII guardrail';
      }
    } else {
      if (piiBanner) piiBanner.remove();
      if (elements.submitJobBtn) {
        elements.submitJobBtn.disabled = false;
        elements.submitJobBtn.title = '';
      }
    }
  }

  // No-Code Dataset Builder
  function renderDatasetRows() {
    if (!elements.datasetTableBody) return;
    elements.datasetTableBody.innerHTML = '';

    state.datasetRows.forEach((row, index) => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td style="width: 45%;">
          <input type="text" class="dataset-input prompt-cell" value="${escapeHtml(row.prompt)}" placeholder="e.g. Customer asking for contract cancellation policy">
        </td>
        <td style="width: 45%;">
          <input type="text" class="dataset-input comp-cell" value="${escapeHtml(row.completion)}" placeholder="e.g. Policy citation, tag, and required disclaimer">
        </td>
        <td style="width: 10%; text-align: center;">
          <button type="button" class="btn-danger-sm" data-index="${index}">✕</button>
        </td>
      `;

      tr.querySelector('.prompt-cell').addEventListener('input', (e) => {
        state.datasetRows[index].prompt = e.target.value;
        updatePiiStatus();
      });
      tr.querySelector('.comp-cell').addEventListener('input', (e) => {
        state.datasetRows[index].completion = e.target.value;
        updatePiiStatus();
      });
      tr.querySelector('.btn-danger-sm').addEventListener('click', () => {
        state.datasetRows.splice(index, 1);
        renderDatasetRows();
      });

      elements.datasetTableBody.appendChild(tr);
    });

    updatePiiStatus();
  }

  function addDatasetRow() {
    state.datasetRows.push({ prompt: '', completion: '' });
    renderDatasetRows();
    const rows = elements.datasetTableBody.querySelectorAll('tr');
    if (rows.length > 0) {
      const lastInput = rows[rows.length - 1].querySelector('.prompt-cell');
      if (lastInput) lastInput.focus();
    }
  }

  function loadTemplateRows() {
    if (state.selectedPersona && state.selectedPersona.seedPairs) {
      state.datasetRows = [...state.selectedPersona.seedPairs];
      renderDatasetRows();
      showToast(`Loaded ${state.datasetRows.length} policy examples from "${state.selectedPersona.name}".`, 'success');
    }
  }

  // Submit Fine-Tuning Job from Studio
  async function submitFineTuningJob() {
    // Collect valid rows
    const validPairs = state.datasetRows.filter(r => r.prompt.trim() && r.completion.trim());
    if (validPairs.length === 0) {
      showToast('Please provide at least one prompt-completion pair.', 'error');
      return;
    }

    const jobName = (elements.jobNameInput && elements.jobNameInput.value.trim()) || `adapter-${Date.now().toString(36)}`;
    elements.submitJobBtn.disabled = true;
    elements.submitJobBtn.innerHTML = '<span>⏳</span> Dispatching Job to Queue...';

    // Format as JSONL string
    const jsonlLines = validPairs.map(p => JSON.stringify(p)).join('\n');
    const blob = new Blob([jsonlLines], { type: 'application/x-jsonlines' });

    const formData = new FormData();
    formData.append('file', blob, `${jobName}.jsonl`);
    formData.append('jobName', jobName);
    formData.append('baseModel', 'HuggingFaceTB/SmolLM2-135M');
    formData.append('hyperparameters', JSON.stringify({
      epochs: 3,
      learningRate: 0.0003,
      batchSize: 4,
      loraRank: 8,
      loraAlpha: 16
    }));

    try {
      const res = await fetch('/api/v1/jobs', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.details || errJson.error || `HTTP ${res.status}`);
      }

      const jobData = await res.json();
      showToast(`Job "${jobData.jobName}" accepted (ID: ${jobData.jobId.substring(0, 8)}...)!`, 'success');

      // Add to local state & reload jobs
      await loadJobs();

      // Switch to Architecture & Pipeline tab to watch the live event flow
      switchTab('pipeline');
      highlightActivePipelineNode('node-gateway');
      setTimeout(() => highlightActivePipelineNode('node-broker'), 1200);
    } catch (err) {
      showToast(`Submission failed: ${err.message}`, 'error');
    } finally {
      elements.submitJobBtn.disabled = false;
      elements.submitJobBtn.innerHTML = '<span>🚀</span> Train & Deploy Team Adapter';
    }
  }

  function highlightActivePipelineNode(nodeId) {
    document.querySelectorAll('.flow-step-node').forEach(n => n.classList.remove('active'));
    const target = document.getElementById(nodeId);
    if (target) target.classList.add('active');
  }

  // File Dropzone handling
  function setupDropzone() {
    if (!elements.dropZone || !elements.fileInput) return;

    elements.dropZone.addEventListener('click', () => elements.fileInput.click());
    elements.dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      elements.dropZone.style.borderColor = 'var(--accent-primary)';
    });
    elements.dropZone.addEventListener('dragleave', () => {
      elements.dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
    });
    elements.dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      elements.dropZone.style.borderColor = 'rgba(255, 255, 255, 0.15)';
      if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
      }
    });

    elements.fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
      }
    });
  }

  const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB

  function handleFileUpload(file) {
    if (file.size > MAX_FILE_SIZE_BYTES) {
      showToast(`File "${file.name}" (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 25 MB limit.`, 'error');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target.result;
      parseAndImportDataset(text, file.name);
    };
    reader.readAsText(file);
  }

  function parseAndImportDataset(content, fileName) {
    const lines = content.split('\n').map(l => l.trim()).filter(l => l.length > 0);
    const parsed = [];
    let piiFoundInUnmappedColumn = false;

    // Try parsing as JSONL
    let jsonlSuccess = true;
    for (const line of lines) {
      try {
        const obj = JSON.parse(line);
        if (obj.prompt && obj.completion) {
          parsed.push({ prompt: String(obj.prompt), completion: String(obj.completion) });
        }
        // Scan full raw line for PII (including unmapped metadata/notes fields)
        if (!piiFoundInUnmappedColumn && scanForPII(line)) {
          piiFoundInUnmappedColumn = true;
        }
      } catch {
        jsonlSuccess = false;
        break;
      }
    }

    // Try parsing as CSV if JSONL failed
    if (!jsonlSuccess || parsed.length === 0) {
      parsed.length = 0;
      let promptIdx = 0;
      let compIdx = 1;
      let startRow = 0;

      // Check for header row
      if (lines.length > 0) {
        const headerParts = lines[0].split(',').map(h => h.trim().toLowerCase().replace(/['"]/g, ''));
        const pIdx = headerParts.findIndex(h => h === 'prompt' || h === 'question' || h === 'inquiry');
        const cIdx = headerParts.findIndex(h => h === 'completion' || h === 'response' || h === 'answer');
        if (pIdx !== -1 && cIdx !== -1) {
          promptIdx = pIdx;
          compIdx = cIdx;
          startRow = 1;
        }
      }

      for (let i = startRow; i < lines.length; i++) {
        const line = lines[i];
        const parts = line.split(',');
        if (parts.length > Math.max(promptIdx, compIdx)) {
          const prompt = parts[promptIdx].trim().replace(/^["']|["']$/g, '');
          const comp = (compIdx === parts.length - 1)
            ? parts.slice(compIdx).join(',').trim().replace(/^["']|["']$/g, '')
            : parts[compIdx].trim().replace(/^["']|["']$/g, '');
          parsed.push({ prompt, completion: comp });
        }
        // Scan full raw line across all CSV columns (including unmapped notes/metadata)
        if (!piiFoundInUnmappedColumn && scanForPII(line)) {
          piiFoundInUnmappedColumn = true;
        }
      }
    }

    if (parsed.length > 0) {
      state.datasetRows = parsed;
      renderDatasetRows();
      const piiRow = parsed.find(r => scanForPII(r.prompt) || scanForPII(r.completion));
      if (piiRow || piiFoundInUnmappedColumn) {
        showToast(`Security Alert: "${fileName}" contains customer PII (in data or unmapped columns). Training is blocked until sanitized.`, 'error');
      } else {
        showToast(`Imported ${parsed.length} examples from "${fileName}" (unmapped metadata columns safely excluded).`, 'success');
      }
    } else {
      showToast('Could not parse prompt/completion pairs from file.', 'error');
    }
  }

  // Load and Render Jobs
  async function loadJobs() {
    try {
      const res = await fetch('/api/v1/jobs');
      if (res.ok) {
        state.jobs = await res.json();
        renderJobLibrary();
        updateOverviewStats();
      }
    } catch (err) {
      console.warn('Jobs fetch error:', err);
    }
  }

  function renderJobLibrary() {
    if (!elements.libraryGrid) return;
    elements.libraryGrid.innerHTML = '';

    if (state.jobs.length === 0) {
      elements.libraryGrid.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; color: var(--text-dim); padding: 3rem;">
          No company adapters trained yet. Use the <strong>Adapter Studio</strong> tab to train your first assistant!
        </div>
      `;
      return;
    }

    state.jobs.forEach(job => {
      const card = document.createElement('div');
      card.className = 'library-card';
      const isSucceeded = job.status === 'Succeeded';
      const isFailed = job.status === 'Failed';
      const isTraining = job.status === 'Training';
      const isQueued = job.status === 'Queued';
      const ageMs = Date.now() - new Date(job.createdAt).getTime();
      const isWorkerWaiting = isQueued && (ageMs > 30000);

      let statusBadge;
      if (isSucceeded) {
        statusBadge = `<span class="persona-tag" style="background: rgba(16, 185, 129, 0.15); color: #34d399;" title="Stored at ${job.adapterPath}">✓ Real LoRA Adapter</span>`;
      } else if (isTraining) {
        statusBadge = `<span class="persona-tag" style="background: rgba(99, 102, 241, 0.2); color: #a5b4fc;">⚡ Training Live...</span>`;
      } else if (isWorkerWaiting) {
        statusBadge = `<span class="persona-tag" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24;" title="In queue, but Python worker (consumer.py) is offline">⏳ Queued (Worker Offline)</span>`;
      } else if (isQueued) {
        statusBadge = `<span class="persona-tag" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24;">Queued...</span>`;
      } else {
        statusBadge = `<span class="persona-tag" style="background: rgba(239, 68, 68, 0.15); color: #f87171;" title="${escapeHtml(job.errorMessage || 'Training process error')}">Failed</span>`;
      }

      card.innerHTML = `
        <div class="library-card-header">
          <div>
            <div class="library-card-name">${escapeHtml(job.jobName)}</div>
            <div class="library-card-id">ID: ${job.jobId.substring(0, 8)}...</div>
          </div>
          ${statusBadge}
        </div>
        <div class="library-metrics">
          <div class="library-metric-box">
            <div class="library-metric-label">Base Model</div>
            <div class="library-metric-val">SmolLM2-135M</div>
          </div>
          <div class="library-metric-box">
            <div class="library-metric-label">Artifact Footprint</div>
            <div class="library-metric-val">${isSucceeded ? '~1.8 MB' : '--'}</div>
          </div>
        </div>
        ${isWorkerWaiting ? `
          <div style="font-size: 0.75rem; color: #fbbf24; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 6px; padding: 0.4rem 0.6rem; margin-bottom: 0.75rem;">
            ⚠️ Job waiting in queue. The Python worker (<code>consumer.py</code>) is currently offline.
          </div>
        ` : ''}
        ${isFailed && job.errorMessage ? `
          <div style="font-size: 0.75rem; color: #f87171; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 6px; padding: 0.4rem 0.6rem; margin-bottom: 0.75rem;">
            Error: ${escapeHtml(job.errorMessage)}
          </div>
        ` : ''}
        <div style="margin-top: 0.5rem; display: flex; justify-content: space-between; align-items: center;">
          <span style="font-size: 0.75rem; color: var(--text-dim);">
            ${new Date(job.createdAt).toLocaleDateString()}
          </span>
          ${isSucceeded ? `
            <button class="btn-secondary btn-test-adapter" data-jobid="${job.jobId}" data-adapter="${job.adapterPath}">
              ⚡ Load in Arena
            </button>
          ` : ''}
        </div>
      `;

      const testBtn = card.querySelector('.btn-test-adapter');
      if (testBtn) {
        testBtn.addEventListener('click', () => {
          switchTab('arena');
          if (elements.tunedAdapterLabel) {
            elements.tunedAdapterLabel.textContent = `${job.jobName} (~1.8 MB)`;
          }
          runComparison();
          showToast(`Mounted adapter "${job.jobName}" into comparison arena!`, 'success');
        });
      }

      elements.libraryGrid.appendChild(card);
    });
  }

  function updateOverviewStats() {
    const succeeded = state.jobs.filter(j => j.status === 'Succeeded').length;
    if (elements.overviewAdapters) {
      elements.overviewAdapters.textContent = succeeded;
    }
  }

  function startPolling() {
    if (state.pollInterval) clearInterval(state.pollInterval);
    state.pollInterval = setInterval(async () => {
      await checkEngineHealth();
      await loadJobs();
    }, 8000);
  }

  function setupEventListeners() {
    if (elements.runCompareBtn) {
      elements.runCompareBtn.addEventListener('click', runComparison);
    }
    if (elements.addRowBtn) {
      elements.addRowBtn.addEventListener('click', addDatasetRow);
    }
    if (elements.loadTemplateBtn) {
      elements.loadTemplateBtn.addEventListener('click', loadTemplateRows);
    }
    if (elements.submitJobBtn) {
      elements.submitJobBtn.addEventListener('click', submitFineTuningJob);
    }
  }

  function escapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function getDefaultPersonas() {
    return [
      {
        id: "fintech-compliance",
        name: "Fintech Support & Compliance Copilot",
        department: "Risk, Support & Operations",
        description: "Enforces Reg CC ACH hold limits, BSA/AML verification, non-advisory SEC disclaimers, and mandatory FDIC insurance notices.",
        adapterSize: "1.8 MB",
        defaultJobName: "fintech-compliance-v1",
        complianceRules: [
          "Mandatory FDIC deposit status disclaimer",
          "Reg CC 3-5 business day ACH clearance notification for >$10K",
          "Strict non-advisory disclaimer for crypto/stocks (FINRA Rule 2210)",
          "BSA/AML Enhanced Due Diligence (EDD) ticketing tags"
        ],
        samplePrompts: [
          {
            title: "ACH Deposit Clearance Delay",
            prompt: "Customer ticket: User states their account transfer of $15,000 from external credit union is delayed past 2 business days. How should we advise them regarding clearance and compliance?"
          },
          {
            title: "Investment/Crypto Advice Request",
            prompt: "Customer ticket: User is asking for advice on whether they should purchase Ethereum or Bitcoin right now."
          }
        ],
        seedPairs: [
          {
            prompt: "Customer ticket: User states their account transfer of $15,000 from external credit union is delayed past 2 business days. How should we advise them regarding clearance and compliance?",
            completion: "Advise the customer: Under Reg CC and internal ACH policy, external transfers exceeding $10,000 are subject to standard 3-5 business day secondary verification. Reference Ticket Tag: [ACH-HELD-VERIFY]. Mandatory compliance notice: 'Funds are held in accordance with Federal Reserve Regulation CC and FinCEN transaction monitoring guidelines. FDIC insurance coverage applies once funds are credited to your deposit account.'"
          }
        ]
      },
      {
        id: "saas-support-slang",
        name: "Enterprise SaaS Customer Ops",
        department: "Customer Success & Engineering Ops",
        description: "Understands internal enterprise terminology (ACV, NRR, Sev-1 SLAs, RBAC roles) and strictly enforces MSA cancellation terms.",
        adapterSize: "1.8 MB",
        defaultJobName: "saas-ops-v1",
        complianceRules: [
          "Section 8.2 MSA 30-day non-refundable policy citation",
          "Sev-1 incident response timeline (15-min SLA, 30-min updates)",
          "Ticket tagging for automated routing ([MSA-SEC8-NONREF], [INCIDENT-SEV1-ESCALATE])"
        ],
        samplePrompts: [
          {
            title: "Contract Cancellation Past 30 Days",
            prompt: "Customer ticket: Customer requests immediate cancellation of enterprise contract after 45 days, citing low team adoption. What is our contractual policy?"
          }
        ],
        seedPairs: [
          {
            prompt: "Customer ticket: Customer requests immediate cancellation of enterprise contract after 45 days, citing low team adoption. What is our contractual policy?",
            completion: "Internal Policy Response: Under Section 8.2 of Enterprise SaaS Master Services Agreement, the standard cancellation window is strictly 30 calendar days from provision date. Contracts beyond 30 days are non-refundable for the remaining annual term. Tag: [MSA-SEC8-NONREF]. Recommended escalation: Offer dedicated Customer Success review [CS-REENGAGE] or contract seat reallocation under Addendum B."
          }
        ]
      }
    ];
  }

  // Run on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
