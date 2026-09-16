const path = require('path');
const fs = require('fs');

// Attempt to load playwright-core from local node_modules or global/scratch path
let playwright;
try {
  playwright = require('playwright-core');
} catch {
  playwright = require('/Users/cl0rkster/.gemini/antigravity-ide/brain/7dc610fc-4ea3-4567-97f0-93b0c30454fd/scratch/node_modules/playwright-core');
}
const { chromium } = playwright;

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

(async () => {
  const repoRoot = path.resolve(__dirname, '..');
  const tempVideoDir = path.join(repoRoot, '.temp_demo_videos');
  if (!fs.existsSync(tempVideoDir)) fs.mkdirSync(tempVideoDir, { recursive: true });

  console.log('Launching browser for automated demo recording...');
  const browserPath = fs.existsSync('/Applications/Brave Browser.app/Contents/MacOS/Brave Browser')
    ? '/Applications/Brave Browser.app/Contents/MacOS/Brave Browser'
    : '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

  const browser = await chromium.launch({
    executablePath: browserPath,
    headless: true,
    args: ['--no-sandbox', '--disable-gpu', '--window-size=1366,860']
  });

  const context = await browser.newContext({
    viewport: { width: 1366, height: 860 },
    recordVideo: {
      dir: tempVideoDir,
      size: { width: 1366, height: 860 }
    }
  });

  const page = await context.newPage();

  console.log('Navigating to Studio...');
  await page.goto('http://localhost:5100/', { waitUntil: 'networkidle' });
  await sleep(2000);

  // 1. Comparison Arena - Persona 1: Fintech Support & Compliance
  console.log('Step 1: Testing Fintech Compliance Persona...');
  const promptPill = await page.$('.prompt-pill');
  if (promptPill) {
    await promptPill.click();
    await sleep(1500);
  }

  const runBtn = await page.$('#btn-run-compare');
  if (runBtn) {
    await runBtn.click();
    console.log('Clicked Compare...');
    await sleep(3500);
  }

  // Smooth scroll down to view side-by-side results and policy checklist
  await page.evaluate(() => {
    window.scrollBy({ top: 350, behavior: 'smooth' });
  });
  await sleep(4000);

  // 2. Switch to Enterprise SaaS Ops Persona
  console.log('Step 2: Testing Enterprise SaaS Ops Persona...');
  await page.evaluate(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  await sleep(1000);

  const personaCards = await page.$$('.persona-card');
  if (personaCards.length > 1) {
    await personaCards[1].click();
    await sleep(1500);

    const saasPrompt = await page.$('.prompt-pill');
    if (saasPrompt) {
      await saasPrompt.click();
      await sleep(1500);
    }

    if (runBtn) {
      await runBtn.click();
      await sleep(3500);
    }

    await page.evaluate(() => {
      window.scrollBy({ top: 350, behavior: 'smooth' });
    });
    await sleep(3500);
  }

  // 3. Switch to No-Code Adapter Studio Tab
  console.log('Step 3: Demonstrating No-Code Studio...');
  await page.evaluate(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  await sleep(1000);

  const studioTab = await page.$('.nav-tab-btn[data-tab="studio"]');
  if (studioTab) {
    await studioTab.click();
    await sleep(2000);
  }

  // Load Persona Policy Examples
  const loadBtn = await page.$('#btn-load-template');
  if (loadBtn) {
    await loadBtn.click();
    await sleep(2000);
  }

  // Scroll down to see dataset table
  await page.evaluate(() => {
    window.scrollBy({ top: 250, behavior: 'smooth' });
  });
  await sleep(1500);

  // Add an extra row and demonstrate PII detection
  console.log('Step 4: Demonstrating Real-Time PII Guardrail...');
  const addRowBtn = await page.$('#btn-add-row');
  if (addRowBtn) {
    await addRowBtn.click();
    await sleep(1000);

    const inputs = await page.$$('.prompt-cell');
    if (inputs.length > 0) {
      const lastInput = inputs[inputs.length - 1];
      // Type SSN to trigger active compliance alert
      await lastInput.fill('Customer SSN is 123-45-6789 requesting contract cancellation');
      await sleep(2500);

      // Sanitize input to clear guardrail
      await lastInput.fill('Customer Account #CORP-8842 requesting early termination review');
      await sleep(1500);
    }

    const compInputs = await page.$$('.comp-cell');
    if (compInputs.length > 0) {
      const lastComp = compInputs[compInputs.length - 1];
      await lastComp.fill('Under Policy Addendum C: Early termination requires VP approval. Tag: [MSA-ADD-C-TERM]');
      await sleep(1500);
    }
  }

  // Enter Job Name & Submit
  const jobNameInput = await page.$('#job-name-input');
  if (jobNameInput) {
    await jobNameInput.fill('enterprise-saas-v2');
    await sleep(1200);
  }

  const submitBtn = await page.$('#btn-submit-job');
  if (submitBtn) {
    await submitBtn.click();
    console.log('Submitted Job from Studio...');
    await sleep(3000);
  }

  // 5. Watch Pipeline Tracker Tab
  console.log('Step 5: Visualizing Event-Driven Pipeline...');
  await page.evaluate(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
  await sleep(3500);

  // 6. Switch to Adapter Library Tab
  console.log('Step 6: Showcasing Adapter Library...');
  const libTab = await page.$('.nav-tab-btn[data-tab="library"]');
  if (libTab) {
    await libTab.click();
    await sleep(3500);
  }

  console.log('Finalizing recording...');
  await page.close();
  await context.close();
  await browser.close();

  // Convert recorded video to demo.mp4
  const { execSync } = require('child_process');
  const videoFiles = fs.readdirSync(tempVideoDir).filter(f => f.endsWith('.webm'));
  if (videoFiles.length > 0) {
    const latestVideo = path.join(tempVideoDir, videoFiles[videoFiles.length - 1]);
    const destMp4 = path.join(repoRoot, 'demo.mp4');

    let ffmpegPath = 'ffmpeg';
    const venvFfmpeg = path.join(repoRoot, 'src/FtaaSService.Worker/.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-x86_64-v7.1');
    if (fs.existsSync(venvFfmpeg)) {
      ffmpegPath = venvFfmpeg;
    }

    try {
      console.log('Converting recording to web-optimized MP4 (H.264)...');
      execSync(`"${ffmpegPath}" -y -i "${latestVideo}" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "${destMp4}"`, { stdio: 'inherit' });
      console.log(`Demo video successfully saved to: ${destMp4} (${(fs.statSync(destMp4).size / (1024 * 1024)).toFixed(2)} MB)`);
    } catch (e) {
      console.warn('FFmpeg conversion failed:', e.message);
    }

    fs.rmSync(tempVideoDir, { recursive: true, force: true });
  }

  console.log('Recording completed successfully.');
})();
