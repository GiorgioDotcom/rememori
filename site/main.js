/* rememori.dev — one small script. Copy button, scroll reveals,
   hero canvas field, lazy live demo. No dependencies (the demo
   lazy-loads transformers.js and rememori itself, on request). */

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* ---------- copy install command ---------- */

const copyBtn = document.getElementById('copy');
copyBtn?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText('npm install rememori');
    copyBtn.textContent = 'copied';
    setTimeout(() => (copyBtn.textContent = 'copy'), 1500);
  } catch {
    copyBtn.textContent = 'ctrl+c?';
  }
});

/* ---------- scroll-triggered reveals (code block, graph) ---------- */

const codeBlock = document.querySelector('.code');
codeBlock?.querySelectorAll('.ln').forEach((ln, i) => ln.style.setProperty('--i', i));
document.querySelectorAll('#graph-fig .g-link').forEach((l, i) => l.style.setProperty('--i', i));

const observer = new IntersectionObserver(
  (entries) => {
    for (const e of entries) {
      if (e.isIntersecting) {
        e.target.classList.add('inview');
        observer.unobserve(e.target);
      }
    }
  },
  { threshold: 0.35 },
);
if (codeBlock) observer.observe(codeBlock);
const graphFig = document.getElementById('graph-fig');
if (graphFig) observer.observe(graphFig);

/* ---------- full-page canvas: drifting memory field, mouse-aware ---------- */

const canvas = document.getElementById('field');
if (canvas && !reduced) {
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(devicePixelRatio || 1, 2);
  let w, h, nodes;

  const resize = () => {
    w = innerWidth;
    h = innerHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const count = Math.min(60, Math.floor((w * h) / 24000));
    nodes = Array.from({ length: count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.18,
      vy: (Math.random() - 0.5) * 0.18,
      r: 1.2 + Math.random() * 1.8,
    }));
  };
  resize();
  addEventListener('resize', resize);

  /* pointer: nodes near the cursor drift away, gently */
  const mouse = { x: -1e4, y: -1e4 };
  const REACH = 150;
  const MAX_V = 0.9;
  addEventListener('pointermove', (e) => { mouse.x = e.clientX; mouse.y = e.clientY; }, { passive: true });
  addEventListener('pointerleave', () => { mouse.x = -1e4; mouse.y = -1e4; });

  const LINK = 130;
  let running = true;
  document.addEventListener('visibilitychange', () => {
    running = !document.hidden;
    if (running) requestAnimationFrame(tick);
  });

  function tick() {
    if (!running) return;
    ctx.clearRect(0, 0, w, h);

    for (const n of nodes) {
      const dx = n.x - mouse.x, dy = n.y - mouse.y;
      const d = Math.hypot(dx, dy);
      if (d < REACH && d > 0.5) {
        const push = ((REACH - d) / REACH) * 0.045;
        n.vx += (dx / d) * push;
        n.vy += (dy / d) * push;
      }
      /* cap speed, add gentle friction so nudges settle back to drift */
      const speed = Math.hypot(n.vx, n.vy);
      if (speed > MAX_V) { n.vx = (n.vx / speed) * MAX_V; n.vy = (n.vy / speed) * MAX_V; }
      n.vx *= 0.995; n.vy *= 0.995;

      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > w) n.vx *= -1;
      if (n.y < 0 || n.y > h) n.vy *= -1;
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const d = Math.hypot(dx, dy);
        if (d < LINK) {
          ctx.strokeStyle = `oklch(0.50 0.13 285 / ${(0.4 * (1 - d / LINK)).toFixed(3)})`;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    for (const n of nodes) {
      ctx.fillStyle = 'oklch(0.74 0.17 285 / 0.55)';
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
      ctx.fill();
    }
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ---------- live demo (lazy, explicit consent) ---------- */

const loadBtn = document.getElementById('demo-load');
const boot = document.getElementById('demo-boot');
const ui = document.getElementById('demo-ui');
const status = document.getElementById('demo-status');
const rememberEl = document.getElementById('demo-remember');
const recallEl = document.getElementById('demo-recall');
const resultsEl = document.getElementById('demo-results');

const setStatus = (text, cls = '') => {
  status.textContent = text;
  status.className = `demo-status ${cls}`;
};

loadBtn?.addEventListener('click', async () => {
  loadBtn.disabled = true;
  boot.hidden = true;
  ui.hidden = false;
  setStatus('Loading embedding model…');

  let mem;
  try {
    const [{ pipeline }, { Memory }] = await Promise.all([
      import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/+esm'),
      import('https://cdn.jsdelivr.net/npm/rememori@0.3.0/dist/index.js'),
    ]);

    const extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      progress_callback: (p) => {
        if (p.status === 'progress' && p.file?.endsWith('.onnx')) {
          setStatus(`Downloading model… ${Math.round(p.progress)}% (once, then cached)`);
        }
      },
    });

    const embedder = {
      async embed(texts) {
        const out = await extractor(texts, { pooling: 'mean', normalize: true });
        return out.tolist().map((v) => new Float32Array(v));
      },
    };

    mem = await Memory.open('idb://rememori-site', { embedder });
  } catch (err) {
    setStatus(`Failed to load: ${err?.message ?? err}`, 'err');
    return;
  }

  const count = (n) => `${n} ${n === 1 ? 'memory' : 'memories'}`;
  setStatus(`Ready — ${count(mem.size)} in this browser. Network tab: silent from here on.`, 'ok');

  rememberEl.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter' || !rememberEl.value.trim()) return;
    await mem.remember(rememberEl.value);
    setStatus(`Remembered. ${count(mem.size)}, stored in IndexedDB.`, 'ok');
    rememberEl.value = '';
  });

  recallEl.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter' || !recallEl.value.trim()) return;
    const hits = await mem.recall(recallEl.value, { limit: 5 });
    resultsEl.innerHTML = '';
    if (hits.length === 0) {
      const li = document.createElement('li');
      li.className = 'none';
      li.textContent = 'nothing relevant — remember something first';
      resultsEl.append(li);
      return;
    }
    hits.forEach((h, i) => {
      const li = document.createElement('li');
      li.style.animationDelay = `${i * 60}ms`;
      const text = document.createElement('span');
      text.textContent = h.text;
      const score = document.createElement('span');
      score.className = 'score';
      score.textContent = h.score.toFixed(3);
      li.append(text, score);
      resultsEl.append(li);
    });
  });
});
