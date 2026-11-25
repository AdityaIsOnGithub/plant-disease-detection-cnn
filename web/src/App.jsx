import { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import {
  Upload,
  Cpu,
  Layers,
  ShieldAlert,
  CheckCircle2,
  FileText,
  Sparkles,
  Activity,
  AlertTriangle,
  RefreshCw,
  Leaf,
  Zap,
  Lock,
  Code2,
} from 'lucide-react';

const SAMPLES = [
  { name: 'Bacterial Blight', tag: 'Bacterial', file: '/samples/bacterial_blight.jpg' },
  { name: 'Early Leaf Spot', tag: 'Fungal', file: '/samples/early_leaf_spot.jpg' },
  { name: 'Fungal Leaf Rust', tag: 'Rust', file: '/samples/fungal_leaf_rust.jpg' },
  { name: 'Healthy Leaf', tag: 'Control', file: '/samples/healthy_leaf.jpg' },
  { name: 'Powdery Mildew', tag: 'Mildew', file: '/samples/powdery_mildew.jpg' },
];

function formatLabel(raw) {
  return raw
    .replace(/___/g, ' — ')
    .replace(/_/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function getProtocol(label) {
  const lower = label.toLowerCase();
  if (lower.includes('healthy')) {
    return {
      severity: 'Healthy',
      pathogen: 'None Detected',
      badge: 'bg-emerald-50 text-emerald-800 border-emerald-200',
      accent: 'border-l-emerald-600',
      summary: 'No pathological lesions or chlorotic stress patterns detected.',
      actions: [
        'Maintain standard irrigation and balanced NPK fertilization schedule.',
        'Continue weekly scouting under lower canopy leaves for early spotting.',
        'Avoid overhead watering late in the evening to prevent foliar moisture buildup.',
      ],
    };
  }
  if (lower.includes('bacterial') || lower.includes('greening')) {
    return {
      severity: 'High Risk',
      pathogen: 'Bacterial Pathogen',
      badge: 'bg-rose-50 text-rose-800 border-rose-200',
      accent: 'border-l-rose-600',
      summary: 'Bacterial infection spreading via water splash, tools, or insect vectors; fungicides are ineffective.',
      actions: [
        'Prune and bag heavily spotted foliage; disinfect tools with 70% alcohol between cuts.',
        'Apply a copper-based bactericide or streptomycin spray where locally approved (do not rely on fungicides).',
        'Switch to drip irrigation and avoid working in the canopy while foliage is wet.',
      ],
    };
  }
  if (lower.includes('blight') || lower.includes('rot') || lower.includes('scorch') || lower.includes('measles')) {
    return {
      severity: 'Critical',
      pathogen: 'Oomycete / Necrotic Fungal',
      badge: 'bg-rose-50 text-rose-800 border-rose-200',
      accent: 'border-l-rose-600',
      summary: 'Aggressive necrotic pathology capable of rapid defoliation and yield loss if uncontained.',
      actions: [
        'Remove and destroy infected leaves immediately; do not compost diseased plant debris.',
        'Apply a labeled protectant fungicide (chlorothalonil, mancozeb, or copper) before wet weather.',
        'Increase plant spacing for airflow and rotate away from host crops next season.',
      ],
    };
  }
  if (lower.includes('virus') || lower.includes('mosaic') || lower.includes('curl')) {
    return {
      severity: 'High Risk',
      pathogen: 'Viral (Insect-Vectored)',
      badge: 'bg-amber-50 text-amber-800 border-amber-200',
      accent: 'border-l-amber-600',
      summary: 'Systemic viral infection; chemical sprays cannot cure already infected tissue.',
      actions: [
        'Rogue symptomatic plants early to stop transmission across the crop row.',
        'Control whiteflies, aphids, or thrips using yellow sticky traps and insecticidal soap/neem oil.',
        'Plant certified virus-resistant cultivars and clear nearby weed reservoirs.',
      ],
    };
  }
  if (lower.includes('mite')) {
    return {
      severity: 'Moderate',
      pathogen: 'Arthropod / Pest Damage',
      badge: 'bg-orange-50 text-orange-800 border-orange-200',
      accent: 'border-l-orange-500',
      summary: 'Active spider mite feeding causing fine stippling and reduced photosynthesis.',
      actions: [
        'Inspect leaf undersides with magnification for fine webbing and active mites.',
        'Apply horticultural oil, neem extract, or release predatory mites (Phytoseiulus persimilis).',
        'Eliminate drought stress, which accelerates mite reproduction.',
      ],
    };
  }
  return {
    severity: 'Moderate',
    pathogen: 'Fungal Foliar Pathology',
    badge: 'bg-amber-50 text-amber-800 border-amber-200',
    accent: 'border-l-amber-500',
    summary: 'Localized foliar fungal infection (scab, rust, mildew, or leaf spot); early treatment halts spread.',
    actions: [
      'Remove infected lower leaves to improve airflow and lower spore pressure.',
      'Apply sulfur, potassium bicarbonate, or a labeled broad-spectrum fungicide.',
      'Water at the soil base in the morning and sanitize garden debris at season end.',
    ],
  };
}

function softmax(logits) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  const exps = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i] - max);
    sum += exps[i];
  }
  for (let i = 0; i < logits.length; i++) exps[i] /= sum;
  return exps;
}

function heatColor(v) {
  const t = Math.max(0, Math.min(1, v));
  if (t < 0.25) return [0, 0, 0, 0];
  if (t < 0.55) {
    const k = (t - 0.25) / 0.3;
    return [245, Math.round(158 + 60 * (1 - k)), 11, Math.round(140 * k)];
  }
  const k = (t - 0.55) / 0.45;
  return [Math.round(220 + 25 * k), Math.round(90 * (1 - k)), 20, Math.round(150 + 85 * k)];
}

export default function App() {
  const [status, setStatus] = useState('loading');
  const [error, setError] = useState('');
  const [meta, setMeta] = useState(null);
  const [imageSrc, setImageSrc] = useState(null);
  const [imageName, setImageName] = useState('');
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [showCam, setShowCam] = useState(true);
  const [camOpacity, setCamOpacity] = useState(0.68);
  const [dragOver, setDragOver] = useState(false);

  const sessionRef = useRef(null);
  const canvasRef = useRef(null);
  const loadedImgRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    let active = true;
    async function initModel() {
      try {
        const version = ort.env.versions?.web;
        if (version) {
          ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${version}/dist/`;
        }
        ort.env.wasm.numThreads = 1;

        const metaRes = await fetch('/models/metadata.json');
        if (!metaRes.ok) throw new Error('Could not load /models/metadata.json');
        const metaJson = await metaRes.json();

        const session = await ort.InferenceSession.create(
          '/models/plant_disease_mobilenetv2.onnx',
          { executionProviders: ['wasm'] }
        );

        const size = metaJson.input_size || 224;
        const warmup = new ort.Tensor('float32', new Float32Array(3 * size * size), [1, 3, size, size]);
        await session.run({ pixel_values: warmup });

        if (!active) return;
        sessionRef.current = session;
        setMeta(metaJson);
        setStatus('ready');
      } catch (err) {
        if (!active) return;
        setError(err.message || 'Failed to initialize ONNX Runtime');
        setStatus('error');
      }
    }
    initModel();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    function onPaste(e) {
      if (status !== 'ready') return;
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile();
          if (file) handleFile(file);
          break;
        }
      }
    }
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [status, meta]);

  useEffect(() => {
    if (!loadedImgRef.current || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const img = loadedImgRef.current;
    canvas.width = 520;
    canvas.height = 520;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const scale = Math.max(canvas.width / img.width, canvas.height / img.height);
    const w = img.width * scale;
    const h = img.height * scale;
    ctx.drawImage(img, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);

    if (showCam && result?.camGrid) {
      const { grid, h: gh, w: gw } = result.camGrid;
      const off = document.createElement('canvas');
      off.width = gw;
      off.height = gh;
      const octx = off.getContext('2d');
      const imgData = octx.createImageData(gw, gh);
      for (let i = 0; i < grid.length; i++) {
        const [r, g, b, a] = heatColor(grid[i]);
        imgData.data[i * 4] = r;
        imgData.data[i * 4 + 1] = g;
        imgData.data[i * 4 + 2] = b;
        imgData.data[i * 4 + 3] = Math.round(a * camOpacity);
      }
      octx.putImageData(imgData, 0, 0);
      ctx.save();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }
  }, [imageSrc, result, showCam, camOpacity]);

  async function runInference(src, name) {
    if (!sessionRef.current || !meta) return;
    setRunning(true);
    setError('');
    setImageSrc(src);
    setImageName(name);

    try {
      const img = await new Promise((resolve, reject) => {
        const el = new Image();
        el.crossOrigin = 'anonymous';
        el.onload = () => resolve(el);
        el.onerror = () => reject(new Error('Unable to decode image'));
        el.src = src;
      });
      loadedImgRef.current = img;

      const size = meta.input_size || 224;
      const off = document.createElement('canvas');
      off.width = size;
      off.height = size;
      const ctx = off.getContext('2d');
      const scale = Math.max(size / img.width, size / img.height);
      const drawW = img.width * scale;
      const drawH = img.height * scale;
      ctx.drawImage(img, (size - drawW) / 2, (size - drawH) / 2, drawW, drawH);
      const { data } = ctx.getImageData(0, 0, size, size);

      const mean = meta.image_mean || [0.5, 0.5, 0.5];
      const std = meta.image_std || [0.5, 0.5, 0.5];
      const hw = size * size;
      const input = new Float32Array(3 * hw);

      let foliagePixels = 0;
      for (let i = 0; i < hw; i++) {
        const r = data[i * 4];
        const g = data[i * 4 + 1];
        const b = data[i * 4 + 2];

        const isGreen = g > r * 1.03 && g > b * 1.05 && g > 30;
        const isAutumn = r > 70 && g > 50 && b < g * 0.82 && Math.abs(r - g) < 75;
        if (isGreen || isAutumn) foliagePixels++;

        input[i] = (r / 255 - mean[0]) / std[0];
        input[hw + i] = (g / 255 - mean[1]) / std[1];
        input[2 * hw + i] = (b / 255 - mean[2]) / std[2];
      }

      const tensor = new ort.Tensor('float32', input, [1, 3, size, size]);
      const t0 = performance.now();
      const outputs = await sessionRef.current.run({ pixel_values: tensor });
      const latencyMs = performance.now() - t0;

      const probs = softmax(outputs.logits.data);
      const ranked = Array.from(probs)
        .map((score, idx) => ({ idx, label: meta.labels[idx], score }))
        .sort((a, b) => b.score - a.score);

      const top = ranked[0];
      const camTensor = outputs.cam;
      const camH = camTensor.dims[2];
      const camW = camTensor.dims[3];
      const camSize = camH * camW;
      const offset = top.idx * camSize;
      const slice = camTensor.data.slice(offset, offset + camSize);

      let minV = Infinity;
      let maxV = -Infinity;
      for (let i = 0; i < slice.length; i++) {
        if (slice[i] < minV) minV = slice[i];
        if (slice[i] > maxV) maxV = slice[i];
      }
      const denom = maxV - minV || 1;
      const normGrid = Array.from(slice, (v) => (v - minV) / denom);

      const foliageRatio = foliagePixels / hw;
      const oodWarning = foliageRatio < 0.08 && top.score < 0.72;

      setResult({
        top,
        top5: ranked.slice(0, 5),
        latencyMs,
        foliagePct: Math.round(foliageRatio * 100),
        oodWarning,
        protocol: getProtocol(top.label),
        camGrid: { grid: normGrid, h: camH, w: camW },
      });
    } catch (err) {
      setError(err.message || 'Inference failed');
    } finally {
      setRunning(false);
    }
  }

  function handleFile(file) {
    if (!file || !file.type.startsWith('image/')) {
      setError('Please upload a valid JPG, PNG, or WebP leaf image.');
      return;
    }
    const url = URL.createObjectURL(file);
    runInference(url, file.name);
  }

  function exportReport() {
    if (!result) return;
    const win = window.open('', '_blank', 'width=840,height=920');
    if (!win) return;
    const topRows = result.top5
      .map(
        (item) =>
          `<tr><td style="padding:8px;border-bottom:1px solid #e4e4e7">${formatLabel(item.label)}</td><td style="padding:8px;border-bottom:1px solid #e4e4e7;text-align:right;font-family:monospace">${(item.score * 100).toFixed(2)}%</td></tr>`
      )
      .join('');
    const actionItems = result.protocol.actions
      .map((a) => `<li style="margin-bottom:6px">${a}</li>`)
      .join('');
    const imgDataUrl = canvasRef.current ? canvasRef.current.toDataURL('image/png') : '';

    win.document.write(`
      <html>
        <head>
          <title>Plant Pathology Diagnostic Report</title>
          <style>
            body { font-family: Inter, system-ui, sans-serif; color: #18181b; padding: 36px; max-width: 760px; margin: auto; }
            h1 { margin: 0 0 4px; font-size: 22px; }
            .meta { color: #52525b; font-size: 13px; margin-bottom: 24px; }
            .grid { display: grid; grid-template-columns: 220px 1fr; gap: 24px; margin-bottom: 24px; }
            .card { border: 1px solid #e4e4e7; border-radius: 10px; padding: 16px; margin-bottom: 18px; }
            img { width: 100%; border-radius: 8px; border: 1px solid #e4e4e7; }
            table { width: 100%; border-collapse: collapse; font-size: 14px; }
          </style>
        </head>
        <body>
          <h1>PhytoScan Edge — Leaf Pathology Report</h1>
          <div class="meta">Generated ${new Date().toLocaleString()} · MobileNetV2 ONNX (38-Class PlantVillage) · Local Browser Inference (${result.latencyMs.toFixed(1)} ms)</div>
          <div class="grid">
            <img src="${imgDataUrl}" alt="Analyzed Leaf" />
            <div class="card" style="margin:0">
              <div style="font-size:12px;text-transform:uppercase;color:#71717a;font-weight:600">Primary Diagnosis</div>
              <div style="font-size:20px;font-weight:700;margin:6px 0">${formatLabel(result.top.label)}</div>
              <div style="font-size:14px;color:#3f3f46;margin-bottom:10px">Confidence: <strong>${(result.top.score * 100).toFixed(2)}%</strong> · Severity: <strong>${result.protocol.severity}</strong></div>
              <div style="font-size:13px;color:#52525b">${result.protocol.summary}</div>
            </div>
          </div>
          <div class="card">
            <div style="font-size:12px;text-transform:uppercase;color:#71717a;font-weight:600;margin-bottom:8px">Recommended Agronomic Protocol</div>
            <ul style="margin:0;padding-left:18px;font-size:14px;color:#27272a">${actionItems}</ul>
          </div>
          <div class="card">
            <div style="font-size:12px;text-transform:uppercase;color:#71717a;font-weight:600;margin-bottom:8px">Top-5 Class Probabilities</div>
            <table>${topRows}</table>
          </div>
          <script>window.onload = () => window.print();</script>
        </body>
      </html>
    `);
    win.document.close();
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-200/90 bg-white/90 backdrop-blur-md sticky top-0 z-30">
        <div className="max-w-6xl mx-auto px-5 py-3.5 flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[#0f3d2e] text-emerald-300 flex items-center justify-center shadow-xs">
              <Leaf className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-bold tracking-tight text-zinc-950 leading-none">
                  PhytoScan Edge
                </h1>
                <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-800 border border-emerald-200/80 font-semibold">
                  v2.0 WASM
                </span>
              </div>
              <p className="text-xs text-zinc-500 mt-0.5">
                38-Class PlantVillage CNN · Zero-Server Browser Inference
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full border border-zinc-200 bg-zinc-50 text-zinc-700 font-mono text-[11px]">
              <Cpu className="w-3.5 h-3.5 text-[#0f3d2e]" />
              MobileNetV2 · 8.97 MB
            </span>
            {status === 'loading' && (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-amber-200 bg-amber-50 text-amber-800 font-medium">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                Warming Tensor Graph...
              </span>
            )}
            {status === 'ready' && (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-emerald-200 bg-emerald-50 text-emerald-800 font-medium">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Edge Runtime Ready
              </span>
            )}
            {status === 'error' && (
              <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full border border-rose-200 bg-rose-50 text-rose-800 font-medium">
                <AlertTriangle className="w-3.5 h-3.5" />
                Engine Error
              </span>
            )}
          </div>
        </div>
      </header>

      <div className="bg-[#0f3d2e] text-white border-b border-emerald-950">
        <div className="max-w-6xl mx-auto px-5 py-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div className="max-w-2xl">
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white">
              Real-Time Crop Pathology & Neural Lesion Attribution
            </h2>
            <p className="text-xs sm:text-sm text-emerald-100/80 mt-1">
              Executes a PlantVillage-trained MobileNetV2 ONNX graph directly inside your browser—zero cloud latency, full image privacy, and live Class Activation Map explainability.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-[11px] font-mono">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/10 border border-white/15 text-emerald-100">
              <Zap className="w-3.5 h-3.5 text-emerald-300" />
              Sub-50ms Inference
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/10 border border-white/15 text-emerald-100">
              <Lock className="w-3.5 h-3.5 text-emerald-300" />
              100% On-Device
            </span>
          </div>
        </div>
      </div>

      <main className="max-w-6xl w-full mx-auto px-5 py-7 grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        <section className="lg:col-span-5 space-y-5">
          <div className="bg-white border border-zinc-200/90 rounded-2xl p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-zinc-900">Leaf Specimen Input</h3>
              <span className="text-[11px] font-mono text-zinc-500">Drop · Browse · Ctrl+V</span>
            </div>

            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFile(e.dataTransfer.files?.[0]);
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition ${
                dragOver
                  ? 'border-emerald-600 bg-emerald-50/70'
                  : 'border-zinc-200 hover:border-emerald-700/50 bg-zinc-50/60'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => handleFile(e.target.files?.[0])}
              />
              <div className="w-10 h-10 rounded-full bg-white border border-zinc-200 flex items-center justify-center mx-auto mb-2.5 shadow-2xs">
                <Upload className="w-4 h-4 text-[#0f3d2e]" />
              </div>
              <p className="text-sm font-semibold text-zinc-800">
                Drop a leaf photo or click to upload
              </p>
              <p className="text-xs text-zinc-500 mt-0.5">
                JPG, PNG, WebP · Or paste any screenshot with Ctrl+V
              </p>
            </div>

            <div className="mt-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] uppercase tracking-wider font-semibold text-zinc-500">
                  Test Bench Specimens
                </span>
                <span className="text-[11px] font-mono text-zinc-400">1-Click Run</span>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
                {SAMPLES.map((s) => {
                  const active = imageName === s.name;
                  return (
                    <button
                      key={s.file}
                      disabled={status !== 'ready' || running}
                      onClick={() => runInference(s.file, s.name)}
                      className={`group flex items-center gap-2 p-1.5 rounded-xl border text-left transition cursor-pointer disabled:opacity-50 ${
                        active
                          ? 'border-[#0f3d2e] bg-emerald-50/80 ring-1 ring-[#0f3d2e]/20'
                          : 'border-zinc-200 bg-white hover:border-zinc-300 hover:bg-zinc-50'
                      }`}
                    >
                      <img
                        src={s.file}
                        alt={s.name}
                        className="w-9 h-9 rounded-lg object-cover shrink-0 border border-zinc-200/80"
                      />
                      <div className="min-w-0">
                        <div className="text-[11px] font-semibold text-zinc-900 truncate leading-tight">
                          {s.name}
                        </div>
                        <div className="text-[10px] font-mono text-zinc-500 truncate mt-0.5">
                          {s.tag}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="bg-white border border-zinc-200/90 rounded-2xl p-5 shadow-xs">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold text-zinc-900 flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-[#0f3d2e]" />
                Neural CAM Attribution
              </h3>
              <label className="inline-flex items-center gap-2 text-xs text-zinc-600 cursor-pointer select-none font-medium">
                <input
                  type="checkbox"
                  checked={showCam}
                  onChange={(e) => setShowCam(e.target.checked)}
                  className="accent-[#0f3d2e] rounded"
                />
                Heatmap Overlay
              </label>
            </div>

            <div className="relative aspect-square w-full rounded-xl overflow-hidden bg-zinc-100 border border-zinc-200 flex items-center justify-center">
              {imageSrc ? (
                <canvas ref={canvasRef} className="w-full h-full object-cover" />
              ) : (
                <div className="text-center px-6 text-zinc-400">
                  <Sparkles className="w-7 h-7 mx-auto mb-2 opacity-60" />
                  <p className="text-xs">Select a specimen above or upload a leaf to visualize convolutional lesion focus.</p>
                </div>
              )}
              {running && (
                <div className="absolute inset-0 bg-white/80 backdrop-blur-xs flex items-center justify-center">
                  <span className="text-xs font-medium text-zinc-800 flex items-center gap-2 bg-white px-3.5 py-2 rounded-full border border-zinc-200 shadow-xs">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-[#0f3d2e]" />
                    Executing MobileNetV2...
                  </span>
                </div>
              )}
            </div>

            {result && showCam && (
              <div className="mt-3.5 space-y-2.5">
                <div className="flex items-center gap-3 text-xs text-zinc-600">
                  <span className="shrink-0 font-medium">Intensity</span>
                  <input
                    type="range"
                    min="0.2"
                    max="1"
                    step="0.05"
                    value={camOpacity}
                    onChange={(e) => setCamOpacity(parseFloat(e.target.value))}
                    className="w-full accent-[#0f3d2e]"
                  />
                  <span className="font-mono w-9 text-right">{Math.round(camOpacity * 100)}%</span>
                </div>
                <div className="flex items-center gap-2 text-[10px] font-mono text-zinc-500">
                  <span>Low</span>
                  <div className="h-1.5 flex-1 rounded-full bg-gradient-to-r from-zinc-200 via-amber-400 to-rose-600" />
                  <span>Peak Lesion Focus</span>
                </div>
              </div>
            )}
          </div>
        </section>

        <section className="lg:col-span-7 space-y-5">
          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {!result ? (
            <div className="bg-white border border-zinc-200/90 rounded-2xl p-10 text-center shadow-xs h-full flex flex-col items-center justify-center min-h-[460px]">
              <div className="w-14 h-14 rounded-2xl bg-emerald-50 border border-emerald-200/70 text-[#0f3d2e] flex items-center justify-center mb-4">
                <Activity className="w-7 h-7" />
              </div>
              <h3 className="text-lg font-bold text-zinc-900">Ready for Edge Diagnostics</h3>
              <p className="text-sm text-zinc-500 max-w-md mt-1.5 leading-relaxed">
                Click any specimen thumbnail on the left, drop a crop leaf image, or press <kbd className="px-1.5 py-0.5 text-xs font-mono bg-zinc-100 border border-zinc-300 rounded">Ctrl+V</kbd> to run 38-class CNN inference and live lesion mapping locally.
              </p>
              <div className="grid grid-cols-3 gap-3 mt-7 w-full max-w-md text-left">
                <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/70 p-3">
                  <div className="text-[11px] font-mono text-zinc-500">DATASET</div>
                  <div className="text-sm font-bold text-zinc-900 mt-0.5">38 Classes</div>
                </div>
                <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/70 p-3">
                  <div className="text-[11px] font-mono text-zinc-500">WEIGHTS</div>
                  <div className="text-sm font-bold text-zinc-900 mt-0.5">8.97 MB</div>
                </div>
                <div className="rounded-xl border border-zinc-200/80 bg-zinc-50/70 p-3">
                  <div className="text-[11px] font-mono text-zinc-500">EXPLAINABILITY</div>
                  <div className="text-sm font-bold text-zinc-900 mt-0.5">Live CAM</div>
                </div>
              </div>
            </div>
          ) : (
            <>
              {result.oodWarning && (
                <div className="rounded-xl border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900 flex items-start gap-2.5">
                  <ShieldAlert className="w-4 h-4 shrink-0 mt-0.5 text-amber-700" />
                  <div>
                    <strong className="font-semibold">OOD Botanical Gatekeeper Warning:</strong> Low chlorophyll signature ({result.foliagePct}%) and modest confidence. Ensure the image clearly frames a single plant leaf.
                  </div>
                </div>
              )}

              <div className={`bg-white border border-zinc-200/90 border-l-4 ${result.protocol.accent} rounded-2xl p-6 shadow-xs`}>
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <div className="text-[11px] uppercase tracking-wider font-mono font-semibold text-zinc-500">
                      Primary Diagnosis
                    </div>
                    <h2 className="text-2xl font-bold tracking-tight text-zinc-950 mt-1">
                      {formatLabel(result.top.label)}
                    </h2>
                    <p className="text-sm text-zinc-600 mt-1">{result.protocol.summary}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-semibold px-2.5 py-1 rounded-full border ${result.protocol.badge}`}>
                      {result.protocol.severity}
                    </span>
                    <button
                      onClick={exportReport}
                      className="inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-lg border border-zinc-200 bg-zinc-900 hover:bg-zinc-800 text-white cursor-pointer transition"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      Export PDF
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-3 mt-5 pt-5 border-t border-zinc-100">
                  <div className="rounded-xl bg-zinc-50 border border-zinc-200/70 p-3.5">
                    <div className="text-[11px] font-mono uppercase text-zinc-500">Confidence</div>
                    <div className="text-xl font-bold font-mono text-zinc-950 mt-0.5">
                      {(result.top.score * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div className="rounded-xl bg-emerald-50/60 border border-emerald-200/70 p-3.5">
                    <div className="text-[11px] font-mono uppercase text-emerald-800/80">Edge Latency</div>
                    <div className="text-xl font-bold font-mono text-[#0f3d2e] mt-0.5">
                      {result.latencyMs.toFixed(0)} ms
                    </div>
                  </div>
                  <div className="rounded-xl bg-zinc-50 border border-zinc-200/70 p-3.5">
                    <div className="text-[11px] font-mono uppercase text-zinc-500">Foliage Ratio</div>
                    <div className="text-xl font-bold font-mono text-zinc-950 mt-0.5">
                      {result.foliagePct}%
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-zinc-200/90 rounded-2xl p-6 shadow-xs">
                <div className="flex items-center justify-between mb-3.5">
                  <h3 className="text-sm font-semibold text-zinc-900">Agronomic Action Protocol</h3>
                  <span className="text-xs font-mono px-2 py-0.5 rounded bg-zinc-100 text-zinc-600">
                    {result.protocol.pathogen}
                  </span>
                </div>
                <ul className="space-y-3 text-sm text-zinc-700">
                  {result.protocol.actions.map((step, i) => (
                    <li key={i} className="flex items-start gap-3">
                      <span className="w-5 h-5 rounded-full bg-[#0f3d2e] text-emerald-100 text-[11px] font-mono font-semibold flex items-center justify-center shrink-0 mt-0.5">
                        {i + 1}
                      </span>
                      <span className="leading-relaxed">{step}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="bg-white border border-zinc-200/90 rounded-2xl p-6 shadow-xs">
                <h3 className="text-sm font-semibold text-zinc-900 mb-3.5">Top-5 Softmax Distribution</h3>
                <div className="space-y-3">
                  {result.top5.map((item, idx) => {
                    const pct = (item.score * 100).toFixed(2);
                    return (
                      <div key={item.idx}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className={idx === 0 ? 'font-semibold text-zinc-950' : 'text-zinc-600'}>
                            {formatLabel(item.label)}
                          </span>
                          <span className="font-mono text-zinc-700">{pct}%</span>
                        </div>
                        <div className="h-2 rounded-full bg-zinc-100 overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-300 ${
                              idx === 0 ? 'bg-[#0f3d2e]' : 'bg-zinc-400'
                            }`}
                            style={{ width: `${Math.max(1.5, item.score * 100)}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </>
          )}
        </section>
      </main>

      <footer className="border-t border-zinc-200/80 bg-white/80 mt-8">
        <div className="max-w-6xl mx-auto px-5 py-4 flex flex-wrap items-center justify-between gap-3 text-xs text-zinc-500">
          <span>
            Built by <strong className="text-zinc-800">Aditya Tiwari</strong> · MobileNetV2 ONNX + React 18 + Tailwind CSS
          </span>
          <a
            href="https://github.com/AdityaIsOnGithub/plant-disease-detection-cnn"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 text-zinc-700 hover:text-zinc-950 font-medium"
          >
            <Code2 className="w-3.5 h-3.5" />
            Source & FastAPI Backend
          </a>
        </div>
      </footer>
    </div>
  );
}