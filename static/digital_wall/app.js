/* ============================================================
   Exhibition Canvas — drag/draw museum scenes -> photorealistic render
   Served by Flask (/exhibition); EXPORT_DATA is embedded by the template.
   ============================================================ */

/* ---------- Config ---------- */
const WALL_COLOR = '#efe9df';
const RENDER_URL = '/exhibition/render';
const MAX_VARIATIONS = 5;   // keep in sync with DigitalWall.VARIATION_STYLES (1 + 4)

// Museum images are loaded through the app's /thumb route: same origin, so the
// canvas stays exportable (cross-origin images without CORS headers would taint it).
function thumbUrl(url) { return '/thumb?url=' + encodeURIComponent(url); }

/* ---------- DOM ---------- */
const stage = document.getElementById('stage');
const colorInput = document.getElementById('brush-color');
const sizeInput = document.getElementById('brush-size');
const countInput = document.getElementById('variation-count');
const doneModal = document.getElementById('done-modal');
const resultModal = document.getElementById('result-modal');
const loading = document.getElementById('loading');
const doneError = document.getElementById('done-error');
const thumbs = document.getElementById('thumbs');

/* ---------- Canvas ---------- */
const canvas = new fabric.Canvas('c', {
  backgroundColor: WALL_COLOR,
  preserveObjectStacking: true,
});

function resizeCanvas() {
  canvas.setWidth(stage.clientWidth);
  canvas.setHeight(stage.clientHeight);
  canvas.renderAll();
}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

/* ---------- Data loading ---------- */
function normalize(json) {
  // The export shape: { meta, items: [{ id, title, properties: Artifact dict }] }
  const items = (json && Array.isArray(json.items)) ? json.items : [];
  return items
    .map((it) => ({
      title: it.title || '(untitled)',
      images: (it.properties && it.properties.images) || [],
    }))
    .filter((it) => it.images.length);
}

let draggedItem = null;

function renderGallery(items) {
  thumbs.innerHTML = '';
  if (!items.length) {
    thumbs.innerHTML = '<p class="hint">No items with images in this selection.</p>';
    return;
  }
  for (const item of items) {
    const card = document.createElement('div');
    card.className = 'thumb';
    card.draggable = true;

    // /thumb answers 404 for "image not available" placeholders: fall through
    // to the object's next image, and drop the card when none is left.
    const img = document.createElement('img');
    let imgIdx = 0;
    img.onerror = () => {
      imgIdx += 1;
      if (imgIdx < item.images.length) img.src = thumbUrl(item.images[imgIdx]);
      else card.remove();
    };
    img.src = thumbUrl(item.images[0]);
    img.alt = item.title;

    const title = document.createElement('div');
    title.className = 't-title';
    title.textContent = item.title;

    card.append(img, title);
    card.addEventListener('click', () => addArtwork(img.src, item.title));
    card.addEventListener('dragstart', () => { draggedItem = { image: img.src, title: item.title }; });
    thumbs.appendChild(card);
  }
}

/* ---------- Adding images ---------- */
function loadImageEl(url) {
  return new Promise((resolve, reject) => {
    const el = new Image();
    el.onload = () => resolve(el);
    el.onerror = () => reject(new Error('image load failed'));
    el.src = url;
  });
}

async function addArtwork(url, title, pos) {
  let el;
  try {
    el = await loadImageEl(url);
  } catch (_) {
    return;
  }

  const img = new fabric.Image(el);
  img.scaleToWidth(220);
  const w = img.getScaledWidth();
  const h = img.getScaledHeight();
  const left = pos ? pos.x - w / 2 : 60 + Math.random() * 140;
  const top = pos ? pos.y - h / 2 : 50 + Math.random() * 90;
  img.set({ left, top });
  canvas.add(img);

  if (title) {
    const caption = new fabric.Textbox(title, {
      left,
      top: top + h + 8,
      width: Math.max(w, 160),
      fontSize: 16,
      fill: '#2a2a2a',
      textAlign: 'center',
      fontFamily: 'Georgia, serif',
    });
    canvas.add(caption);
  }

  canvas.setActiveObject(img);
  canvas.requestRenderAll();
}

/* ---------- Tools ---------- */
let currentTool = 'select';

function setTool(tool) {
  currentTool = tool;
  document.querySelectorAll('.tool[data-tool]').forEach((b) =>
    b.classList.toggle('active', b.dataset.tool === tool));

  if (tool === 'pen' || tool === 'eraser') {
    canvas.isDrawingMode = true;
    const brush = new fabric.PencilBrush(canvas);
    brush.width = Number(sizeInput.value);
    brush.color = tool === 'eraser' ? '#000000' : colorInput.value;  // color irrelevant for eraser
    canvas.freeDrawingBrush = brush;
  } else {
    canvas.isDrawingMode = false;
  }
}

document.getElementById('tool-select').addEventListener('click', () => setTool('select'));
document.getElementById('tool-pen').addEventListener('click', () => setTool('pen'));
document.getElementById('tool-eraser').addEventListener('click', () => setTool('eraser'));

colorInput.addEventListener('input', () => {
  if (canvas.isDrawingMode && currentTool === 'pen') canvas.freeDrawingBrush.color = colorInput.value;
});
sizeInput.addEventListener('input', () => {
  if (canvas.isDrawingMode) canvas.freeDrawingBrush.width = Number(sizeInput.value);
});

document.getElementById('tool-text').addEventListener('click', () => {
  setTool('select');
  const t = new fabric.Textbox('Double-click to edit', {
    left: canvas.getWidth() / 2 - 100,
    top: canvas.getHeight() / 2 - 20,
    width: 220,
    fontSize: 22,
    fill: colorInput.value,
    fontFamily: 'Georgia, serif',
  });
  canvas.add(t);
  canvas.setActiveObject(t);
  canvas.requestRenderAll();
});

/* ---------- Undo / clear ----------
   Undo keeps whole-canvas JSON snapshots, one per change (the last
   HISTORY_MAX); the newest is also saved for the next visit (persistWall).
   isRestoring stops the restore itself from being recorded as a change. */
const HISTORY_MAX = 30;
let history = [];
let isRestoring = false;

function saveState() {
  if (isRestoring) return;
  history.push(JSON.stringify(canvas.toJSON()));
  if (history.length > HISTORY_MAX) history.shift();
  persistWall(history[history.length - 1]);
}

canvas.on('object:added', (e) => {
  if (e.target && e.target.type === 'path') return;   // paths handled in path:created
  saveState();
});
canvas.on('object:modified', saveState);
canvas.on('object:removed', saveState);
canvas.on('path:created', (e) => {
  if (currentTool === 'eraser') e.path.set({ globalCompositeOperation: 'destination-out' });
  canvas.requestRenderAll();
  saveState();
});

function undo() {
  if (history.length <= 1) return;
  history.pop();                                // drop current state
  const prev = history[history.length - 1];
  isRestoring = true;
  canvas.loadFromJSON(prev, () => {
    canvas.backgroundColor = WALL_COLOR;
    canvas.renderAll();
    isRestoring = false;
  });
  persistWall(prev);
}

function clearCanvas() {
  isRestoring = true;
  canvas.clear();
  canvas.backgroundColor = WALL_COLOR;
  canvas.renderAll();
  isRestoring = false;
  saveState();
}

document.getElementById('btn-undo').addEventListener('click', undo);
document.getElementById('btn-clear').addEventListener('click', () => {
  if (confirm('Clear the entire canvas?')) clearCanvas();
});

/* ---------- Keyboard: delete selection, ctrl/cmd+z ---------- */
window.addEventListener('keydown', (e) => {
  const typing = /^(INPUT|TEXTAREA)$/.test(document.activeElement && document.activeElement.tagName);
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
    if (typing) return;
    e.preventDefault();
    undo();
    return;
  }
  if (e.key === 'Delete' || e.key === 'Backspace') {
    if (typing) return;
    const active = canvas.getActiveObject();
    if (!active || active.isEditing) return;
    e.preventDefault();
    if (active.type === 'activeSelection') active.forEachObject((o) => canvas.remove(o));
    else canvas.remove(active);
    canvas.discardActiveObject();
    canvas.requestRenderAll();
  }
});

/* ---------- Drag a thumbnail onto the wall ---------- */
stage.addEventListener('dragover', (e) => {
  e.preventDefault();
  e.dataTransfer.dropEffect = 'copy';
});
stage.addEventListener('drop', (e) => {
  e.preventDefault();
  if (!draggedItem) return;
  const rect = canvas.upperCanvasEl.getBoundingClientRect();
  addArtwork(draggedItem.image, draggedItem.title, { x: e.clientX - rect.left, y: e.clientY - rect.top });
  draggedItem = null;
});

/* ---------- Props (pillar / frame): click or drag onto the wall ---------- */
document.querySelectorAll('#props .prop').forEach((el) => {
  const url = el.src;
  el.addEventListener('click', () => addArtwork(url, '', { x: 220, y: canvas.getHeight() - 200 }));
  el.addEventListener('dragstart', () => { draggedItem = { image: url, title: '' }; });
});

/* ---------- File inputs ---------- */
// Load JSON accepts a file saved with the curation page's Export button.
document.getElementById('load-json').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => {
    try {
      renderGallery(normalize(JSON.parse(fr.result)));
    } catch (_) {
      alert('Could not parse that JSON file.');
    }
  };
  fr.readAsText(file);
  e.target.value = '';
});

document.getElementById('add-image').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fr = new FileReader();
  fr.onload = () => addArtwork(fr.result, file.name.replace(/\.[^.]+$/, ''));
  fr.readAsDataURL(file);
  e.target.value = '';
});

/* ---------- Done -> server-side render ---------- */
function showDoneError(msg) {
  doneError.textContent = msg;
  doneError.classList.remove('hidden');
}

document.getElementById('btn-done').addEventListener('click', () => {
  setTool('select');
  doneError.classList.add('hidden');
  doneModal.classList.remove('hidden');
});
document.getElementById('done-cancel').addEventListener('click', () => doneModal.classList.add('hidden'));
document.getElementById('result-back').addEventListener('click', () => resultModal.classList.add('hidden'));
document.getElementById('btn-generate').addEventListener('click', generate);

async function generate() {
  const desc = document.getElementById('scene-desc').value.trim();
  // How many variations to render: one faithful + creative re-imaginings.
  const count = Math.min(MAX_VARIATIONS, Math.max(1, Math.round(Number(countInput.value)) || 1));
  doneError.classList.add('hidden');

  // Flatten the scene to a PNG blob.
  canvas.discardActiveObject();
  canvas.renderAll();
  let blob;
  try {
    const dataURL = canvas.toDataURL({ format: 'png' });
    blob = await (await fetch(dataURL)).blob();
  } catch (_) {
    showDoneError('Could not export the canvas — one of its images blocked it (no CORS).');
    return;
  }

  doneModal.classList.add('hidden');
  loading.classList.remove('hidden');
  const loadingText = loading.querySelector('p');
  try {
    const images = [];
    for (let i = 0; i < count; i++) {
      loadingText.textContent =
        count > 1 ? 'Generating ' + (i + 1) + ' of ' + count + '…' : 'Transforming your scene…';
      images.push(await callRender(blob, desc, i));
    }
    showResults(images);
  } catch (err) {
    loading.classList.add('hidden');
    doneModal.classList.remove('hidden');
    showDoneError(err.message || 'Generation failed.');
  } finally {
    loadingText.textContent = 'Transforming your scene…';
  }
}

// Returns an object URL for the rendered PNG.
async function callRender(blob, description, variation) {
  const form = new FormData();
  form.append('scene', blob, 'scene.png');
  form.append('description', description);
  form.append('variation', String(variation));

  const res = await fetch(RENDER_URL, { method: 'POST', body: form });
  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.error || 'Request failed (' + res.status + ').');
  }
  return URL.createObjectURL(await res.blob());
}

/* ---------- Result album (click through the variations) ---------- */
let resultImages = [];
let resultIndex = 0;

function showResults(images) {
  loading.classList.add('hidden');
  resultImages.forEach((u) => URL.revokeObjectURL(u));   // free the previous album
  resultImages = images;
  resultIndex = 0;
  renderAlbum();
  resultModal.classList.remove('hidden');
}

function renderAlbum() {
  const total = resultImages.length;
  const src = resultImages[resultIndex];
  document.getElementById('result-img').src = src;
  const dl = document.getElementById('btn-download');
  dl.href = src;
  dl.setAttribute('download', 'exhibition-' + (resultIndex + 1) + '.png');
  document.getElementById('album-counter').textContent = (resultIndex + 1) + ' / ' + total;
  const single = total <= 1;                          // hide arrows for a single image
  document.getElementById('album-prev').classList.toggle('hidden', single);
  document.getElementById('album-next').classList.toggle('hidden', single);
}

document.getElementById('album-prev').addEventListener('click', () => {
  resultIndex = (resultIndex - 1 + resultImages.length) % resultImages.length;
  renderAlbum();
});
document.getElementById('album-next').addEventListener('click', () => {
  resultIndex = (resultIndex + 1) % resultImages.length;
  renderAlbum();
});

/* ---------- Keep the wall across visits ----------
   The wall is saved in sessionStorage (this tab only, gone when the tab
   closes), so going back to the curation page to add objects and returning
   doesn't lose the composition. */
const WALL_KEY = 'digital-wall';

function persistWall(json) {
  try {
    sessionStorage.setItem(WALL_KEY, json);
  } catch (_) { /* storage full (e.g. large uploaded images) or blocked: keep working without */ }
}

function restoreWall() {
  let saved = null;
  try { saved = sessionStorage.getItem(WALL_KEY); } catch (_) {}
  if (!saved) {
    saveState();          // baseline empty-canvas state for undo
    return;
  }
  isRestoring = true;
  canvas.loadFromJSON(saved, () => {
    canvas.backgroundColor = WALL_COLOR;
    canvas.renderAll();
    isRestoring = false;
    saveState();          // the restored wall is the undo baseline
  });
}

/* ---------- Boot ---------- */
setTool('select');
restoreWall();
renderGallery(normalize(EXPORT_DATA));
