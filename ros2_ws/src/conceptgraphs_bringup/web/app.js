const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

const state = {
  mode: 'auto', history: [], contextObject: null, contextLabel: null,
  images: [], imageTab: 'overview', busy: false, status: null
};

const pretty = value => (value ?? 'unknown').toString().replaceAll('_', ' ');
const title = value => pretty(value).replace(/\b\w/g, letter => letter.toUpperCase());
const finite = value => Number.isFinite(Number(value));
const formattedNumber = (value, digits = 2) => finite(value) ? Number(value).toFixed(digits) : 'Unknown';
const short = (value, length = 22) => value && value.length > length ? `${value.slice(0, length - 1)}…` : value;
const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, character => ({
  '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
}[character]));

function setBusy(value) {
  state.busy = value;
  $('#loading').classList.toggle('hidden', !value);
  const send = $('#search-form button[type="submit"]');
  send.disabled = value;
  send.textContent = value ? 'Searching' : 'Send';
}

async function loadStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Status unavailable');
    state.status = await response.json();
    $('#dataset-name').textContent = state.status.dataset_id || 'Greenhouse dataset';
    $('#aisle-count').textContent = state.status.aisle_count ?? '–';
    $('#object-count').textContent = state.status.object_count ?? '–';
    $('#view-count').textContent = state.status.view_count ?? '–';
  } catch (_) {
    $('#dataset-name').textContent = 'Dataset connected';
  }
}

function routingLabel(source) {
  return ({rule: 'Rule-based routing', gemma_classifier: 'Gemma classifier', manual: 'Manual mode'})[source] || title(source || 'automatic');
}

function answerLabel(source) {
  return ({gemma: 'Gemma answer', template_fallback: 'Verified fallback', template: 'Verified template', rule: 'Verified rule'})[source] || title(source || 'verified answer');
}

function setMode(nextMode) {
  state.mode = nextMode;
  $$('[data-mode]').forEach(button => button.classList.toggle('active', button.dataset.mode === nextMode));
  $('#composer-mode').textContent = ({auto: 'Auto routing', chat: 'General conversation', map: 'Map query only'})[nextMode];
}

$$('[data-mode]').forEach(button => button.addEventListener('click', () => setMode(button.dataset.mode)));
$$('[data-query]').forEach(button => button.addEventListener('click', () => {
  $('#query').value = button.dataset.query;
  $('#query').focus();
}));

function addBubble(role, text, meta, error = false) {
  const article = document.createElement('article');
  article.className = `bubble ${role}${error ? ' error' : ''}`;
  const body = document.createElement('div');
  body.textContent = text;
  const detail = document.createElement('small');
  detail.textContent = meta;
  article.append(body, detail);
  $('#chat-history').append(article);
  $('#chat-history').scrollTop = $('#chat-history').scrollHeight;
}

function updateContext(object) {
  state.contextObject = object?.object_id || null;
  state.contextLabel = object?.canonical_label || null;
  $('#active-context').classList.toggle('hidden', !state.contextObject);
  $('#context-label').textContent = title(state.contextLabel || '');
  $('#context-id').textContent = state.contextObject || '';
}

function clearContext() { updateContext(null); }
$('#clear-context').addEventListener('click', clearContext);

function resetConversation() {
  state.history = [];
  clearContext();
  $$('.bubble').forEach(element => element.remove());
  $('#suggestions').classList.remove('hidden');
  $('#results').classList.add('hidden');
  $('#not-found').classList.add('hidden');
  $('#empty').classList.remove('hidden');
  $('#query').value = '';
  $('#query').focus();
}
$('#new-conversation').addEventListener('click', resetConversation);

async function submitQuery(query) {
  if (!query || state.busy) return;
  const requestHistory = state.history.slice(-10);
  addBubble('user', query, title(state.mode));
  state.history.push({role: 'user', content: query});
  $('#suggestions').classList.add('hidden');
  $('#empty').classList.add('hidden');
  $('#not-found').classList.add('hidden');
  setBusy(true);
  try {
    const response = await fetch('/api/query', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({query, mode: state.mode, history: requestHistory, context_object_id: state.contextObject, top_k: 5})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The map query failed');
    state.history.push({role: 'assistant', content: data.answer});
    addBubble('assistant', data.answer, `${answerLabel(data.answer_source)} · ${routingLabel(data.routing?.source)}`);
    render(data);
    if (window.innerWidth <= 820) toggleConversation(false);
  } catch (error) {
    addBubble('assistant', error.message, 'Request error', true);
    renderError(error.message);
  } finally {
    setBusy(false);
  }
}

$('#search-form').addEventListener('submit', event => {
  event.preventDefault();
  const query = $('#query').value.trim();
  $('#query').value = '';
  submitQuery(query);
});
$('#query').addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#search-form').requestSubmit();
  }
});

function render(data) {
  if (data.response_type === 'general_chat') {
    clearContext();
    return;
  }
  if (data.status !== 'success' || !data.matched_object) {
    clearContext();
    $('#not-found-text').textContent = data.answer || 'Try adding an object category or aisle.';
    $('#not-found').classList.remove('hidden');
    return;
  }
  const object = data.matched_object;
  const traversal = object.traversability || {};
  const results = data.matched_objects || [];
  if (results.length === 1 && !data.retrieval?.ambiguous) updateContext(object);
  else clearContext();

  $('#answer').textContent = data.answer;
  const similarity = object.similarity;
  const latency = data.retrieval?.latency_sec;
  $('#confidence').textContent = [finite(similarity) ? `Similarity ${Number(similarity).toFixed(3)}` : '', finite(latency) ? `${Number(latency).toFixed(1)} s` : ''].filter(Boolean).join(' · ');
  $('#answer-source').textContent = answerLabel(data.answer_source);
  $('#routing-source').textContent = routingLabel(data.routing?.source);
  $('#intent-badge').textContent = `Intent: ${title(data.intent)}`;

  renderBreadcrumb(object, data.scene_graph);
  renderObjectList(results, object.object_id);
  renderFacts(object);
  renderImages(data.related_images || []);
  renderGraph(data.scene_graph || {nodes: [], edges: []});
  renderTraversal(traversal);
  $('#results').classList.remove('hidden');
  $('#not-found').classList.add('hidden');
}

function renderBreadcrumb(object, graph) {
  const greenhouse = graph?.nodes?.find(node => node.type === 'Greenhouse')?.label || state.status?.greenhouse_id || 'Greenhouse';
  const values = [greenhouse, object.aisle_id, object.canonical_label].filter(Boolean);
  $('#breadcrumb').innerHTML = values.map((value, index) => `${index ? '<i>/</i>' : ''}<${index === values.length - 1 ? 'b' : 'span'}>${escapeHtml(title(value))}</${index === values.length - 1 ? 'b' : 'span'}>`).join('');
}

function renderObjectList(objects, selectedId) {
  const container = $('#object-list');
  if (objects.length < 2) {
    container.classList.add('hidden'); container.innerHTML = ''; return;
  }
  container.innerHTML = objects.map(object => `
    <button class="object-chip ${object.object_id === selectedId ? 'active' : ''}" type="button" data-object-id="${escapeHtml(object.object_id)}">
      <strong>${escapeHtml(title(object.canonical_label))}</strong>
      <span>${escapeHtml(object.object_id)} · ${escapeHtml(title(object.aisle_id))}</span>
      <span>${escapeHtml(formatCentroid(object.centroid_3d))}</span>
    </button>`).join('');
  container.classList.remove('hidden');
  $$('.object-chip').forEach(button => button.addEventListener('click', () => inspectObject(button.dataset.objectId)));
}

function factRows(rows) {
  const available = rows.filter(([, value]) => value !== null && value !== undefined && value !== '');
  return available.length ? available.map(([key, value]) => `<div><dt>${escapeHtml(key)}</dt><dd>${escapeHtml(pretty(value))}</dd></div>`).join('') : '<div><dt>Status</dt><dd>Not available</dd></div>';
}

function formatCentroid(centroid) {
  return Array.isArray(centroid) && centroid.length >= 3 ? centroid.slice(0, 3).map(value => formattedNumber(value)).join(', ') : 'Unknown';
}

function renderFacts(object) {
  const physical = object.physical_properties || {};
  const attributes = object.attributes || {};
  const traversal = object.traversability || {};
  $('#object-label').textContent = title(object.canonical_label);
  $('#object-id').textContent = object.object_id || 'Unknown ID';
  $('#frame-id').textContent = `Frame: ${object.frame_id || state.status?.frame_id || 'unknown'}`;
  $('#aisle').textContent = title(object.aisle_id || 'Unassigned');
  $('#identity-facts').innerHTML = factRows([
    ['Category', attributes.category || object.canonical_label], ['Color', attributes.color],
    ['Material', attributes.material], ['Shape', attributes.shape], ['State', attributes.state]
  ]);
  $('#physical-facts').innerHTML = factRows([
    ['Rigidity', physical.rigidity], ['Deformability', physical.deformability],
    ['Compressibility', physical.compressibility], ['Fragility', physical.fragility],
    ['Mass class', physical.estimated_mass_class || physical.mass_class],
    ['Anchored', physical.anchored_to_ground ?? physical.anchored],
    ['Object traversal', traversal.object_semantic_class]
  ]);
  $('#geometry-facts').innerHTML = factRows([
    ['Centroid (X, Y, Z)', formatCentroid(object.centroid_3d)], ['Coordinate frame', object.frame_id || state.status?.frame_id],
    ['Bounding box min', formatCentroid(object.bbox_3d_min)], ['Bounding box max', formatCentroid(object.bbox_3d_max)],
    ['Aisle', object.aisle_id]
  ]);
}

function renderImages(images) {
  state.images = images;
  $('#image-count').textContent = `${images.length} images`;
  renderImageTab(state.imageTab);
}

function renderImageTab(tab) {
  state.imageTab = tab;
  $$('[data-image-tab]').forEach(button => button.classList.toggle('active', button.dataset.imageTab === tab));
  let selected;
  if (tab === 'overlay') selected = state.images.filter(image => image.kind === 'overlay');
  else if (tab === 'views') selected = state.images.filter(image => image.kind === 'view');
  else {
    const crop = state.images.find(image => image.kind === 'crop');
    const views = state.images.filter(image => image.kind === 'view').slice(0, crop ? 3 : 4);
    selected = [crop, ...views].filter(Boolean);
  }
  const gallery = $('#image-gallery');
  if (!selected.length) {
    gallery.innerHTML = `<div class="gallery-empty">No ${escapeHtml(tab)} imagery is available for this object.</div>`;
    return;
  }
  gallery.innerHTML = selected.map((image, index) => `
    <figure class="image-tile" data-image-index="${state.images.indexOf(image)}" tabindex="0" role="button" aria-label="Expand ${escapeHtml(image.label)}">
      <img src="${escapeHtml(image.url)}" alt="${escapeHtml(image.label)}" loading="lazy">
      <figcaption>${escapeHtml(image.label)}</figcaption>
    </figure>`).join('');
  $$('.image-tile').forEach(tile => {
    const open = () => openImage(Number(tile.dataset.imageIndex));
    tile.addEventListener('click', open);
    tile.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') open(); });
  });
}
$$('[data-image-tab]').forEach(button => button.addEventListener('click', () => renderImageTab(button.dataset.imageTab)));

function openImage(index) {
  const image = state.images[index];
  if (!image) return;
  $('#dialog-image').src = image.url;
  $('#dialog-image').alt = image.label;
  $('#dialog-caption').textContent = image.view_id ? `${image.label} · ${image.view_id}` : image.label;
  $('#image-dialog').showModal();
}
$('#close-image').addEventListener('click', () => $('#image-dialog').close());
$('#image-dialog').addEventListener('click', event => { if (event.target === $('#image-dialog')) $('#image-dialog').close(); });

function renderTraversal(traversal) {
  const objectClass = traversal.object_semantic_class || 'unknown';
  const path = traversal.path_assessment || 'uncertain';
  const clearance = traversal.available_clearance_m;
  const required = traversal.required_clearance_m;
  $('#object-traversability').textContent = title(objectClass);
  $('#path-assessment').textContent = title(path);
  $('#object-traversability').className = /non.?traversable|obstacle|blocked/i.test(objectClass) ? 'blocked' : /traversable|passable/i.test(objectClass) ? 'passable' : 'uncertain';
  $('#path-assessment').className = /not.?passable|blocked/i.test(path) ? 'blocked' : /passable/i.test(path) ? 'passable' : 'uncertain';
  $('#clearance').textContent = finite(clearance) ? `Estimated side clearance: ${formattedNumber(clearance)} m${finite(required) ? `; required: ${formattedNumber(required)} m.` : '.'}` : 'No geometric clearance estimate is available.';
  $('#assessment-source').textContent = traversal.assessment_source ? title(traversal.assessment_source) : 'Semantic evidence; geometry when available';
  $('#reason').textContent = traversal.reasoning_summary || 'No reasoning summary is available for this object.';
  const requires = traversal.requires_geometric_verification !== false;
  $('#verification-badge').textContent = requires ? 'Verification required' : 'Verified estimate';
}

function renderGraph(graph) {
  const svg = $('#scene-graph');
  const nodes = graph.nodes || [], edges = graph.edges || [];
  if (!nodes.length) { svg.innerHTML = '<text x="30" y="50" class="node-label">No graph relations available</text>'; return; }
  const byType = type => nodes.filter(node => node.type === type);
  const greenhouse = byType('Greenhouse')[0], aisle = byType('Aisle')[0];
  const objects = byType('Object'), views = byType('View');
  const positions = {};
  if (greenhouse) positions[greenhouse.id] = [85, 195];
  if (aisle) positions[aisle.id] = [235, 195];
  objects.forEach((node, index) => { positions[node.id] = [430, 65 + index * Math.min(65, 270 / Math.max(1, objects.length - 1))]; });
  views.forEach((node, index) => { positions[node.id] = [675, 55 + index * Math.min(68, 275 / Math.max(1, views.length - 1))]; });
  let html = edges.filter(edge => positions[edge.source] && positions[edge.target]).map(edge => {
    const a = positions[edge.source], b = positions[edge.target];
    const focused = edge.role === 'focus' || edge.role === 'evidence';
    return `<line class="edge ${focused ? 'focus-edge' : ''}" x1="${a[0]}" y1="${a[1]}" x2="${b[0]}" y2="${b[1]}"></line><text class="edge-label" x="${(a[0] + b[0]) / 2}" y="${(a[1] + b[1]) / 2 - 5}">${escapeHtml(edge.relation)}</text>`;
  }).join('');
  html += nodes.filter(node => positions[node.id]).map(node => graphNode(node, positions[node.id])).join('');
  svg.innerHTML = html;
  svg.querySelectorAll('[data-object-node]').forEach(group => group.addEventListener('click', () => inspectObject(group.dataset.objectNode)));
}

function graphNode(node, position) {
  const [x, y] = position;
  const focused = node.role === 'focus' || node.focus;
  const sibling = node.role === 'sibling';
  const className = node.type === 'Greenhouse' ? 'node-greenhouse' : node.type === 'Aisle' ? 'node-aisle' : node.type === 'View' ? 'node-view' : focused ? 'node-object' : 'node-sibling';
  const width = node.type === 'Object' ? 145 : node.type === 'View' ? 118 : 120;
  const dark = className === 'node-object';
  const clickable = node.type === 'Object';
  return `<g class="graph-node ${clickable ? 'clickable' : ''}" ${clickable ? `data-object-node="${escapeHtml(node.id)}"` : ''}>
    <rect class="${className}" x="${x - width / 2}" y="${y - 26}" rx="9" width="${width}" height="52"></rect>
    <text class="node-type ${dark ? 'on-dark' : ''}" text-anchor="middle" x="${x}" y="${y - 7}">${escapeHtml(node.type.toUpperCase())}</text>
    <text class="node-label ${dark ? 'on-dark' : ''}" text-anchor="middle" x="${x}" y="${y + 10}">${escapeHtml(short(title(node.label || node.id), 24))}</text>
  </g>`;
}

function inspectObject(objectId) {
  if (!objectId || state.busy) return;
  submitQuery(`Show me ${objectId}`);
}

function renderError(message) {
  $('#not-found-text').textContent = message;
  $('#not-found').classList.remove('hidden');
}

function toggleConversation(force) {
  const panel = $('#conversation-panel');
  const open = force ?? !panel.classList.contains('open');
  panel.classList.toggle('open', open);
  $('#conversation-toggle').setAttribute('aria-expanded', String(open));
}
$('#conversation-toggle').addEventListener('click', () => toggleConversation());

loadStatus();
