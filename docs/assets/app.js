// research-everything viz — vanilla JS, no build step.
// Notebook / arXiv template. Reads ./data.json, renders:
//   paper head + abstract · rail (metrics/tally/refs) · research map (DAG)
//   · leaderboard table · progress chart · milestone logbook · methodology
//   · references · orbit detail panel (numbered annotations).
// Hash route: #/ → index. Node click → slide-in side panel.

const app = document.getElementById('app');

// Active eval-version. Module-level so toggle clicks + URL params all flow
// through the same source of truth. Set after data loads.
let activeEval = null;

(async function main() {
  let data;
  try {
    const res = await fetch('./data.json', { cache: 'no-store' });
    data = await res.json();
  } catch (e) {
    app.innerHTML = `<p class="empty">No data.json found. Run <code>/publish</code>.</p>`;
    return;
  }
  document.title = data.campaign?.title || 'Campaign';
  cytoscape.use(cytoscapeDagre);
  normalizeEvalVersions(data);
  activeEval = pickInitialEval(data);
  applyActiveEval(data);
  window.addEventListener('hashchange', () => render(data));
  app.addEventListener('click', (e) => {
    const pill = e.target.closest('[data-eval-pill]');
    if (!pill) return;
    const v = pill.getAttribute('data-eval-pill');
    if (!v || v === activeEval) return;
    activeEval = v;
    const url = new URL(location.href);
    url.searchParams.set('eval', v);
    history.replaceState(null, '', url);
    applyActiveEval(data);
    render(data);
  });
  render(data);
})();

// ====== Eval-version selection ============================================
// Backfill every orbit with a metrics map so legacy single-metric data.json
// files still render. Source of truth for the toggle UI lives on
// data.campaign.eval_versions; if publish.py didn't populate it, derive it
// from whatever versions the orbits expose.
function normalizeEvalVersions(data) {
  const c = data.campaign || (data.campaign = {});
  const orbits = data.orbits || [];
  const fallbackVer = c.current_eval_version || c.eval_version || 'eval-v?';

  for (const o of orbits) {
    if (!o.metrics || typeof o.metrics !== 'object') {
      o.metrics = {};
      const ver = o.eval_version || fallbackVer;
      if (typeof o.metric === 'number' && Number.isFinite(o.metric)) {
        o.metrics[ver] = o.metric;
      }
    }
  }

  if (!Array.isArray(c.eval_versions) || c.eval_versions.length === 0) {
    const seen = new Set();
    for (const o of orbits) {
      for (const v of Object.keys(o.metrics || {})) seen.add(v);
    }
    if (seen.size === 0) seen.add(fallbackVer);
    c.eval_versions = [...seen].sort().map(id => ({ id }));
  }
  if (!c.current_eval_version) {
    c.current_eval_version = c.eval_version || c.eval_versions[c.eval_versions.length - 1].id;
  }
}

function pickInitialEval(data) {
  const versions = (data.campaign?.eval_versions || []).map(v => v.id);
  const url = new URL(location.href);
  const fromUrl = url.searchParams.get('eval');
  if (fromUrl && versions.includes(fromUrl)) return fromUrl;
  return data.campaign?.current_eval_version || versions[versions.length - 1] || null;
}

// Project the active eval-version onto the per-orbit `metric` field and the
// campaign `best`, so every existing renderer (leaderboard, DAG, charts,
// detail panel) keeps working without per-callsite changes.
function applyActiveEval(data) {
  const c = data.campaign || {};
  const v = activeEval;
  for (const o of data.orbits || []) {
    const m = (o.metrics || {})[v];
    o.metric = (typeof m === 'number' && Number.isFinite(m)) ? m : null;
  }
  // Campaign `best`: prefer the precomputed best_by_eval entry; recompute
  // locally if missing (e.g. legacy data.json or sparse version).
  const baseBest = c.best || {};
  const direction = baseBest.direction || 'min';
  const target = (typeof baseBest.target === 'number') ? baseBest.target : undefined;
  let bestEntry = (c.best_by_eval || {})[v] || null;
  if (!bestEntry) {
    const finite = (data.orbits || []).filter(o => typeof o.metric === 'number' && Number.isFinite(o.metric));
    if (finite.length) {
      const winner = direction.startsWith('max')
        ? finite.reduce((a, b) => (a.metric > b.metric ? a : b))
        : finite.reduce((a, b) => (a.metric < b.metric ? a : b));
      bestEntry = { orbit: winner.name, metric: winner.metric };
    }
  }
  c.best = bestEntry
    ? { ...bestEntry, direction, ...(target !== undefined ? { target } : {}) }
    : { direction, ...(target !== undefined ? { target } : {}) };
  // Surface the active version under the same key the existing UI reads.
  c.eval_version = v;
}

function render(data) {
  app.innerHTML = renderIndex(data);
  mountDag(data);
  mountProgress(data);
  mountConceptGraph(data);
}

// ====== Top-level render ==================================================
function renderIndex(data) {
  const c = data.campaign || {};
  const orbits = data.orbits || [];
  const counts = tallyStatus(orbits);
  const best = c.best;
  const direction = c.best?.direction === 'max' ? 'max' : 'min';
  const sorted = [...orbits].sort((a, b) => metricRank(a, c) - metricRank(b, c));

  // Authors = unique agent commenters across all orbits; fallback to defaults.
  const agentSet = new Set();
  for (const o of orbits) {
    for (const cm of (o.issue_comments || [])) if (cm.author) agentSet.add(cm.author);
  }
  const agents = [...agentSet].slice(0, 8);
  const authors = agents.length
    ? agents.map(a => `<span class="agent">${esc(a)}</span>`).join(', ')
    : ['orbit-agent', 'reviewer', 'verifier'].map(a => `<span class="agent">${esc(a)}</span>`).join(', ');

  const today = new Date().toISOString().slice(0, 10);

  // Masthead = framed card with identity only (eyebrow, title, authors,
  // repo). Abstract + teaser + background live below the metadata strip
  // so the reader sees "what · who · scoreboard · abstract" in that
  // order — the canele-pitch notebook layout.
  const headerBlock = `
    <header class="paper-head">
      <div class="id-line">
        <span>research-everything · campaign · ${esc(c.eval_version || 'eval-v?')}</span>
        <span>${esc(today)} · rev.${orbits.length}</span>
      </div>
      <h1>${esc(c.title || 'Untitled Campaign')}</h1>
      <p class="authors">${authors}</p>
      ${c.repo ? `<p class="affiliation">repo: <a href="https://github.com/${esc(c.repo)}" target="_blank" rel="noopener">${esc(c.repo)}</a> · branch ancestry across ${orbits.length} orbit${orbits.length === 1 ? '' : 's'}</p>` : ''}
    </header>
  `;

  // Goal + Metric history come from problem.md (parsed by publish.py via
  // read_goal_and_history). When present they replace the older `c.problem`
  // render — same source, less duplication. Falls back to the legacy
  // `renderAbstract(c, best)` if the campaign hasn't adopted the new
  // problem.md scaffolding (## Goal section + <!-- metric: ... --> header).
  const goalBlock = c.goal ? `
    <section class="goal-section">
      <h2 class="s"><span class="n">§0</span>Goal</h2>
      <div class="goal-body">${renderMarkdown(c.goal)}</div>
      ${c.metric_header ? `
        <p class="metric-header-line">
          metric <span class="mono">${esc(c.metric_header.metric)}</span>
          · target <span class="mono">${esc(c.metric_header.target)}</span>
          · eval <span class="mono">${esc(c.metric_header.eval)}</span>
        </p>` : ''}
    </section>
  ` : '';

  const metricHistoryBlock = (c.eval_versions && c.eval_versions.some(v => v.description)) ? `
    <section class="metric-history">
      <h3>Metric history</h3>
      <ul>
        ${c.eval_versions.map(v =>
          v.description
            ? `<li><span class="mono">${esc(v.id)}</span>: ${esc(v.description)}</li>`
            : `<li><span class="mono">${esc(v.id)}</span></li>`
        ).join('')}
      </ul>
    </section>
  ` : '';

  // When `c.goal` is present, suppress the duplicated `c.problem` abstract
  // and `c.background` block — both render the same framework prose (the
  // outsider audit measured ~9.5 KB of overlap between background +
  // problem + the page header). Goal + Metric history is the canonical
  // top-of-page header.
  const introBlock = c.goal ? `
    ${goalBlock}
    ${metricHistoryBlock}
    ${c.teaser_image ? `<img src="${esc(c.teaser_image)}" class="teaser-hero" alt="">` : ''}
  ` : `
    ${c.problem ? renderAbstract(c, best) : ''}
    ${c.teaser_image ? `<img src="${esc(c.teaser_image)}" class="teaser-hero" alt="">` : ''}
    ${c.background ? `<details class="background-section"><summary>Research Background</summary><div class="bg-body">${renderMarkdown(c.background)}</div></details>` : ''}
  `;

  const railBlock = `
    <aside class="rail">
      <div class="block">
        <div class="hd">Campaign metrics</div>
        <dl>
          ${best?.metric != null ? `<dt>best</dt><dd class="best">${fmt(best.metric)}</dd>` : ''}
          ${best?.target != null ? `<dt>target</dt><dd>${fmt(best.target)}</dd>` : ''}
          <dt>direction</dt><dd>${direction === 'min' ? '↓ min' : '↑ max'}</dd>
          <dt>orbits</dt><dd>${orbits.length}</dd>
          <dt>eval</dt><dd>${renderEvalToggle(c)}</dd>
        </dl>
      </div>
      ${counts.total ? `<div class="block">
        <div class="hd">Status tally</div>
        <dl>
          ${counts.graduated ? `<dt>winner</dt><dd>${counts.graduated}</dd>` : ''}
          ${counts.active ? `<dt>active</dt><dd>${counts.active}</dd>` : ''}
          ${counts['dead-end'] ? `<dt>dead-end</dt><dd>${counts['dead-end']}</dd>` : ''}
        </dl>
      </div>` : ''}
      ${orbits.filter(o => typeof o.metric === 'number' && Number.isFinite(o.metric)).length >= 2 ? `
        <div class="block progress-mini">
          <div class="hd">Progress</div>
          <div class="chart-wrap-mini"><canvas id="progress-scatter-mini"></canvas></div>
        </div>
      ` : ''}
      ${c.references?.length ? renderRailReferences(c.references) : ''}
    </aside>
  `;

  const researchMap = orbits.length ? `
    <section class="wide-section">
      <h2 class="s"><span class="n">§1</span>Research map</h2>
      <div id="dag"></div>
      <p class="tbl-caption"><span class="num">Fig. 1</span>Orbit lineage DAG. Node fill intensity = metric quality; node size scales with quality; ★ marks the winner. Click a node for orbit detail; Ctrl+scroll to zoom.</p>
    </section>
  ` : `<section class="wide-section"><p class="empty">0 orbits yet.</p></section>`;

  // Numbered sections count up so §1 is the research map, then body flows.
  let n = 1;
  const nextSec = () => ++n;

  // Split orbits into Active (everything except `demoted`) and Demoted
  // (status="demoted"), so the leaderboard reflects the current eval's
  // honest standings. Demoted orbits remain visible but collapsed below
  // — readers can audit history without it cluttering the active view.
  const activeOrbits = sorted.filter(o => o.status !== 'demoted');
  const demotedOrbits = sorted.filter(o => o.status === 'demoted');

  const leaderboardSec = orbits.length ? `
    <h2 class="s"><span class="n">§${nextSec()}</span>Leaderboard</h2>
    <p>${activeOrbits.length} active orbit${activeOrbits.length === 1 ? '' : 's'} evaluated under <span class="mono">${esc(c.eval_version || 'eval-v?')}</span>, sorted by metric (${direction === 'min' ? 'lower is better' : 'higher is better'}).${demotedOrbits.length ? ` ${demotedOrbits.length} demoted (collapsed below).` : ''}</p>
    ${renderLeaderboard(activeOrbits, c, direction)}
    ${demotedOrbits.length ? `
      <details class="demoted-history">
        <summary>Demoted history (${demotedOrbits.length} orbit${demotedOrbits.length === 1 ? '' : 's'} no longer beating baseline under current eval)</summary>
        ${renderLeaderboard(demotedOrbits, c, direction)}
        <p class="tbl-caption"><span class="num">Tab. 1b</span>Orbits previously graduated but demoted by <code>/doctor --fix</code> after the eval changed.</p>
      </details>` : ''}
    <p class="tbl-caption"><span class="num">Tab. 1</span>Orbits ranked by metric. Indentation indicates parent-of lineage; rows with no metric are omitted from ranking.</p>
  ` : '';

  const progressSec = orbits.filter(o => typeof o.metric === 'number').length >= 2 ? `
    <h2 class="s"><span class="n">§${nextSec()}</span>Progress</h2>
    <div class="progress-charts">
      <div class="chart-wrap"><canvas id="progress-scatter"></canvas></div>
    </div>
    <p class="tbl-caption"><span class="num">Fig. 2</span>Per-orbit metric vs. completion index. The new-best trajectory (solid line) traces orbits that improved over the running best at their completion time.</p>
  ` : '';

  const milestonesSec = c.timeline?.length ? `
    <h2 class="s"><span class="n">§${nextSec()}</span>Milestone log</h2>
    ${renderTimeline(c.timeline)}
  ` : '';

  const methodologySec = (c.metric_description || c.eval_methodology) ? `
    <h2 class="s"><span class="n">§${nextSec()}</span>Evaluation</h2>
    ${renderMethodology(c)}
  ` : '';

  const referencesSec = c.references?.length ? `
    <h2 class="s"><span class="n">§${nextSec()}</span>References</h2>
    ${renderReferences(c.references)}
  ` : '';

  const bodyBlock = (leaderboardSec || progressSec || milestonesSec || methodologySec || referencesSec) ? `
    <article class="paper narrow">
      ${leaderboardSec}
      ${progressSec}
      ${milestonesSec}
      ${methodologySec}
      ${referencesSec}
      ${renderFooter(counts, orbits.length)}
    </article>
  ` : `<article class="paper narrow">${renderFooter(counts, orbits.length)}</article>`;

  const landscapeSec = c.exploration ? renderLandscape(c.exploration) : '';

  // Canele-pitch notebook pattern: wrap the whole page body in a 2-column
  // grid (main column + right rail). The rail is sticky so it scroll-
  // follows the prose for the whole page. Full-bleed elements (#dag,
  // .concept-graph) escape via 100vw + negative margins as before.
  return `
    <div class="doc-grid">
      <main class="doc-main">
        <article class="paper top">${headerBlock}</article>
        <article class="paper top intro">${introBlock}</article>
        ${researchMap}
        ${bodyBlock}
        ${landscapeSec}
      </main>
      ${railBlock}
    </div>
  `;
}

function renderEvalToggle(c) {
  const versions = (c.eval_versions || []).map(v => v.id);
  if (versions.length <= 1) return esc(c.eval_version || '—');
  const pills = versions.map(v => {
    const cls = v === c.eval_version ? 'eval-pill active' : 'eval-pill';
    return `<button type="button" class="${cls}" data-eval-pill="${esc(v)}">${esc(v)}</button>`;
  }).join('');
  return `<span class="eval-toggle">${pills}</span>`;
}

function renderAbstract(c, best) {
  return `
    <div class="abstract">
      <div class="label">Abstract</div>
      <div class="abstract-body">${renderMarkdown(c.problem || '')}</div>
      ${best?.metric != null ? `<p class="abstract-result"><strong>Result:</strong> best metric <span class="mono">${fmt(best.metric)}</span>${best.orbit ? ` (orbit <a href="javascript:void(0)" class="abstract-orbit-link" data-orbit="${esc(best.orbit)}">${esc(best.orbit)}</a>)` : ''}${best.target != null ? `; target <span class="mono">${fmt(best.target)}</span>` : ''}.</p>` : ''}
    </div>
  `;
}

// ====== Leaderboard =======================================================
function renderLeaderboard(orbits, c, direction) {
  // Compute indent class per row: depth based on distance from root in parent chain.
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  function depth(name, seen = new Set()) {
    if (seen.has(name)) return 0;
    seen.add(name);
    const o = byName[name];
    if (!o || !o.parents || o.parents.length === 0) return 0;
    let d = 0;
    for (const p of o.parents) {
      if (byName[p]) d = Math.max(d, 1 + depth(p, seen));
    }
    return d;
  }
  const target = typeof c?.best?.target === 'number' ? c.best.target : null;
  const deltaLabel = target != null ? 'Δ target' : 'Δ best';

  const rankedByMetric = orbits.filter(o => typeof o.metric === 'number');
  const bestMetric = rankedByMetric.length
    ? (direction === 'min'
        ? Math.min(...rankedByMetric.map(o => o.metric))
        : Math.max(...rankedByMetric.map(o => o.metric)))
    : null;
  const baseline = target != null ? target : bestMetric;

  const rows = orbits.map((o, i) => {
    const d = Math.min(3, depth(o.name));
    const indentCls = d > 0 ? `indent-${d}` : '';
    const statusCls = o.status === 'winner' || o.status === 'graduated' ? 'winner'
      : o.status === 'demoted' ? 'demoted'
      : o.status === 'dead-end' ? 'dead'
      : o.status === 'malformed' ? 'malformed'
      : '';
    let delta = '—';
    if (typeof o.metric === 'number' && baseline != null) {
      const d = o.metric - baseline;
      if (Math.abs(d) < 1e-9) delta = '—';
      else delta = (d > 0 ? '+' : '') + fmt(d);
    }
    const citedByCount = (o.cited_by || []).length;
    const citedBadge = citedByCount
      ? ` <span class="cited-by-badge" title="cited by ${citedByCount} orbit${citedByCount === 1 ? '' : 's'}">↩${citedByCount}</span>`
      : '';
    const nameCell = o.status === 'demoted' && o.demotion_reason
      ? `${esc(o.name)} <span class="demotion-reason" title="${esc(o.demotion_reason)}">⚠</span>${citedBadge}`
      : `${esc(o.name)}${citedBadge}`;
    return `
      <tr class="${statusCls} ${indentCls}" onclick="location.hash='#/orbit/${encodeURIComponent(o.name)}'">
        <td class="rank">${String(i + 1).padStart(2, '0')}</td>
        <td class="name">${nameCell}</td>
        <td class="strategy-col">${esc(o.strategy || '')}</td>
        <td class="stat">${fmt(o.metric)}</td>
        <td class="stat delta">${delta}</td>
      </tr>
    `;
  }).join('');

  return `
    <table class="lbnb">
      <thead>
        <tr>
          <th class="rank">#</th>
          <th>Orbit</th>
          <th>Strategy</th>
          <th class="stat">Metric</th>
          <th class="stat">${esc(deltaLabel)}</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

// ====== Milestones (logbook) =============================================
function renderTimeline(entries) {
  const items = entries.map(e => {
    let body = e.text || '';
    const raw = e.at || '';
    const date = raw.length >= 10 ? raw.slice(0, 10) : raw;
    const time = raw.length >= 16 ? raw.slice(11, 16) : '';
    const isDebate = body.includes('🗳️ Debate') || body.includes('debate-agent');
    const isMilestone = body.includes('## Round') || body.includes('## Milestone') || body.includes('Leaderboard');
    const isOrbitComplete = body.includes('complete') && body.includes('orbit/');
    const marker = isDebate ? 'debate' : isMilestone ? 'milestone' : isOrbitComplete ? 'result' : 'update';
    const author = e.author || '';
    // Strip the redundant "**author:** Milestone N (after X orbits)" prefix
    // and the duplicate "## Milestone N" header on milestone entries — the
    // marker chip + agent chip already convey both. Without this we get a
    // triple-label visual: chip, bold prefix, h2 header all saying "milestone".
    if (marker === 'milestone') {
      body = body.replace(/^\s*\*\*[\w\s.-]+:\*\*\s*Milestone\s+\d+[^\n]*\n+/i, '');
      body = body.replace(/^\s*##\s+Milestone\s+\d+\s*\n+/i, '');
    }
    const imgs = (e.images || []).map(url =>
      `<img src="${esc(url)}" loading="lazy" alt="">`
    ).join('');
    return `
      <li>
        <div class="timestamp">${esc(date)}${time ? `T${esc(time)}` : ''}${author ? `<span class="agent">${esc(author)}</span>` : ''}</div>
        <div class="entry-body"><span class="marker">${esc(marker)}</span>${renderMarkdown(body)}${imgs ? `<div>${imgs}</div>` : ''}</div>
      </li>
    `;
  }).join('');
  return `<ol class="logbook">${items}</ol>`;
}

// ====== Methodology =======================================================
function renderMethodology(c) {
  const ev = c.eval_methodology || {};
  const stages = (ev.stages || []).map(s => `<li>${esc(s)}</li>`).join('');
  return `
    <div class="methodology">
      ${c.metric_description ? `<p class="metric-desc"><strong>Metric:</strong> ${esc(c.metric_description)}</p>` : ''}
      ${ev.summary ? `<p>${esc(ev.summary)}</p>` : ''}
      ${stages ? `<ol class="stages">${stages}</ol>` : ''}
      ${ev.baseline ? `<p class="baseline"><strong>Baseline:</strong> ${esc(ev.baseline)}</p>` : ''}
      ${ev.params ? `<p class="eval-params"><span class="mono">${esc(ev.params)}</span></p>` : ''}
    </div>
  `;
}

// ====== References ========================================================
function renderReferences(refs) {
  const items = refs.map(r => {
    const title = r.title || r.cite || '';
    const finding = r.finding || r.note || '';
    const year = r.year ? ` (${r.year})` : '';
    const authors = r.authors ? `${r.authors}${year}. ` : '';
    const linkTitle = r.url ? `<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(title)}</a>` : esc(title);
    return `
      <li>
        <span class="ref-cite">${authors}${linkTitle}</span>
        ${finding ? `<span class="ref-note">${esc(finding)}</span>` : ''}
      </li>
    `;
  }).join('');
  return `<ul class="references">${items}</ul>`;
}

function renderRailReferences(refs) {
  // All refs go into the rail — the container is scrollable so there's
  // no reason to truncate. A small count badge in the header tells the
  // reader how many there are without needing a "+N more" footer.
  const items = refs.map((r, i) => {
    const title = r.title || r.cite || '';
    const finding = r.finding || r.note || '';
    const year = r.year ? ` · ${r.year}` : '';
    const authors = r.authors || '';
    const href = r.url || '';
    const titleHtml = href
      ? `<a href="${esc(href)}" target="_blank" rel="noopener">${esc(title)}</a>`
      : esc(title);
    return `
      <div class="item">
        <span class="idx">[${i + 1}]</span>
        <span class="body">
          <span class="cite">${titleHtml}</span>
          ${authors ? `<span class="meta">${esc(authors)}${year}</span>` : ''}
          ${finding ? `<span class="note">${esc(finding)}</span>` : ''}
        </span>
      </div>
    `;
  }).join('');
  return `
    <div class="rail-refs">
      <div class="hd">References <span class="count">${refs.length}</span></div>
      <div class="rail-refs-scroll">${items}</div>
    </div>
  `;
}

// ====== Landscape (exploration) ==========================================
function renderLandscape(exploration) {
  const entries = exploration.steering_entries || [];
  const graph = exploration.concept_graph;
  const landscape = exploration.landscape || '';
  const openQs = exploration.open_questions || [];
  const ruledOut = exploration.ruled_out || [];

  const typeColor = {
    prompt: '#6e7ef0', taste: '#60a5a0', direction: '#88bb60',
    'ruling-out': '#cc7760', 'open-question': '#b070cc', steering: '#c0a840',
  };

  const steeringChips = entries.length ? `
    <div class="steering-log">
      ${entries.map(e => {
        const col = typeColor[e.type] || '#888';
        return `<span class="steering-chip" style="border-color:${col};color:${col}" title="${esc(e.type)}">${esc(e.content)}</span>`;
      }).join('')}
    </div>
  ` : '';

  const conceptGraphEl = graph && (graph.nodes || []).length ? `
    <div id="concept-graph" class="concept-graph"></div>
    <p class="tbl-caption">Concept graph — ${(graph.nodes || []).length} nodes, ${(graph.edges || []).length} edges. Node size = intent relevance. Click to expand.</p>
  ` : '';

  const landscapeBody = landscape ? `
    <div class="landscape-body">${renderMarkdown(landscape)}</div>
  ` : '';

  const sidebar = (openQs.length || ruledOut.length) ? `
    <div class="landscape-sidebar">
      ${openQs.length ? `<div class="sidebar-block"><div class="hd">Open questions</div>${openQs.map(q => `<p class="sidebar-item">${esc(q)}</p>`).join('')}</div>` : ''}
      ${ruledOut.length ? `<div class="sidebar-block"><div class="hd">Ruled out</div>${ruledOut.map(r => `<p class="sidebar-item ruled-out">${esc(r)}</p>`).join('')}</div>` : ''}
    </div>
  ` : '';

  return `
    <section class="wide-section landscape-section">
      <h2 class="s">Exploration landscape</h2>
      ${steeringChips}
      ${conceptGraphEl}
      <div class="landscape-grid">
        ${landscapeBody}
        ${sidebar}
      </div>
    </section>
  `;
}

function mountConceptGraph(data) {
  const el = document.getElementById('concept-graph');
  if (!el) return;
  const graph = data.campaign?.exploration?.concept_graph;
  if (!graph || !(graph.nodes || []).length) return;

  const cs = getComputedStyle(document.body);
  const fg = cs.getPropertyValue('--foreground').trim();
  const bg = cs.getPropertyValue('--background').trim() || '#fefdfb';
  const border = cs.getPropertyValue('--border').trim();
  const muted = cs.getPropertyValue('--muted').trim() || '#888';
  const accent = cs.getPropertyValue('--accent').trim()
    || cs.getPropertyValue('--destructive').trim() || '#7a3a1f';

  const clusterColors = {};
  for (const cl of (graph.clusters || [])) clusterColors[cl.id] = cl.color;

  const typeShapes = {
    paper: 'ellipse', method: 'roundrectangle', concept: 'diamond',
    question: 'pentagon', gap: 'star', hypothesis: 'hexagon',
  };

  const nodes = (graph.nodes || []).map(n => ({
    data: {
      id: n.id,
      label: n.label || n.id,
      type: n.type || 'concept',
      cluster: n.cluster || '',
      summary: n.summary || '',
      url: n.url || '',
      year: n.year || '',
      authors: n.authors || '',
      relevance: typeof n.relevance === 'number' ? n.relevance : 0.5,
      intentRelevance: typeof n.intent_relevance === 'number' ? n.intent_relevance : 0.5,
      size: 16 + (typeof n.intent_relevance === 'number' ? n.intent_relevance * 20 : 10),
      color: clusterColors[n.cluster] || fg,
    },
  }));

  const edges = (graph.edges || []).map((e, i) => ({
    data: { id: `e${i}`, source: e.source, target: e.target, relation: e.relation || '', label: e.label || '' },
  }));

  // Preset cluster-circle layout. Cose and its cousins kept collapsing
  // clustered groups into a horizontal line for this data shape. The
  // clusters are already declared in data.json, so we just honor them:
  //
  //   1. Group nodes by cluster.
  //   2. Place cluster centers on a large outer circle.
  //   3. Within each cluster, place nodes on a smaller inner circle
  //      around their cluster center. Single-node clusters sit at
  //      the center itself.
  //
  // Result: every cluster gets its own visual region, inter-cluster
  // edges show up as diameter-crossing lines, and no two labels sit
  // within the text-max-width of each other.
  const rect = el.getBoundingClientRect();
  const W = rect.width || 1600;
  const H = rect.height || 860;
  const centerX = W / 2, centerY = H / 2;

  const byCluster = {};
  for (const n of nodes) {
    const cid = n.data.cluster || '__none__';
    (byCluster[cid] = byCluster[cid] || []).push(n);
  }
  // Sort biggest-first for stable ordering, then alternate big/small
  // around the wheel so two huge clusters aren't adjacent.
  const sortedBySize = Object.keys(byCluster).sort(
    (a, b) => byCluster[b].length - byCluster[a].length
  );
  const clusterIds = [];
  for (let i = 0; i < sortedBySize.length; i += 2) clusterIds.push(sortedBySize[i]);
  for (let i = 1; i < sortedBySize.length; i += 2) clusterIds.push(sortedBySize[i]);

  // Inner-cluster radius: nodes arranged on a circle with chord spacing
  // >= minChord. minChord picks a label-width worth of horizontal room.
  const minChord = 110;
  function innerR(n) {
    if (n <= 1) return 0;
    // radius s.t. chord between neighbors equals minChord:
    //   chord = 2r sin(pi/n)  =>  r = chord / (2 sin(pi/n))
    return Math.max(42, minChord / (2 * Math.sin(Math.PI / n)));
  }

  // Outer radius sized so adjacent cluster disks (biggest neighbors)
  // don't touch. Chord between cluster centers = 2 * R * sin(pi / k).
  // Require chord >= maxInnerR * 2 + gap.
  const radii = clusterIds.map(cid => innerR(byCluster[cid].length));
  const k = Math.max(clusterIds.length, 1);
  const maxNeighborRadii = radii.length > 1
    ? Math.max(...radii.map((r, i) => r + radii[(i + 1) % radii.length]))
    : 0;
  const gap = 70;
  const outerMin = (maxNeighborRadii + gap) / (2 * Math.sin(Math.PI / Math.max(k, 2)));
  // Also bound by container — allow it to stretch beyond if the
  // container is smaller than the computed minimum, so content
  // doesn't clip.
  const outerRadius = Math.max(outerMin, Math.min(W, H) * 0.32);

  const presetPositions = {};
  clusterIds.forEach((cid, ci) => {
    const members = byCluster[cid];
    const angle = (ci / k) * 2 * Math.PI - Math.PI / 2;
    const ccx = centerX + outerRadius * Math.cos(angle);
    const ccy = centerY + outerRadius * Math.sin(angle);
    const r = innerR(members.length);
    members.forEach((n, ni) => {
      const theta = members.length === 1 ? 0 : (ni / members.length) * 2 * Math.PI;
      presetPositions[n.data.id] = {
        x: ccx + r * Math.cos(theta),
        y: ccy + r * Math.sin(theta),
      };
    });
  });

  const cy = cytoscape({
    container: el,
    elements: [...nodes, ...edges],
    pixelRatio: 'auto',
    layout: {
      name: 'preset',
      positions: (node) => presetPositions[node.id()] || { x: centerX, y: centerY },
      fit: true,
      padding: 40,
    },
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'data(color)',
          'background-opacity': 0.28,
          'border-width': 1,
          'border-color': 'data(color)',
          'border-opacity': 0.55,
          'label': 'data(label)',
          'font-family': "'IBM Plex Mono', ui-monospace, monospace",
          'font-size': 9,
          'color': fg,
          'text-valign': 'bottom',
          'text-halign': 'center',
          'text-margin-y': 8,
          // Narrower labels wrap onto a second line instead of colliding
          // horizontally with neighbours.
          'text-max-width': '78px',
          'text-wrap': 'wrap',
          'text-opacity': 0.88,
          // Transparent label — matches the research-map (DAG) style so
          // the two graphs read as siblings. Cluster-circle layout
          // already guarantees chord spacing >= a label width.
          'text-background-opacity': 0,
          'width': 'data(size)', 'height': 'data(size)',
          'shape': 'ellipse',
          'min-zoomed-font-size': 4,
        },
      },
      {
        selector: 'node[intentRelevance > 0.7]',
        style: {
          'background-opacity': 0.48,
          'border-opacity': 0.85,
          'font-size': 10.5,
          'font-weight': 600,
          'text-opacity': 1,
        },
      },
      {
        selector: 'edge',
        style: {
          /* Lines dark enough to read against the dotted grid without a
             label background. Curves (unbundled-bezier) offset slightly
             on shared endpoints so overlapping edges in dense clusters
             are still distinguishable. */
          'width': 1.25,
          'line-color': muted,
          'line-opacity': 0.6,
          'opacity': 1,
          'curve-style': 'straight',
          'target-arrow-shape': 'none',
          /* Hide labels by default — on a 50+ edge concept graph the
             overlay of relation strings just competes with the lines.
             They reappear on edge.cg-active when a node is selected. */
          'label': '',
          'font-family': "'IBM Plex Mono', ui-monospace, monospace",
          'font-size': 9,
          'color': fg,
        },
      },
      /* Selection / neighborhood highlight states ------------------- */
      { selector: '.cg-dim', style: { 'opacity': 0.08, 'text-opacity': 0 } },
      { selector: 'node.cg-hover', style: {
          'overlay-color': 'data(color)',
          'overlay-opacity': 0.22,
          'overlay-padding': 10,
          'border-opacity': 1,
          'text-opacity': 1,
          'z-index': 80,
      } },
      { selector: '.cg-selected', style: {
          'border-width': 3,
          'border-opacity': 1,
          'background-opacity': 0.75,
          'font-size': 11,
          'font-weight': 700,
          'text-opacity': 1,
          'z-index': 100,
      } },
      { selector: '.cg-neighbor', style: {
          'border-width': 2,
          'border-opacity': 0.9,
          'background-opacity': 0.55,
          'font-weight': 600,
          'text-opacity': 1,
          'z-index': 50,
      } },
      { selector: 'edge.cg-active', style: {
          'line-color': accent,
          'line-opacity': 0.9,
          'opacity': 1,
          'width': 2.25,
          'label': 'data(label)',
          'font-size': 10,
          'font-weight': 600,
          'color': accent,
          'text-opacity': 1,
          'text-rotation': 'autorotate',
          'text-background-color': bg,
          'text-background-opacity': 0.92,
          'text-background-padding': 3,
          'text-background-shape': 'roundrectangle',
          'z-index': 40,
      } },
    ],
  });

  cy.userZoomingEnabled(false);
  el.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      cy.zoom({ level: cy.zoom() * factor, renderedPosition: { x: e.offsetX, y: e.offsetY } });
    }
  }, { passive: false });

  const tip = document.createElement('div');
  tip.className = 'dag-tooltip';
  el.appendChild(tip);

  cy.on('mouseover', 'node', (evt) => {
    const n = evt.target;
    const d = n.data();
    const pos = n.renderedPosition();
    tip.innerHTML = `<strong>${esc(d.label)}</strong>
      ${d.authors ? `<span class="tip-strategy">${esc(d.authors)}${d.year ? ` (${d.year})` : ''}</span>` : ''}
      ${d.summary ? `<span class="tip-metric">${esc(d.summary)}</span>` : ''}
      ${d.url ? `<a href="${esc(d.url)}" target="_blank" rel="noopener" style="font-size:9px">↗ link</a>` : ''}`;
    tip.style.left = pos.x + 'px';
    tip.style.top = (pos.y - 12) + 'px';
    tip.classList.add('visible');
    n.addClass('cg-hover');
  });
  cy.on('mouseout', 'node', (evt) => {
    tip.classList.remove('visible');
    evt.target.removeClass('cg-hover');
  });

  // --- Selection / neighborhood detail panel ------------------------
  // Container must position children; set on the graph wrapper.
  el.style.position = el.style.position || 'relative';
  const panel = document.createElement('div');
  panel.className = 'cg-detail';
  panel.innerHTML = '';
  el.appendChild(panel);

  function clearSelection() {
    cy.elements().removeClass('cg-dim cg-selected cg-neighbor cg-active');
    panel.classList.remove('open');
    panel.innerHTML = '';
  }

  function selectNode(node) {
    const neighborhood = node.closedNeighborhood();
    cy.elements().addClass('cg-dim');
    neighborhood.removeClass('cg-dim');
    node.removeClass('cg-neighbor').addClass('cg-selected');
    neighborhood.edges().addClass('cg-active');
    neighborhood.nodes().not(node).addClass('cg-neighbor');

    const d = node.data();
    const neighbors = neighborhood.nodes().not(node).map(n => {
      const nd = n.data();
      // Find the edge that connects this neighbor to the selected node
      // so we can show the relation label.
      const edge = cy.edges().filter(e =>
        (e.source().id() === node.id() && e.target().id() === nd.id) ||
        (e.target().id() === node.id() && e.source().id() === nd.id)
      )[0];
      const rel = edge ? (edge.data('relation') || edge.data('label') || '') : '';
      return { ...nd, relation: rel };
    });

    panel.innerHTML = renderConceptDetailPanel(d, neighbors);
    panel.classList.add('open');

    // Zoom/fit to the selected neighborhood.
    cy.animate({
      fit: { eles: neighborhood, padding: 80 },
      duration: 280,
      easing: 'ease-out',
    });
  }

  cy.on('tap', 'node', (evt) => selectNode(evt.target));
  cy.on('tap', (evt) => {
    if (evt.target === cy) {
      clearSelection();
      cy.animate({ fit: { padding: 40 }, duration: 240, easing: 'ease-out' });
    }
  });
  // Close-button + follow-link are delegated to the panel itself:
  panel.addEventListener('click', (e) => {
    const t = e.target;
    if (t && t.classList && t.classList.contains('cg-close')) {
      clearSelection();
      cy.animate({ fit: { padding: 40 }, duration: 240, easing: 'ease-out' });
    }
    const nid = t && t.getAttribute && t.getAttribute('data-goto');
    if (nid) {
      const n = cy.getElementById(nid);
      if (n && n.length) selectNode(n);
    }
  });
}

function renderConceptDetailPanel(d, neighbors) {
  const url = d.url
    ? `<a class="cg-link" href="${esc(d.url)}" target="_blank" rel="noopener">↗ open source</a>`
    : '';
  const header = `
    <div class="cg-head">
      <button class="cg-close" aria-label="close">×</button>
      <div class="cg-cluster">${esc(d.cluster || 'concept')}</div>
      <div class="cg-title">${esc(d.label || d.id || '')}</div>
      ${d.authors ? `<div class="cg-meta">${esc(d.authors)}${d.year ? ` · ${d.year}` : ''}</div>` : ''}
      ${d.summary ? `<div class="cg-summary">${esc(d.summary)}</div>` : ''}
      ${url}
    </div>
  `;
  const neighborsHtml = neighbors.length ? `
    <div class="cg-section">
      <div class="cg-section-hd">Connected · ${neighbors.length}</div>
      <ul class="cg-neighbors">
        ${neighbors.map(n => `
          <li>
            <button class="cg-neighbor-btn" data-goto="${esc(n.id)}">
              <span class="cg-dot" style="background:${esc(n.color || '#888')}"></span>
              <span class="cg-neighbor-body">
                <span class="cg-neighbor-label">${esc(n.label || n.id)}</span>
                ${n.relation ? `<span class="cg-neighbor-rel">${esc(n.relation)}</span>` : ''}
              </span>
            </button>
          </li>
        `).join('')}
      </ul>
    </div>
  ` : '<div class="cg-section"><div class="cg-empty">No connections.</div></div>';
  return header + neighborsHtml;
}

// ====== Markdown ==========================================================
function renderMarkdown(text) {
  let clean = (text || '').replace(/!\[[^\]]*\]\([^)]+\)/g, '');
  try {
    // breaks:false is required for KaTeX: if soft newlines became <br>,
    // any $…$ math pair straddling a source-line wrap would be split
    // across two text nodes and auto-render couldn't pair the delimiters.
    // Standard CommonMark behavior — paragraphs separate on blank lines.
    return marked.parse(clean, { breaks: false, gfm: true });
  } catch (e) {
    return esc(clean);
  }
}

// ====== DAG (Cytoscape) ==================================================
function mountDag(data) {
  const el = document.getElementById('dag');
  if (!el) return;
  const orbits = data.orbits || [];

  const metrics = orbits.map(o => o.metric).filter(m => m != null && typeof m === 'number');
  const metricMin = Math.min(...metrics), metricMax = Math.max(...metrics);
  const dir = data.campaign?.best?.direction === 'max' ? 'max' : 'min';

  function nodeSize(o) {
    if (o.metric == null) return 20;
    if (metricMin === metricMax) return 24;
    const norm = dir === 'min'
      ? 1 - (o.metric - metricMin) / (metricMax - metricMin)
      : (o.metric - metricMin) / (metricMax - metricMin);
    return 18 + norm * 20;
  }
  function nodeQuality(o) {
    if (o.metric == null) return 0;
    if (metricMin === metricMax) return 0.5;
    return dir === 'min'
      ? 1 - (o.metric - metricMin) / (metricMax - metricMin)
      : (o.metric - metricMin) / (metricMax - metricMin);
  }
  function truncLabel(s, n) { return s && s.length > n ? s.slice(0, n) + '…' : s || ''; }

  const nodes = orbits.map(o => ({
    data: {
      id: o.name,
      label: truncLabel(o.strategy || o.name, 24),
      shortName: o.name,
      status: o.status || 'exploring',
      metric: o.metric,
      strategy: o.strategy || '',
      size: nodeSize(o),
      quality: nodeQuality(o),
    },
  }));
  const edges = [];
  for (const o of orbits) {
    for (const p of (o.parents || [])) {
      if (orbits.find(x => x.name === p)) edges.push({ data: { source: p, target: o.name } });
    }
  }
  const cs = getComputedStyle(document.body);
  const fg = cs.getPropertyValue('--foreground').trim();
  const bg = cs.getPropertyValue('--background').trim();
  const border = cs.getPropertyValue('--border').trim();
  const destructive = cs.getPropertyValue('--destructive').trim();

  const cy = cytoscape({
    container: el,
    elements: [...nodes, ...edges],
    pixelRatio: 'auto',
    layout: { name: 'dagre', rankDir: 'TB', nodeSep: 80, rankSep: 100, edgeSep: 25, ranker: 'tight-tree' },
    style: [
      {
        selector: 'node',
        style: {
          'background-color': 'mapData(quality, 0, 1, ' + bg + ', ' + fg + ')',
          'background-opacity': 0.18,
          'border-width': 0.75,
          'border-color': fg,
          'border-opacity': 0.3,
          'label': 'data(label)',
          'font-family': "'IBM Plex Mono', ui-monospace, monospace",
          'font-size': 10,
          'font-weight': 400,
          'color': fg,
          'text-valign': 'bottom',
          'text-halign': 'center',
          'text-margin-y': 8,
          'text-max-width': '120px',
          'text-wrap': 'ellipsis',
          'text-opacity': 0.8,
          'width': 'data(size)', 'height': 'data(size)',
          'shape': 'ellipse',
          'transition-property': 'border-width, width, height, opacity, background-opacity, border-opacity',
          'transition-duration': 150,
        },
      },
      { selector: 'node[quality > 0.3]',  style: { 'background-opacity': 0.28, 'border-opacity': 0.45 } },
      { selector: 'node[quality > 0.6]',  style: { 'background-opacity': 0.38, 'border-opacity': 0.55, 'font-weight': 600 } },
      { selector: 'node[quality > 0.85]', style: { 'background-opacity': 0.48, 'border-opacity': 0.7, 'border-width': 1.25 } },
      { selector: 'node[status = "dead-end"]', style: {
          'opacity': 0.35, 'background-opacity': 0.08, 'border-style': 'dashed',
          'border-opacity': 0.2, 'font-size': 9, 'text-opacity': 0.5,
        }
      },
      { selector: 'node[status = "malformed"]', style: {
          'opacity': 0.2, 'background-opacity': 0.05, 'border-style': 'dotted', 'font-size': 9,
        }
      },
      { selector: 'node[status = "promising"]', style: { 'border-width': 1.25, 'border-opacity': 0.6 } },
      { selector: 'node[status = "exploring"]', style: { 'border-style': 'dotted', 'border-width': 0.75 } },
      { selector: 'node[status = "winner"], node[status = "graduated"]', style: {
          'background-color': destructive, 'background-opacity': 0.65,
          'border-color': destructive, 'border-width': 2, 'border-opacity': 1,
          'color': fg, 'font-weight': 600, 'font-size': 11, 'text-opacity': 1,
        }
      },
      { selector: 'node:active', style: { 'overlay-opacity': 0.04, 'overlay-color': fg } },
      {
        selector: 'edge',
        style: {
          'width': 1, 'line-color': border, 'curve-style': 'bezier',
          'control-point-step-size': 55, 'target-arrow-shape': 'none', 'opacity': 0.5,
        },
      },
      ...multiParentEdgeStyles(orbits, fg),
      ...edgeConnectednessStyles(orbits),
      { selector: '.dimmed', style: { 'opacity': 0.12, 'transition-duration': 200 } },
      { selector: '.highlighted-edge', style: {
          'line-color': destructive, 'opacity': 0.8, 'width': 2.5,
          'line-style': 'dashed', 'line-dash-pattern': [8, 4], 'transition-duration': 200,
        }
      },
      { selector: '.ancestor-node', style: {
          'border-color': destructive, 'border-opacity': 0.6, 'border-width': 2, 'transition-duration': 200,
        }
      },
    ],
  });

  cy.userZoomingEnabled(false);
  el.addEventListener('wheel', (e) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const factor = e.deltaY < 0 ? 1.1 : 0.9;
      cy.zoom({ level: cy.zoom() * factor, renderedPosition: { x: e.offsetX, y: e.offsetY } });
    }
  }, { passive: false });

  const _hint = document.createElement('div');
  _hint.className = 'dag-zoom-hint';
  _hint.textContent = 'Ctrl + scroll to zoom · drag to pan';
  el.appendChild(_hint);

  const ancestry = winnerAncestryPath(orbits);
  if (ancestry.size > 1) {
    cy.edges().forEach(e => {
      const src = e.data('source'), tgt = e.data('target');
      if (ancestry.has(src) && ancestry.has(tgt)) {
        e.style({
          'line-color': destructive, 'opacity': 0.7,
          'line-style': 'dashed', 'line-dash-pattern': [8, 4],
        });
      }
    });
    let offset = 0;
    function animateFlow() {
      offset = (offset + 0.5) % 12;
      cy.edges().forEach(e => {
        const src = e.data('source'), tgt = e.data('target');
        if (ancestry.has(src) && ancestry.has(tgt)) e.style('line-dash-offset', -offset);
      });
      requestAnimationFrame(animateFlow);
    }
    requestAnimationFrame(animateFlow);

    cy.nodes().forEach(n => {
      if (ancestry.has(n.id()) && n.data('status') !== 'winner') {
        n.style({
          'border-color': destructive, 'border-width': 1.75, 'border-opacity': 0.45,
        });
      }
    });
  }

  const tip = document.createElement('div');
  tip.className = 'dag-tooltip';
  el.appendChild(tip);

  cy.on('mouseover', 'node', (evt) => {
    const n = evt.target;
    const d = n.data();
    const pos = n.renderedPosition();
    const metricStr = d.metric != null ? fmt(d.metric) : '—';
    const parents = (data.orbits.find(o => o.name === d.id) || {}).parents || [];
    tip.innerHTML = `<strong>${esc(d.id)}</strong>
      <span class="tip-strategy">${esc(d.strategy)}</span>
      <span class="tip-metric">${d.status} · ${metricStr}</span>
      ${parents.length ? '<span class="tip-parents">← ' + parents.map(esc).join(', ') + '</span>' : ''}`;
    tip.style.left = pos.x + 'px';
    tip.style.top = (pos.y - 12) + 'px';
    tip.classList.add('visible');
    const glowColor = n.data('status') === 'winner' || n.data('status') === 'graduated' ? destructive : fg;
    n.style({
      'border-width': 2, 'border-opacity': 1,
      'background-opacity': Math.min(0.7, (parseFloat(n.style('background-opacity')) || 0.18) + 0.25),
      'overlay-color': glowColor,
      'overlay-opacity': 0.18,
      'overlay-padding': 8,
      'z-index': 999,
      'width': n.data('size') * 1.2, 'height': n.data('size') * 1.2,
    });
  });

  cy.on('mouseout', 'node', (evt) => {
    tip.classList.remove('visible');
    evt.target.removeStyle('overlay-color');
    evt.target.removeStyle('overlay-opacity');
    evt.target.removeStyle('overlay-padding');
    evt.target.removeStyle('border-width');
    evt.target.removeStyle('border-opacity');
    evt.target.removeStyle('background-opacity');
    evt.target.removeStyle('z-index');
    evt.target.removeStyle('width');
    evt.target.removeStyle('height');
  });

  el.style.opacity = '0';
  cy.ready(() => { el.style.transition = 'opacity 350ms ease'; el.style.opacity = '1'; });

  let selectedNode = null;
  function selectOrbit(name) {
    if (!name) { deselectNode(); return; }
    if (selectedNode === name) { deselectNode(); return; }
    selectedNode = name;
    const orbit = (data.orbits || []).find(o => o.name === name);

    cy.elements().removeClass('dimmed highlighted-edge ancestor-node');
    const ancestors = traceAncestry(name, data.orbits);
    cy.nodes().forEach(n => {
      if (!ancestors.has(n.id())) n.addClass('dimmed');
      else n.addClass('ancestor-node');
    });
    cy.edges().forEach(e => {
      if (ancestors.has(e.data('source')) && ancestors.has(e.data('target'))) e.addClass('highlighted-edge');
      else e.addClass('dimmed');
    });

    showDetailPanel(data, orbit, el, selectOrbit);
  }
  function deselectNode() {
    selectedNode = null;
    cy.elements().removeClass('dimmed highlighted-edge ancestor-node');
    hideDetailPanel();
  }

  cy.on('tap', 'node', (evt) => { tip.classList.remove('visible'); selectOrbit(evt.target.id()); });
  cy.on('tap', function(evt) { if (evt.target === cy) deselectNode(); });

  // Abstract result link + leaderboard rows can trigger selectOrbit too
  document.querySelectorAll('.abstract-orbit-link').forEach(a => {
    a.addEventListener('click', () => selectOrbit(a.dataset.orbit));
  });
}

// ====== Progress chart ====================================================
function mountProgress(data) {
  const hostEls = [
    { el: document.getElementById('progress-scatter'),      mini: false },
    { el: document.getElementById('progress-scatter-mini'), mini: true },
  ].filter(h => h.el && typeof Chart !== 'undefined');
  if (!hostEls.length) return;

  // Exclude inf so it doesn't blow up the y-scale (infinity = crashed).
  const orbits = (data.orbits || []).filter(o =>
    typeof o.metric === 'number' && !Number.isNaN(o.metric) && Number.isFinite(o.metric)
  );
  if (orbits.length < 2) return;

  const direction = (data.campaign?.best?.direction || 'minimize').toLowerCase();
  const isMin = !direction.startsWith('max');

  const completed = [...orbits].sort((a, b) => {
    const ta = a.last_commit_at ? Date.parse(a.last_commit_at) : 0;
    const tb = b.last_commit_at ? Date.parse(b.last_commit_at) : 0;
    return ta - tb;
  });

  const scatterSeries = completed.map((o, i) => ({ x: i + 1, y: o.metric, name: o.name, status: o.status }));

  // Pareto/running-best front as a step function: one point per orbit
  // carrying the running best up to (and including) that orbit. This
  // makes the trajectory extend edge-to-edge of the x-axis, rendered
  // as a classic step chart via stepped:'before' on the dataset.
  const recordSeries = [];
  let runningBest = null;
  completed.forEach((o, i) => {
    const improves = runningBest === null
      || (isMin ? o.metric < runningBest : o.metric > runningBest);
    if (improves) runningBest = o.metric;
    recordSeries.push({
      x: i + 1,
      y: runningBest,
      name: improves ? `${o.name} (new best)` : `running best after ${o.name}`,
      isNewBest: improves,
    });
  });

  const target = typeof data.campaign?.best?.target === 'number' ? data.campaign.best.target : null;
  const baselineValue = parseBaseline(data.campaign?.eval_methodology?.baseline);

  const cs = getComputedStyle(document.documentElement);
  const fg = cs.getPropertyValue('--foreground').trim() || '#222';
  const muted = cs.getPropertyValue('--muted').trim() || '#888';
  const accent = cs.getPropertyValue('--destructive').trim() || '#c44';

  const refAnnotations = [];
  if (target !== null) refAnnotations.push({ label: `target (${target})`, value: target, color: accent, dash: [6, 4] });
  if (baselineValue !== null) refAnnotations.push({ label: `baseline (${baselineValue})`, value: baselineValue, color: muted, dash: [2, 3] });

  const refLinePlugin = {
    id: 'refLines',
    afterDatasetsDraw(chart) {
      const { ctx, chartArea: { left, right }, scales: { y } } = chart;
      refAnnotations.forEach(ref => {
        if (ref.value < y.min || ref.value > y.max) return;
        const py = y.getPixelForValue(ref.value);
        ctx.save();
        ctx.strokeStyle = ref.color;
        ctx.setLineDash(ref.dash);
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(left, py); ctx.lineTo(right, py); ctx.stroke();
        ctx.fillStyle = ref.color;
        ctx.font = '11px "IBM Plex Mono", ui-monospace, monospace';
        ctx.textAlign = 'right';
        ctx.fillText(ref.label, right - 6, py - 4);
        ctx.restore();
      });
    }
  };

  const baseOpts = (title, { mini = false } = {}) => ({
    plugins: {
      legend: { display: false },
      title: mini
        ? { display: false }
        : { display: true, text: title, color: fg, font: { size: 13, weight: '600', family: 'Source Serif 4, Georgia, serif' } },
      tooltip: {
        callbacks: { label: (ctx) => `${ctx.raw.name}: ${fmt(ctx.raw.y)}` }
      }
    },
    scales: {
      x: {
        title: mini ? { display: false } : { display: true, text: 'orbit completion #', color: muted, font: { family: 'IBM Plex Mono, monospace' } },
        ticks: {
          color: muted,
          font: { family: 'IBM Plex Mono, monospace', size: mini ? 9 : 11 },
          maxTicksLimit: mini ? 4 : undefined,
          precision: 0,
        },
        grid: { display: !mini },
      },
      y: {
        title: mini ? { display: false } : { display: true, text: `metric (${isMin ? '↓ lower is better' : '↑ higher is better'})`, color: muted, font: { family: 'IBM Plex Mono, monospace' } },
        ticks: {
          color: muted,
          font: { family: 'IBM Plex Mono, monospace', size: mini ? 9 : 11 },
          maxTicksLimit: mini ? 4 : undefined,
        },
        grid: { display: !mini },
      },
    },
    responsive: true,
    maintainAspectRatio: false,
    layout: mini ? { padding: 2 } : undefined,
  });

  for (const { el, mini } of hostEls) {
    // Avoid stacking charts when render() re-runs (e.g., on hashchange).
    const prev = Chart.getChart?.(el);
    if (prev) prev.destroy();
    new Chart(el, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'per-orbit metric',
            data: scatterSeries,
            backgroundColor: muted,
            borderColor: muted,
            pointRadius: mini ? 2.5 : 4,
            showLine: false,
            order: 2,
          },
          {
            label: 'new-best trajectory',
            data: recordSeries,
            type: 'line',
            borderColor: fg,
            backgroundColor: fg,
            // Step function so the Pareto front extends edge-to-edge:
            // y carries the running best horizontally until a new best
            // improves it, then steps down (or up, for maximize).
            stepped: 'before',
            pointRadius: (ctx) => ctx.raw?.isNewBest ? (mini ? 3 : 5) : 0,
            pointBackgroundColor: fg,
            borderWidth: mini ? 1.5 : 2,
            showLine: true,
            fill: false,
            order: 1,
          },
        ],
      },
      options: baseOpts('Per-orbit metric + new-best trajectory', { mini }),
      plugins: [refLinePlugin],
    });
  }
}

function parseBaseline(s) {
  if (!s || typeof s !== 'string') return null;
  const matches = [...s.matchAll(/-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g)];
  if (matches.length === 0) return null;
  return parseFloat(matches[matches.length - 1][0]);
}

// ====== Detail panel (orbit slide-in) =====================================
function showDetailPanel(data, orbit, dagEl, onSelectOrbit) {
  let panel = document.body.querySelector('.detail-panel');
  if (!panel) {
    panel = document.createElement('div');
    panel.className = 'detail-panel';
    document.body.appendChild(panel);
    requestAnimationFrame(() => requestAnimationFrame(() => panel.classList.add('open')));
  } else {
    panel.classList.add('open');
    panel.classList.remove('closing');
  }
  if (!orbit) return;

  const orbits = data.orbits || [];
  const campaign = data.campaign || {};
  const repo = campaign.repo || '';

  // Authors = unique commenters on this orbit
  const agentSet = new Set();
  for (const cm of (orbit.issue_comments || [])) if (cm.author) agentSet.add(cm.author);
  const authorsLine = [...agentSet].map(a => `<span class="agent">${esc(a)}</span>`).join(', ')
    || `<span class="agent">orbit-agent</span>`;

  const issueUrlStr = orbit.issue && repo ? `https://github.com/${repo}/issues/${orbit.issue}` : '';
  const branchUrl = repo ? `https://github.com/${esc(repo)}/tree/orbit/${esc(orbit.name)}` : '';
  const logUrl = orbit.log_url || (repo ? `https://github.com/${esc(repo)}/blob/orbit/${esc(orbit.name)}/orbits/${esc(orbit.name)}/log.md` : '');

  const statusTag = orbit.status === 'winner' || orbit.status === 'graduated'
    ? `★ ${esc(orbit.status)}` : esc(orbit.status || '—');

  const parents = orbit.parents || [];
  const parentsLine = parents.length
    ? parents.map(p => `<a href="javascript:void(0)" class="parent-link" data-orbit="${esc(p)}">${esc(p)}</a>`).join(', ')
    : 'none (root)';

  const figs = (orbit.figures || []).map(f =>
    `<img src="${esc(f.url || f.path)}" alt="${esc(f.caption || '')}" title="${esc(f.caption || '')}">`
  ).join('');
  const extLinks = (orbit.links || []).map(l =>
    `<a href="${esc(l.url)}" target="_blank" rel="noopener">${esc(l.title || l.url)}</a>`
  ).join('');

  let sec = 0;
  const nextSec = () => ++sec;

  panel.innerHTML = `
    <button class="detail-close" title="Close">&times;</button>
    <header class="paper-head">
      <div class="id-line">
        <span>orbit · issue ${orbit.issue ? '#' + orbit.issue : '—'} · branch orbit/${esc(orbit.name)}</span>
        <span class="status-tag">${statusTag}</span>
      </div>
      <h1 class="orbit-name">${esc(orbit.name)}</h1>
      <p class="authors">${authorsLine}</p>
      <p class="affiliation">parents: ${parentsLine} · ${esc(campaign.eval_version || 'eval-v?')} · ${(orbit.issue_comments || []).length} comment${(orbit.issue_comments || []).length === 1 ? '' : 's'}${(orbit.figures || []).length ? ` · ${orbit.figures.length} figure${orbit.figures.length === 1 ? '' : 's'}` : ''}</p>
      ${orbit.strategy ? `<div class="abstract">
        <div class="label">Strategy</div>
        <div class="abstract-body"><p>${esc(orbit.strategy)}</p></div>
      </div>` : ''}
    </header>

    <h2 class="s"><span class="n">§${nextSec()}</span>Metric</h2>
    ${renderMetricTable(orbit, campaign, orbits)}

    <h2 class="s"><span class="n">§${nextSec()}</span>Lineage</h2>
    ${renderLineageTree(orbit, orbits)}

    ${(orbit.borrows && orbit.borrows.length) || (orbit.cited_by && orbit.cited_by.length) ? `
      <h2 class="s"><span class="n">§${nextSec()}</span>Cross-orbit references</h2>
      ${renderCrossRefs(orbit, orbits)}
    ` : ''}

    <h2 class="s"><span class="n">§${nextSec()}</span>Links</h2>
    <div class="detail-links">
      ${issueUrlStr ? `<a href="${esc(issueUrlStr)}" target="_blank" rel="noopener">GitHub issue #${orbit.issue}</a>` : ''}
      ${branchUrl ? `<a href="${esc(branchUrl)}" target="_blank" rel="noopener">branch orbit/${esc(orbit.name)}</a>` : ''}
      ${repo ? `<a href="https://github.com/${esc(repo)}/tree/orbit/${esc(orbit.name)}/orbits/${esc(orbit.name)}" target="_blank" rel="noopener">orbits/${esc(orbit.name)}/</a>` : ''}
      ${logUrl ? `<a href="${esc(logUrl)}" target="_blank" rel="noopener">log.md</a>` : ''}
      ${extLinks}
    </div>

    ${figs ? `<h2 class="s"><span class="n">§${nextSec()}</span>Figures</h2>
      <div class="detail-figs">${figs}</div>` : ''}

    ${(orbit.issue_comments || []).length ? `<h2 class="s"><span class="n">§${nextSec()}</span>Activity</h2>
      ${renderIssueComments(orbit.issue_comments)}` : ''}
  `;

  panel.scrollTop = 0;

  panel.querySelector('.detail-close').addEventListener('click', () => {
    hideDetailPanel();
    if (onSelectOrbit) onSelectOrbit(null);
  });
  panel.querySelectorAll('.parent-link').forEach(a => {
    a.addEventListener('click', () => {
      const name = a.dataset.orbit;
      if (onSelectOrbit) onSelectOrbit(name);
    });
  });
}

function hideDetailPanel() {
  const panel = document.body.querySelector('.detail-panel');
  if (!panel) return;
  panel.classList.remove('open');
  panel.classList.add('closing');
  setTimeout(() => { if (panel.parentNode) panel.remove(); }, 300);
}

// ====== Metric table (inside panel) ======================================
function renderMetricTable(orbit, campaign, orbits) {
  const rows = [];
  const dir = campaign?.best?.direction === 'max' ? 'max' : 'min';
  const isMin = dir === 'min';
  const fmtDelta = (d) => {
    if (d == null || Number.isNaN(d)) return '—';
    if (Math.abs(d) < 1e-9) return '—';
    return (d > 0 ? '+' : '') + fmt(d);
  };
  const classForDelta = (d) => {
    if (d == null || Number.isNaN(d)) return '';
    if (Math.abs(d) < 1e-9) return '';
    const improving = isMin ? d < 0 : d > 0;
    return improving ? 'good' : '';
  };

  if (typeof orbit.metric === 'number') {
    rows.push(`<tr><td>This orbit (${esc(orbit.name)})</td><td class="great">${fmt(orbit.metric)}</td><td>—</td></tr>`);
  }

  // Target (if set)
  if (typeof campaign?.best?.target === 'number' && typeof orbit.metric === 'number') {
    const d = orbit.metric - campaign.best.target;
    rows.push(`<tr><td>Target</td><td>${fmt(campaign.best.target)}</td><td class="${classForDelta(d)}">${fmtDelta(d)}</td></tr>`);
  }

  // Best parent
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  const parentMetrics = (orbit.parents || []).map(p => byName[p]).filter(p => p && typeof p.metric === 'number');
  if (parentMetrics.length && typeof orbit.metric === 'number') {
    const bp = parentMetrics.reduce((best, p) => {
      if (!best) return p;
      return (isMin ? p.metric < best.metric : p.metric > best.metric) ? p : best;
    }, null);
    if (bp) {
      const d = orbit.metric - bp.metric;
      rows.push(`<tr><td>Best parent (${esc(bp.name)})</td><td>${fmt(bp.metric)}</td><td class="${classForDelta(d)}">${fmtDelta(d)}</td></tr>`);
    }
  }

  // Baseline from methodology
  const baseline = parseBaseline(campaign?.eval_methodology?.baseline);
  if (baseline != null && typeof orbit.metric === 'number') {
    const d = orbit.metric - baseline;
    rows.push(`<tr><td>Baseline</td><td>${fmt(baseline)}</td><td class="${classForDelta(d)}">${fmtDelta(d)}</td></tr>`);
  }

  if (rows.length === 0) return '<p class="empty">No metric recorded for this orbit.</p>';

  return `
    <table class="metric-tbl">
      <thead><tr><th>Reference</th><th>Value</th><th>Δ this orbit</th></tr></thead>
      <tbody>${rows.join('')}</tbody>
    </table>
  `;
}

// ====== Cross-orbit references (inside panel) ===========================
// Auto-computed by publish.py from grep'ing import_from_orbit() and
// metrics_of() calls in each orbit's source. Outbound: orbit.borrows
// (this orbit reads from N siblings). Inbound: orbit.cited_by (this orbit
// is read by N siblings). Each entry carries {orbit, kind, symbol} where
// kind∈{code,data}. No frontmatter schema — code is the declaration.
//
// Multiple borrows from the same orbit get collapsed into a single line
// listing the symbols, so a 5-borrows-from-orbit-X dependency renders
// as one row with five symbol chips, not five separate rows.
function renderCrossRefs(orbit, orbits) {
  const groupBy = (entries) => {
    const m = new Map();
    for (const e of entries || []) {
      if (!m.has(e.orbit)) m.set(e.orbit, []);
      m.get(e.orbit).push(e);
    }
    return m;
  };
  const linkOrbit = (name) =>
    `<a href="javascript:void(0)" class="parent-link" data-orbit="${esc(name)}">${esc(name)}</a>`;
  const renderRow = ([orbitName, entries]) => {
    const codeSyms = entries.filter(e => e.kind === 'code').map(e => e.symbol);
    const dataSyms = entries.filter(e => e.kind === 'data').map(e => e.symbol);
    const symbols = [
      ...codeSyms.map(s => `<code>${esc(s)}</code>`),
      ...dataSyms.map(s => `<code class="data-sym">${esc(s)}</code>`),
    ];
    return `<li>${linkOrbit(orbitName)} <span class="ref-syms">${symbols.join(', ')}</span></li>`;
  };

  const outGrouped = [...groupBy(orbit.borrows).entries()];
  const inGrouped = [...groupBy(orbit.cited_by).entries()];
  const outBlock = outGrouped.length
    ? `<div class="crossref-block">
         <h3>Builds on (${outGrouped.length})</h3>
         <ul class="crossref-list">${outGrouped.map(renderRow).join('')}</ul>
       </div>` : '';
  const inBlock = inGrouped.length
    ? `<div class="crossref-block">
         <h3>Cited by (${inGrouped.length})</h3>
         <ul class="crossref-list">${inGrouped.map(renderRow).join('')}</ul>
       </div>` : '';
  return `<div class="crossref-section">${outBlock}${inBlock}</div>`;
}

// ====== Lineage tree (inside panel) ======================================
function renderLineageTree(orbit, orbits) {
  const paths = buildAncestryPath(orbit.name, orbits);
  if (paths.length === 0 || paths[0].length <= 1) {
    return `<p class="lineage-none">Root orbit — no parents.</p>`;
  }
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  // Show up to 3 longest paths so multi-parent lineage is visible.
  const shown = [...paths].sort((a, b) => b.length - a.length).slice(0, 3);
  const lines = shown.map(path =>
    path.map((name, i) => {
      const o = byName[name];
      const isCurrent = name === orbit.name;
      const cls = isCurrent ? 'current' : (o?.status === 'dead-end' ? 'dim' : 'node-name');
      const label = `<span class="${cls}">${esc(name)}</span>`;
      const arrow = i < path.length - 1 ? ' <span class="edge">→</span> ' : '';
      return `${label}${arrow}`;
    }).join('')
  ).join('\n');
  return `<pre class="tree">${lines}</pre>`;
}

// ====== Issue comments (numbered annotations) ============================
function renderIssueComments(comments) {
  const items = comments.map(c => {
    const author = (c.author || '').toLowerCase();
    let role = '';
    if (author.includes('review')) role = 'reviewer';
    else if (author.includes('verif')) role = 'verifier';
    else if (author.includes('panel') || author.includes('brainstorm') || author.includes('debate')) role = 'panel';
    const roleLabel = role || (author.includes('agent') ? 'agent' : 'note');

    const imgs = (c.images || []).map(url =>
      `<img src="${esc(url)}" alt="" loading="lazy">`
    ).join('');
    return `
      <li>
        <div class="hdr">
          <span class="who">${esc(c.author || '')}</span>
          <span class="role ${role}">${esc(roleLabel)}</span>
          <span>${esc(c.date || '')}</span>
        </div>
        <div class="body">${renderMarkdown(c.body || c.excerpt || '')}${imgs ? imgs : ''}</div>
      </li>
    `;
  }).join('');
  return `<ol class="annotations">${items}</ol>`;
}

// ====== Helpers ===========================================================
function traceAncestry(id, orbits) {
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  const visited = new Set();
  const queue = [id];
  while (queue.length) {
    const cur = queue.shift();
    if (visited.has(cur)) continue;
    visited.add(cur);
    for (const p of (byName[cur]?.parents || [])) {
      if (byName[p]) queue.push(p);
    }
  }
  return visited;
}

function buildAncestryPath(orbitName, orbits) {
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  const paths = [];
  function walk(name, path, seen) {
    if (seen.has(name)) return;               // cycle guard
    const o = byName[name];
    if (!o) return;
    const nextSeen = new Set(seen); nextSeen.add(name);
    const parents = o.parents || [];
    if (parents.length === 0) { paths.push([name, ...path]); return; }
    for (const p of parents) walk(p, [name, ...path], nextSeen);
    if (paths.length > 8) return;              // avoid combinatorial blowup
  }
  walk(orbitName, [], new Set());
  return paths;
}

function multiParentEdgeStyles(orbits, fg) {
  const styles = [];
  for (const o of orbits) {
    if ((o.parents || []).length > 1) {
      styles.push({
        selector: `edge[target = "${o.name}"]`,
        style: { 'width': 2.25, 'line-color': fg, 'opacity': 0.5 },
      });
    }
  }
  return styles;
}

function edgeConnectednessStyles(orbits) {
  const degree = {};
  for (const o of orbits) {
    degree[o.name] = (degree[o.name] || 0);
    for (const p of (o.parents || [])) {
      degree[p] = (degree[p] || 0) + 1;
      degree[o.name] = (degree[o.name] || 0) + 1;
    }
  }
  const maxDeg = Math.max(1, ...Object.values(degree));
  const styles = [];
  for (const o of orbits) {
    for (const p of (o.parents || [])) {
      const avgDeg = ((degree[p] || 0) + (degree[o.name] || 0)) / 2;
      const w = 1.25 + (avgDeg / maxDeg) * 1.75;
      styles.push({
        selector: `edge[source = "${p}"][target = "${o.name}"]`,
        style: { 'width': w },
      });
    }
  }
  return styles;
}

function winnerAncestryPath(orbits) {
  const winner = orbits.find(o => o.status === 'winner');
  if (!winner) return new Set();
  const byName = Object.fromEntries(orbits.map(o => [o.name, o]));
  const visited = new Set();
  const queue = [winner.name];
  while (queue.length) {
    const cur = queue.shift();
    if (visited.has(cur)) continue;
    visited.add(cur);
    for (const p of (byName[cur]?.parents || [])) {
      if (byName[p]) queue.push(p);
    }
  }
  return visited;
}

function tallyStatus(orbits) {
  const t = { total: orbits.length, graduated: 0, 'dead-end': 0, active: 0 };
  for (const o of orbits) {
    if (o.status === 'graduated' || o.status === 'winner') t.graduated++;
    else if (o.status === 'dead-end') t['dead-end']++;
    else t.active++;
  }
  return t;
}

function renderFooter(t, total) {
  return `<footer class="site">
    <span>${total} orbit${total === 1 ? '' : 's'} explored · ${t.graduated} graduated · ${t['dead-end']} dead-end · ${t.active} active</span>
    <span>research-everything</span>
  </footer>`;
}

function metricRank(o, c) {
  if (o.metric == null) return Infinity;
  return c.best && c.best.direction === 'max' ? -o.metric : o.metric;
}

function fmt(v) {
  if (v == null) return '—';
  if (typeof v === 'number') return v.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
  if (typeof v === 'object' && 'value' in v) return fmt(v.value);
  return String(v);
}

function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
}

function issueUrl(data, n) {
  const r = data.campaign?.repo;
  return r ? `https://github.com/${r}/issues/${n}` : '#';
}
