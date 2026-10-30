/**
 * RescueIQ - Disaster Intelligence & Incident Triage System
 * Frontend Architecture & Real FastAPI Integration
 */

// Application State
const state = {
  activeView: 'landing',
  incidents: [],
  stats: {},
  duplicates: [],
  auditLogs: [],
  systemHealth: null,
  currentIncident: null,
  map: null,
  clusterGroup: null,
  citizenMap: null,
  citizenMarker: null,
  dashMap: null,
  dashClusterGroup: null,
  blindSpotLayer: null,
  showBlindSpots: false,
  charts: {},
  filters: {
    search: '',
    verificationStatus: 'all',
    category: 'all',
    damageClass: 'all',
    incidentStatus: 'all',
    sortBy: 'priority_desc'
  },
  userRole: 'Coordinator' // Coordinator | Responder | Citizen
};

// Color Tokens
const SEVERITY_COLORS = {
  'Severe damage': '#dc2626',
  'Mild damage': '#ea580c',
  'Little-or-none': '#16a34a',
  'Abstained / Uncertain': '#9333ea'
};

// Utilities
function esc(str) {
  return String(str ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function formatDate(isoStr) {
  if (!isoStr) return 'N/A';
  try {
    const d = new Date(isoStr);
    return d.toLocaleString(undefined, {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
    });
  } catch (e) {
    return isoStr;
  }
}

function showToast(message, type = 'success') {
  const container = document.getElementById('toastContainer');
  if (!container) return;
  const toast = document.createElement('div');
  toast.className = `toast-msg ${type}`;
  toast.innerHTML = `
    <span>${esc(message)}</span>
    <button style="background:none;border:none;cursor:pointer;color:inherit;font-size:16px;margin-left:8px;" onclick="this.parentElement.remove()">×</button>
  `;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}

// Navigation & View Routing
function switchView(viewId) {
  state.activeView = viewId;
  window.location.hash = viewId;

  // Update Sidebar Links
  document.querySelectorAll('.nav-item').forEach(item => {
    if (item.dataset.view === viewId) {
      item.classList.add('active');
    } else {
      item.classList.remove('active');
    }
  });

  // Switch View Containers
  document.querySelectorAll('.view-container').forEach(container => {
    if (container.id === `view-${viewId}`) {
      container.classList.add('active');
    } else {
      container.classList.remove('active');
    }
  });

  // Update Topbar Title
  const titles = {
    'landing': 'Platform Overview',
    'citizen-portal': 'Citizen Incident Reporting Portal',
    'dashboard': 'Emergency Operations Dashboard',
    'incidents': 'Incident Triage & Management Queue',
    'map-view': 'Interactive Situational Hazard Map',
    'ai-analysis': 'AI Vision & Multimodal Explainability',
    'analytics': 'Disaster Intelligence Analytics',
    'admin': 'System Status & Audit Control'
  };
  const titleEl = document.getElementById('topbarTitle');
  if (titleEl) titleEl.textContent = titles[viewId] || 'Damage Mapper';

  // Invalidate Map sizes when switching to map views
  if (viewId === 'map-view' && state.map) {
    setTimeout(() => state.map.invalidateSize(), 200);
  }
  if (viewId === 'dashboard' && state.dashMap) {
    setTimeout(() => state.dashMap.invalidateSize(), 200);
  }
  if (viewId === 'citizen-portal' && state.citizenMap) {
    setTimeout(() => state.citizenMap.invalidateSize(), 200);
  }
  if (viewId === 'analytics') {
    renderAnalyticsCharts();
  }

  // Close mobile sidebar if open
  closeMobileSidebar();
}

function setupRouting() {
  window.addEventListener('hashchange', () => {
    const hash = window.location.hash.replace('#', '');
    if (hash && document.getElementById(`view-${hash}`)) {
      switchView(hash);
    }
  });

  const initialHash = window.location.hash.replace('#', '');
  if (initialHash && document.getElementById(`view-${initialHash}`)) {
    switchView(initialHash);
  } else {
    switchView('landing');
  }
}

// Mobile Sidebar Controls
function toggleMobileSidebar() {
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  if (sidebar) sidebar.classList.toggle('open');
  if (backdrop) backdrop.classList.toggle('active');
}
window.toggleSidebar = toggleMobileSidebar;
window.toggleMobileSidebar = toggleMobileSidebar;
window.switchView = switchView;

function closeMobileSidebar() {
  const sidebar = document.getElementById('sidebar');
  const backdrop = document.getElementById('sidebarBackdrop');
  if (sidebar) sidebar.classList.remove('open');
  if (backdrop) backdrop.classList.remove('active');
}
window.closeMobileSidebar = closeMobileSidebar;

// Data Fetching & Sync with FastAPI
async function loadAllData() {
  try {
    const [statsRes, incRes, dupRes, healthRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/incidents'),
      fetch('/api/duplicates'),
      fetch('/api/health')
    ]);

    if (statsRes.ok) state.stats = await statsRes.json();
    if (incRes.ok) state.incidents = await incRes.json();
    if (dupRes.ok) state.duplicates = await dupRes.json();
    if (healthRes.ok) state.systemHealth = await healthRes.json();

    updateDashboardKPIs();
    renderIncidentsQueue();
    renderInteractiveMap();
    renderDashboardSnapshot();
    renderDuplicatesView();
    renderAdminView();
    if (state.activeView === 'analytics') {
      renderAnalyticsCharts();
    }
  } catch (err) {
    console.error('Error loading data from FastAPI:', err);
    showToast('Failed to connect to RescueIQ backend service.', 'error');
  }
}

// Update KPI Stats on Dashboard & Landing
function updateDashboardKPIs() {
  const s = state.stats;
  const setText = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.textContent = val ?? 0;
  };

  setText('statTotal', s.total);
  setText('statPending', s.pending);
  setText('statSevere', s.severe);
  setText('statAbstained', s.abstained);
  setText('statDuplicates', s.duplicates);
  setText('statAssigned', s.assigned);

  // Landing Page Counters
  setText('heroStatTotal', s.total);
  setText('heroStatSevere', s.severe);
  setText('heroStatVerified', s.verified);
  setText('heroStatAssigned', s.assigned);

  // Sidebar alert badge
  const pendingBadge = document.getElementById('sidebarPendingBadge');
  if (pendingBadge) {
    pendingBadge.textContent = s.pending || 0;
    pendingBadge.style.display = s.pending > 0 ? 'inline-block' : 'none';
  }

  const dupBadge = document.getElementById('sidebarDupBadge');
  if (dupBadge) {
    dupBadge.textContent = s.duplicates || 0;
    dupBadge.style.display = s.duplicates > 0 ? 'inline-block' : 'none';
  }

  // Dashboard Urgent Alerts Banner
  renderUrgentAlerts();
}

function renderUrgentAlerts() {
  const container = document.getElementById('urgentAlertsContainer');
  if (!container) return;

  const urgent = state.incidents
    .filter(i => i.damage_class === 'Severe damage' || i.priority_score >= 70)
    .slice(0, 3);

  if (!urgent.length) {
    container.innerHTML = `
      <div style="background:#f0fdf4;border:1px solid #bbf7d0;color:#166534;padding:12px 16px;border-radius:var(--radius-md);font-size:13px;display:flex;align-items:center;gap:10px;">
        <span>🛡️</span>
        <span><b>All Clear:</b> No high-severity critical incidents currently awaiting triage.</span>
      </div>
    `;
    return;
  }

  container.innerHTML = `
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:var(--radius-md);padding:14px 18px;margin-bottom:16px;">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
        <span style="color:#b91c1c;font-weight:700;font-size:13px;display:flex;align-items:center;gap:6px;">
          <span style="font-size:16px">🚨</span> Urgent Triage Attention Required (${urgent.length})
        </span>
        <button class="btn btn-sm btn-ghost" onclick="switchView('incidents')" style="color:#b91c1c;">View All in Queue →</button>
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(280px, 1fr));gap:10px;">
        ${urgent.map(inc => `
          <div style="background:#ffffff;border:1px solid #fca5a5;border-radius:var(--radius-sm);padding:10px 12px;display:flex;gap:10px;align-items:center;">
            <div style="width:38px;height:38px;border-radius:var(--radius-sm);background:#fee2e2;color:#dc2626;font-weight:800;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0;">
              ${Number(inc.priority_score).toFixed(0)}
            </div>
            <div style="flex:1;min-width:0;">
              <div style="font-weight:700;font-size:12px;color:var(--navy-900);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                ${esc(inc.category)} · ${esc(inc.damage_class)}
              </div>
              <div style="font-size:11px;color:var(--text-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                ${esc(inc.description)}
              </div>
            </div>
            <button class="btn btn-sm btn-primary" onclick="openIncidentDetails('${esc(inc.id)}')">Inspect</button>
          </div>
        `).join('')}
      </div>
    </div>
  `;
}

// Incident Management & Queue
function renderIncidentsQueue() {
  const container = document.getElementById('incidentsQueueList');
  if (!container) return;

  const f = state.filters;
  let filtered = state.incidents.filter(inc => {
    // Text search
    if (f.search) {
      const q = f.search.toLowerCase();
      const matchDesc = (inc.description || '').toLowerCase().includes(q);
      const matchReporter = (inc.reporter || '').toLowerCase().includes(q);
      const matchId = (inc.id || '').toLowerCase().includes(q);
      const matchCat = (inc.category || '').toLowerCase().includes(q);
      if (!matchDesc && !matchReporter && !matchId && !matchCat) return false;
    }
    // Verification filter
    if (f.verificationStatus !== 'all' && inc.verification_status !== f.verificationStatus) return false;
    // Category filter
    if (f.category !== 'all' && inc.category !== f.category) return false;
    // Damage class filter
    if (f.damageClass !== 'all' && inc.damage_class !== f.damageClass) return false;
    // Incident status filter
    if (f.incidentStatus !== 'all' && inc.incident_status !== f.incidentStatus) return false;

    return true;
  });

  // Sorting
  filtered.sort((a, b) => {
    if (f.sortBy === 'priority_desc') return b.priority_score - a.priority_score;
    if (f.sortBy === 'priority_asc') return a.priority_score - b.priority_score;
    if (f.sortBy === 'date_desc') return new Date(b.created_at) - new Date(a.created_at);
    if (f.sortBy === 'urgency_desc') return b.urgency_score - a.urgency_score;
    if (f.sortBy === 'confidence_desc') return b.model_confidence - a.model_confidence;
    return 0;
  });

  const countBadge = document.getElementById('filteredIncidentsCount');
  if (countBadge) countBadge.textContent = `${filtered.length} of ${state.incidents.length} incidents`;

  if (!filtered.length) {
    container.innerHTML = `
      <div style="background:var(--bg-surface);border:1px dashed var(--border-strong);padding:40px;text-align:center;border-radius:var(--radius-md);color:var(--text-muted)">
        <p style="font-size:16px;font-weight:600;margin-bottom:6px;">No incidents match active filters.</p>
        <p style="font-size:12px;">Try adjusting your search query, hazard category, or verification status filters.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = filtered.map(inc => {
    const b = inc.factor_breakdown || {};
    const sev = b.severity || { raw: 0, weighted: 0 };
    const urg = b.urgency || { raw: 0, weighted: 0 };
    const imp = b.impact || { raw: 0, weighted: 0 };
    const acc = b.access || { raw: 0, weighted: 0 };

    const tagClass = inc.damage_class.includes('Severe') ? 'Severe'
      : inc.damage_class.includes('Mild') ? 'Mild'
      : inc.damage_class.includes('Little') ? 'Little'
      : 'Abstained';

    const photoUrl = inc.image_url || '/uploads/demo-scenario.png';

    return `
      <article class="incident-card" id="inc-card-${esc(inc.id)}">
        <img src="${esc(photoUrl)}" class="incident-thumb" alt="Incident Photo" onclick="openLightbox('${esc(photoUrl)}')" onerror="this.src='/uploads/demo-scenario.png'">
        
        <div class="incident-body">
          <div class="incident-header">
            <span class="tag ${tagClass}">
              <span style="width:7px;height:7px;border-radius:50%;background:${SEVERITY_COLORS[inc.damage_class] || '#94a3b8'}"></span>
              ${esc(inc.damage_class)}
            </span>
            <span style="font-size:12px;font-weight:600;color:var(--navy-800)">${esc(inc.category)}</span>
            <span class="badge-status ${esc(inc.verification_status)}">${esc(inc.verification_status)}</span>
            <span style="font-size:11px;color:var(--text-muted);background:var(--bg-subtle);padding:2px 7px;border-radius:var(--radius-sm)">${esc(inc.incident_status)}</span>
            ${inc.assigned_to ? `<span style="font-size:11px;color:#0369a1;background:#e0f2fe;padding:2px 7px;border-radius:var(--radius-sm)">👤 ${esc(inc.assigned_to)}</span>` : ''}
          </div>

          <p class="incident-desc">${esc(inc.description)}</p>

          <!-- Factor Breakdown Bar -->
          <div class="factor-breakdown-bar">
            <div class="factor-item"><span>Severity (40%):</span><b>${sev.raw} → ${sev.weighted}</b></div>
            <div class="factor-item"><span>Urgency (25%):</span><b>${urg.raw} → ${urg.weighted}</b></div>
            <div class="factor-item"><span>Impact (20%):</span><b>${imp.raw} → ${imp.weighted}</b></div>
            <div class="factor-item"><span>Access (15%):</span><b>${acc.raw} → ${acc.weighted}</b></div>
          </div>

          <div class="incident-meta">
            <span>📅 ${formatDate(inc.created_at)}</span>
            <span>📍 ${inc.latitude.toFixed(4)}, ${inc.longitude.toFixed(4)}</span>
            <span>👤 ${esc(inc.reporter || 'Anonymous')}</span>
            <span>🤖 Conf: <b>${(inc.model_confidence * 100).toFixed(1)}%</b></span>
            ${inc.access_score > 50 ? `<span style="color:#b91c1c;font-weight:600">⚠️ Reported access blockage (unconfirmed)</span>` : ''}
            ${inc.duplicate_of ? `<span style="color:#d97706;font-weight:600">🔗 Duplicate of ${esc(inc.duplicate_of)}</span>` : ''}
          </div>

          <!-- Actions -->
          <div class="incident-actions">
            <button class="btn btn-sm btn-success" onclick="verifyIncident('${esc(inc.id)}', 'Verified')">✓ Verify</button>
            <button class="btn btn-sm btn-danger" onclick="verifyIncident('${esc(inc.id)}', 'Rejected')">✕ Reject</button>
            <button class="btn btn-sm btn-secondary" onclick="openAssignModal('${esc(inc.id)}')">👤 Assign Team</button>
            
            <select class="form-select" style="width:auto;font-size:11px;padding:4px 8px;height:auto;" onchange="overrideSeverity('${esc(inc.id)}', this.value)">
              <option value="">⚙ Override AI Severity…</option>
              <option value="Severe damage">Severe damage (95)</option>
              <option value="Mild damage">Mild damage (55)</option>
              <option value="Little-or-none">Little-or-none (15)</option>
            </select>

            <button class="btn btn-sm btn-ghost" onclick="openIncidentDetails('${esc(inc.id)}')">🔍 Details & Audit Log</button>
          </div>
        </div>

        <div class="priority-badge-box">
          <div class="priority-num">${Number(inc.priority_score).toFixed(1)}</div>
          <div class="priority-label">Priority Score</div>
        </div>
      </article>
    `;
  }).join('');
}

// Coordinator Action Handlers
async function verifyIncident(id, status) {
  try {
    const res = await fetch(`/api/incidents/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ verification_status: status, actor: `${state.userRole} (${status})` })
    });
    if (!res.ok) throw new Error('Verification update failed');
    showToast(`Incident successfully marked as ${status}.`);
    await loadAllData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

async function overrideSeverity(id, newClass) {
  if (!newClass) return;
  try {
    const res = await fetch(`/api/incidents/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ override_damage_class: newClass, actor: `${state.userRole} Override` })
    });
    if (!res.ok) throw new Error('Severity override failed');
    showToast(`AI Severity overridden to ${newClass}. Priority recalculated!`);
    await loadAllData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

function openAssignModal(id) {
  const inc = state.incidents.find(i => i.id === id);
  if (!inc) return;
  state.currentIncident = inc;

  const input = document.getElementById('assignTeamInput');
  if (input) input.value = inc.assigned_to || '';

  const modal = document.getElementById('assignModal');
  if (modal && typeof modal.showModal === 'function') {
    modal.showModal();
  }
}

async function submitAssignment() {
  if (!state.currentIncident) return;
  const team = (document.getElementById('assignTeamInput').value || '').trim();
  const status = team ? 'Assigned' : 'Reported';
  try {
    const res = await fetch(`/api/incidents/${state.currentIncident.id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ assigned_to: team, incident_status: status, actor: state.userRole })
    });
    if (!res.ok) throw new Error('Assignment failed');
    showToast(`Assigned to ${team || 'Unassigned'}. Response status: ${status}.`);
    closeDialog('assignModal');
    await loadAllData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Full Incident Details & Audit Trail Modal
async function openIncidentDetails(id) {
  try {
    const res = await fetch(`/api/incidents/${id}`);
    if (!res.ok) throw new Error('Could not fetch incident details');
    const data = await res.json();
    state.currentIncident = data;

    const modal = document.getElementById('incidentDetailsModal');
    const body = document.getElementById('incidentDetailsBody');
    if (!modal || !body) return;

    const b = data.factor_breakdown || {};
    const sev = b.severity || { raw: 0, weighted: 0 };
    const urg = b.urgency || { raw: 0, weighted: 0 };
    const imp = b.impact || { raw: 0, weighted: 0 };
    const acc = b.access || { raw: 0, weighted: 0 };

    body.innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-bottom:16px;">
        <div>
          <img src="${esc(data.image_url)}" style="width:100%;border-radius:var(--radius-sm);border:1px solid var(--border-light);" onerror="this.src='/uploads/demo-scenario.png'">
          <div style="margin-top:10px;font-size:12px;color:var(--text-subtle);">
            <div><b>Report ID:</b> ${esc(data.id)}</div>
            <div><b>Created:</b> ${formatDate(data.created_at)}</div>
            <div><b>Reporter:</b> ${esc(data.reporter || 'Anonymous')}</div>
            <div><b>Coordinates:</b> ${data.latitude.toFixed(6)}, ${data.longitude.toFixed(6)}</div>
          </div>
        </div>

        <div>
          <div style="background:var(--bg-subtle);padding:14px;border-radius:var(--radius-sm);margin-bottom:12px;border:1px solid var(--border-light);">
            <div style="font-size:11px;text-transform:uppercase;color:var(--text-subtle);font-weight:700;">Calculated Priority Score</div>
            <div style="font-size:32px;font-weight:800;color:var(--navy-900);line-height:1.1;">${Number(data.priority_score).toFixed(1)} <span style="font-size:14px;font-weight:600;color:var(--text-muted)">/ 100</span></div>
            <div style="font-size:11px;color:var(--text-subtle);margin-top:4px;">Formula: 0.40·S + 0.25·U + 0.20·I + 0.15·A</div>
          </div>

          <div style="font-size:12px;display:flex;flex-direction:column;gap:6px;margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between"><span>Severity (40%):</span><b>${sev.raw} → ${sev.weighted} pts</b></div>
            <div style="display:flex;justify-content:space-between"><span>Urgency (25%):</span><b>${urg.raw} → ${urg.weighted} pts</b></div>
            <div style="display:flex;justify-content:space-between"><span>Impact (20%):</span><b>${imp.raw} → ${imp.weighted} pts</b></div>
            <div style="display:flex;justify-content:space-between"><span>Access (15%):</span><b>${acc.raw} → ${acc.weighted} pts</b></div>
          </div>

          <div style="padding:10px;background:#f8fafc;border-radius:var(--radius-sm);font-size:12px;border:1px solid var(--border-light)">
            <div><b>AI Vision Model:</b> ${esc(data.model_name)}</div>
            <div><b>Confidence:</b> ${(data.model_confidence * 100).toFixed(1)}%</div>
            <div><b>Abstained / Uncertain:</b> ${data.is_abstained ? 'YES (routed for manual review)' : 'NO'}</div>
          </div>
        </div>
      </div>

      <div style="margin-bottom:16px;">
        <h4 style="font-size:13px;font-weight:700;margin-bottom:6px;color:var(--navy-900)">Citizen Description & Hazard Assessment</h4>
        <p style="font-size:13px;line-height:1.5;background:var(--bg-subtle);padding:10px;border-radius:var(--radius-sm);border:1px solid var(--border-light)">${esc(data.description)}</p>
      </div>

      <div>
        <h4 style="font-size:13px;font-weight:700;margin-bottom:8px;color:var(--navy-900)">Immutable Audit History</h4>
        <div style="max-height:180px;overflow-y:auto;border:1px solid var(--border-light);border-radius:var(--radius-sm)">
          <table class="data-table">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Action</th>
                <th>Change</th>
                <th>Actor</th>
              </tr>
            </thead>
            <tbody>
              ${(data.history || []).map(h => `
                <tr>
                  <td>${formatDate(h.created_at)}</td>
                  <td><b>${esc(h.action)}</b></td>
                  <td>${esc(h.previous_value)} → ${esc(h.new_value)}</td>
                  <td>${esc(h.actor)}</td>
                </tr>
              `).join('') || '<tr><td colspan="4" style="text-align:center;color:var(--text-muted)">No audit entries recorded yet.</td></tr>'}
            </tbody>
          </table>
        </div>
      </div>
    `;

    modal.showModal();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Lightbox modal for photo zoom
function openLightbox(url) {
  const modal = document.getElementById('lightboxModal');
  const img = document.getElementById('lightboxImage');
  if (modal && img) {
    img.src = url;
    modal.showModal();
  }
}

function closeDialog(modalId) {
  const modal = document.getElementById(modalId);
  if (modal && typeof modal.close === 'function') {
    modal.close();
  }
}

// Setup Dialog Light Dismiss & Escape behavior per modern-web-guidance
function setupDialogBehaviors() {
  document.querySelectorAll('dialog').forEach(dialog => {
    // Esc closes dialog naturally in native <dialog>
    dialog.addEventListener('cancel', () => {
      dialog.returnValue = '';
    });

    // Pointer-down and up outside detection for light-dismiss fallback
    const isOutside = (e, d) => {
      const r = d.getBoundingClientRect();
      return e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom;
    };

    let pressedOutside = false;
    dialog.addEventListener('pointerdown', e => {
      pressedOutside = isOutside(e, dialog);
    });
    dialog.addEventListener('pointerup', e => {
      if (pressedOutside && isOutside(e, dialog)) {
        dialog.close();
      }
    });
  });
}

// Citizen Incident Report Form Submission
function setupCitizenForm() {
  const form = document.getElementById('citizenReportForm');
  if (!form) return;

  const fileInput = document.getElementById('citizenImageInput');
  const dropzone = document.getElementById('citizenDropzone');
  const preview = document.getElementById('citizenImagePreview');
  const previewImg = document.getElementById('citizenPreviewImg');

  // Drag and drop handling
  if (dropzone && fileInput) {
    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, e => {
        e.preventDefault();
        dropzone.classList.add('dragover');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, e => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
      });
    });
    dropzone.addEventListener('drop', e => {
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        fileInput.files = e.dataTransfer.files;
        handleImageSelection(fileInput.files[0]);
      }
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files && fileInput.files[0]) {
        handleImageSelection(fileInput.files[0]);
      }
    });
  }

  function handleImageSelection(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
      if (previewImg && preview) {
        previewImg.src = e.target.result;
        preview.style.display = 'block';
      }
    };
    reader.readAsDataURL(file);
  }

  // Urgency slider label sync
  const urgencyInput = document.getElementById('citizenUrgencyInput');
  const urgencyLabel = document.getElementById('citizenUrgencyLabel');
  if (urgencyInput && urgencyLabel) {
    urgencyInput.addEventListener('input', () => {
      const val = parseInt(urgencyInput.value, 10);
      let desc = 'Low Priority';
      if (val >= 80) desc = 'Critical / Life-Threatening';
      else if (val >= 60) desc = 'High Urgency';
      else if (val >= 35) desc = 'Moderate Urgency';
      urgencyLabel.textContent = `${val} (${desc})`;
    });
  }

  // Form Submit
  form.addEventListener('submit', async e => {
    e.preventDefault();
    const submitBtn = document.getElementById('citizenSubmitBtn');
    submitBtn.disabled = true;
    submitBtn.innerHTML = '<span>⚙️</span> Analyzing Photo & Triaging Report…';

    try {
      const fd = new FormData(form);
      fd.set('access_blocked', form.access_blocked && form.access_blocked.checked ? 'true' : 'false');

      const res = await fetch('/api/incidents', {
        method: 'POST',
        body: fd
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Report submission failed');

      form.reset();
      if (preview) preview.style.display = 'none';

      // Show Confirmation Card with Tracking ID
      displaySubmissionSuccess(data);
      showToast(`Report triaged! Priority Score: ${data.priority_score}`, 'success');

      await loadAllData();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = '<span>🚀</span> Submit Emergency Report';
    }
  });
}

function displaySubmissionSuccess(incident) {
  const container = document.getElementById('citizenSuccessNotice');
  if (!container) return;
  container.innerHTML = `
    <div style="background:#ecfdf5;border:1px solid #a7f3d0;border-radius:var(--radius-md);padding:20px;margin-top:20px;">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
        <span style="font-size:24px;">✅</span>
        <div>
          <h4 style="font-size:16px;font-weight:700;color:#065f46;margin:0;">Report Successfully Triaged & Dispatched!</h4>
          <p style="font-size:12px;color:#047857;margin:0;">Your report has been analyzed by RescueIQ Vision AI and queued for emergency coordinators.</p>
        </div>
      </div>

      <div style="background:#ffffff;border:1px solid #6ee7b7;border-radius:var(--radius-sm);padding:14px;display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;">
        <div>
          <div style="font-size:11px;text-transform:uppercase;color:var(--text-subtle);font-weight:700">Citizen Tracking ID</div>
          <div style="font-size:18px;font-weight:800;color:var(--navy-900);font-family:monospace">${esc(incident.id)}</div>
        </div>
        <button class="btn btn-sm btn-secondary" onclick="navigator.clipboard.writeText('${esc(incident.id)}');showToast('Tracking ID copied to clipboard!')">📋 Copy ID</button>
      </div>

      <div style="margin-top:12px;font-size:12px;display:grid;grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));gap:8px;color:#065f46;">
        <div>Damage Class: <b>${esc(incident.damage_class)}</b></div>
        <div>Model Confidence: <b>${(incident.model_confidence * 100).toFixed(1)}%</b></div>
        <div>Priority Score: <b>${Number(incident.priority_score).toFixed(1)}</b></div>
        <div>Status: <b>${esc(incident.verification_status)}</b></div>
      </div>
    </div>
  `;
  container.scrollIntoView({ behavior: 'smooth' });
}

// Citizen Tracking Lookup Tool
async function trackReportById() {
  const input = document.getElementById('citizenTrackInput');
  const resultBox = document.getElementById('citizenTrackResult');
  if (!input || !resultBox) return;

  const id = input.value.trim();
  if (!id) {
    showToast('Please enter a valid Tracking ID', 'error');
    return;
  }

  try {
    const res = await fetch(`/api/incidents/${id}`);
    if (!res.ok) throw new Error('Tracking ID not found. Please verify the ID.');
    const d = await res.json();

    resultBox.innerHTML = `
      <div style="background:#ffffff;border:1px solid var(--border-light);border-radius:var(--radius-md);padding:16px;margin-top:14px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
          <span style="font-weight:700;font-size:14px;color:var(--navy-900)">Report Status Details</span>
          <span class="badge-status ${esc(d.verification_status)}">${esc(d.verification_status)}</span>
        </div>
        <div style="font-size:12px;display:flex;flex-direction:column;gap:6px;">
          <div><b>Category:</b> ${esc(d.category)}</div>
          <div><b>Description:</b> ${esc(d.description)}</div>
          <div><b>AI Triage Class:</b> ${esc(d.damage_class)} (${(d.model_confidence * 100).toFixed(1)}% conf)</div>
          <div><b>Priority Score:</b> ${d.priority_score}</div>
          <div><b>Response Status:</b> ${esc(d.incident_status)}</div>
          <div><b>Assigned Response Unit:</b> ${esc(d.assigned_to || 'Pending team assignment')}</div>
          <div><b>Reported At:</b> ${formatDate(d.created_at)}</div>
        </div>
      </div>
    `;
  } catch (err) {
    resultBox.innerHTML = `
      <div style="background:#fef2f2;border:1px solid #fecaca;color:#b91c1c;padding:12px;border-radius:var(--radius-sm);font-size:12px;margin-top:10px;">
        ${esc(err.message)}
      </div>
    `;
  }
}

// Maps Integration (Leaflet & MarkerCluster)
function initMaps() {
  // 1. Full Interactive Map
  const mapEl = document.getElementById('interactiveMap');
  if (mapEl && !state.map) {
    state.map = L.map('interactiveMap').setView([12.9716, 77.5946], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap contributors'
    }).addTo(state.map);

    state.clusterGroup = L.markerClusterGroup({
      showCoverageOnHover: false,
      maxClusterRadius: 50
    });
    state.map.addLayer(state.clusterGroup);
  }

  // 2. Citizen Location Picker Mini-Map
  const citizenMapEl = document.getElementById('citizenLocationMap');
  if (citizenMapEl && !state.citizenMap) {
    state.citizenMap = L.map('citizenLocationMap').setView([12.9716, 77.5946], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OpenStreetMap'
    }).addTo(state.citizenMap);

    state.citizenMap.on('click', e => {
      const lat = e.latlng.lat.toFixed(6);
      const lon = e.latlng.lng.toFixed(6);
      document.getElementById('citizenLat').value = lat;
      document.getElementById('citizenLon').value = lon;

      if (state.citizenMarker) {
        state.citizenMarker.setLatLng(e.latlng);
      } else {
        state.citizenMarker = L.marker(e.latlng).addTo(state.citizenMap);
      }
      state.citizenMarker.bindPopup('Incident Location Pin').openPopup();
    });
  }

  // 3. Dashboard Mini Snapshot Map
  const dashMapEl = document.getElementById('dashboardMapSnapshot');
  if (dashMapEl && !state.dashMap) {
    state.dashMap = L.map('dashboardMapSnapshot').setView([12.9716, 77.5946], 12);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      maxZoom: 19,
      attribution: '© OSM'
    }).addTo(state.dashMap);

    state.dashClusterGroup = L.markerClusterGroup();
    state.dashMap.addLayer(state.dashClusterGroup);
  }
}

function renderInteractiveMap() {
  if (!state.map || !state.clusterGroup) return;
  state.clusterGroup.clearLayers();

  const incs = state.incidents.filter(i => i.latitude != null && i.longitude != null);
  const bounds = [];

  incs.forEach(inc => {
    const color = SEVERITY_COLORS[inc.damage_class] || '#94a3b8';
    const marker = L.circleMarker([inc.latitude, inc.longitude], {
      radius: 9,
      color: '#ffffff',
      fillColor: color,
      fillOpacity: 0.9,
      weight: 2
    });

    marker.bindPopup(`
      <div style="font-family:inherit;font-size:12px;min-width:200px">
        <div style="font-weight:700;color:var(--navy-900);font-size:13px;margin-bottom:4px;">
          ${esc(inc.damage_class)}
        </div>
        <div style="color:var(--text-muted);margin-bottom:6px;">
          <b>${esc(inc.category)}</b> · Priority: <b>${Number(inc.priority_score).toFixed(1)}</b>
        </div>
        <p style="margin:0 0 8px;font-size:11px;color:var(--text-main);">${esc(inc.description)}</p>
        <div style="display:flex;gap:6px;">
          <button class="btn btn-sm btn-primary" onclick="openIncidentDetails('${esc(inc.id)}')">Inspect</button>
        </div>
      </div>
    `);

    state.clusterGroup.addLayer(marker);
    bounds.push([inc.latitude, inc.longitude]);
  });

  if (bounds.length > 0) {
    state.map.fitBounds(bounds, { padding: [40, 40] });
  }

  // Blind-spot Silent-Zone Indicator Layer (GEO-6)
  renderSilentZoneBlindSpots();
}

function renderDashboardSnapshot() {
  if (!state.dashMap || !state.dashClusterGroup) return;
  state.dashClusterGroup.clearLayers();

  const bounds = [];
  state.incidents.forEach(inc => {
    if (inc.latitude == null) return;
    const color = SEVERITY_COLORS[inc.damage_class] || '#94a3b8';
    const marker = L.circleMarker([inc.latitude, inc.longitude], {
      radius: 7,
      color: '#ffffff',
      fillColor: color,
      fillOpacity: 0.85,
      weight: 1.5
    });
    state.dashClusterGroup.addLayer(marker);
    bounds.push([inc.latitude, inc.longitude]);
  });

  if (bounds.length > 0) {
    state.dashMap.fitBounds(bounds, { padding: [20, 20] });
  }
}

// Rule-Based Silent Zone / Blind Spot Flag (GEO-6)
function renderSilentZoneBlindSpots() {
  if (!state.map) return;
  if (state.blindSpotLayer) {
    state.map.removeLayer(state.blindSpotLayer);
    state.blindSpotLayer = null;
  }

  if (!state.showBlindSpots || !state.incidents.length) return;

  const valid = state.incidents.filter(i => i.latitude != null);
  if (!valid.length) return;

  const lats = valid.map(i => i.latitude);
  const lons = valid.map(i => i.longitude);
  const minLat = Math.min(...lats) - 0.02;
  const maxLat = Math.max(...lats) + 0.02;
  const minLon = Math.min(...lons) - 0.02;
  const maxLon = Math.max(...lons) + 0.02;

  const gridSteps = 4;
  const latStep = (maxLat - minLat) / gridSteps;
  const lonStep = (maxLon - minLon) / gridSteps;

  const blindPolys = [];

  for (let r = 0; r < gridSteps; r++) {
    for (let c = 0; c < gridSteps; c++) {
      const cellMinLat = minLat + r * latStep;
      const cellMaxLat = cellMinLat + latStep;
      const cellMinLon = minLon + c * lonStep;
      const cellMaxLon = cellMinLon + lonStep;

      // Count incidents inside cell
      const count = valid.filter(i =>
        i.latitude >= cellMinLat && i.latitude < cellMaxLat &&
        i.longitude >= cellMinLon && i.longitude < cellMaxLon
      ).length;

      // If cell has 0 reports in an active disaster perimeter, highlight as potential silent zone
      if (count === 0) {
        const poly = L.rectangle([[cellMinLat, cellMinLon], [cellMaxLat, cellMaxLon]], {
          color: '#e11d48',
          weight: 1.5,
          dashArray: '4, 4',
          fillColor: '#fda4af',
          fillOpacity: 0.25
        }).bindPopup('<b>Silent Zone Alert (GEO-6)</b><br>Zero citizen reports received in this exposed zone. May indicate communication outage or isolated population.');
        blindPolys.push(poly);
      }
    }
  }

  state.blindSpotLayer = L.layerGroup(blindPolys).addTo(state.map);
}

function toggleBlindSpots() {
  state.showBlindSpots = !state.showBlindSpots;
  const btn = document.getElementById('btnToggleBlindSpots');
  if (btn) {
    btn.textContent = state.showBlindSpots ? 'Hide Silent Zones' : 'Show Silent Zones (GEO-6)';
    btn.classList.toggle('btn-primary', state.showBlindSpots);
  }
  renderSilentZoneBlindSpots();
}

function captureDeviceGPS() {
  if (!navigator.geolocation) {
    showToast('Geolocation is not supported by your browser.', 'error');
    return;
  }
  navigator.geolocation.getCurrentPosition(
    pos => {
      const lat = pos.coords.latitude.toFixed(6);
      const lon = pos.coords.longitude.toFixed(6);
      document.getElementById('citizenLat').value = lat;
      document.getElementById('citizenLon').value = lon;

      if (state.citizenMap) {
        state.citizenMap.setView([lat, lon], 15);
        if (state.citizenMarker) {
          state.citizenMarker.setLatLng([lat, lon]);
        } else {
          state.citizenMarker = L.marker([lat, lon]).addTo(state.citizenMap);
        }
      }
      showToast('GPS coordinates captured accurately.');
    },
    err => showToast(`GPS error: ${err.message}`, 'error'),
    { enableHighAccuracy: true, timeout: 8000 }
  );
}

function locateMeOnMainMap() {
  if (!navigator.geolocation) return showToast('Geolocation unsupported', 'error');
  navigator.geolocation.getCurrentPosition(pos => {
    if (state.map) {
      state.map.setView([pos.coords.latitude, pos.coords.longitude], 14);
      L.circleMarker([pos.coords.latitude, pos.coords.longitude], {
        radius: 8, color: '#2563eb', fillColor: '#38bdf8', fillOpacity: 0.9
      }).addTo(state.map).bindPopup('Your Current Location').openPopup();
    }
  });
}

// AI Analysis View: Live Vision Classifier Playground
function setupLiveVisionPlayground() {
  const fileInput = document.getElementById('aiPlaygroundInput');
  const analyzeBtn = document.getElementById('aiPlaygroundBtn');
  const resultBox = document.getElementById('aiPlaygroundResult');
  const previewImg = document.getElementById('aiPlaygroundPreview');

  if (!fileInput || !analyzeBtn) return;

  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) {
      const reader = new FileReader();
      reader.onload = e => {
        if (previewImg) {
          previewImg.src = e.target.result;
          previewImg.style.display = 'block';
        }
      };
      reader.readAsDataURL(fileInput.files[0]);
    }
  });

  analyzeBtn.addEventListener('click', async () => {
    if (!fileInput.files || !fileInput.files[0]) {
      showToast('Please select an image first', 'error');
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Running EfficientNet-B0 Inference…';

    try {
      const fd = new FormData();
      fd.append('image', fileInput.files[0]);
      const res = await fetch('/api/analyze/image', {
        method: 'POST',
        body: fd
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Inference failed');

      const p = data.prediction;
      const probs = p.probabilities || {};

      resultBox.innerHTML = `
        <div style="background:#ffffff;border:1px solid var(--border-light);border-radius:var(--radius-md);padding:18px;margin-top:14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <span style="font-size:15px;font-weight:700;color:var(--navy-900);">Prediction Result</span>
            <span class="tag ${p.damage_class.includes('Severe') ? 'Severe' : p.damage_class.includes('Mild') ? 'Mild' : p.damage_class.includes('Little') ? 'Little' : 'Abstained'}">
              ${esc(p.damage_class)}
            </span>
          </div>

          <div style="font-size:12px;margin-bottom:14px;color:var(--text-muted);">
            <div><b>Model:</b> ${esc(p.model_name)}</div>
            <div><b>Confidence:</b> ${(p.confidence * 100).toFixed(1)}%</div>
            <div><b>Abstention Decision (τ=0.50):</b> ${p.is_abstained ? '⚠️ Max probability below threshold → Abstained' : '✅ Confident Prediction'}</div>
            <div><b>Image Perceptual Hash (dHash):</b> <code>${esc(data.dhash)}</code></div>
          </div>

          <h5 style="font-size:12px;font-weight:700;margin-bottom:6px;color:var(--navy-900)">Class Probability Distribution</h5>
          <div style="display:flex;flex-direction:column;gap:8px;">
            ${Object.keys(probs).map(c => {
              const pct = (probs[c] * 100).toFixed(1);
              const color = SEVERITY_COLORS[c] || '#2563eb';
              return `
                <div>
                  <div style="display:flex;justify-content:space-between;font-size:11px;margin-bottom:2px;">
                    <span>${esc(c)}</span>
                    <b>${pct}%</b>
                  </div>
                  <div class="progress-track">
                    <div class="progress-fill" style="width:${pct}%;background:${color}"></div>
                  </div>
                </div>
              `;
            }).join('')}
          </div>
        </div>
      `;
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = 'Run Model Inference';
    }
  });
}

// AI Priority Engine Simulator
function setupPrioritySimulator() {
  const sevInput = document.getElementById('simSev');
  const urgInput = document.getElementById('simUrg');
  const impInput = document.getElementById('simImp');
  const accInput = document.getElementById('simAcc');

  const sevVal = document.getElementById('simSevVal');
  const urgVal = document.getElementById('simUrgVal');
  const impVal = document.getElementById('simImpVal');
  const accVal = document.getElementById('simAccVal');

  const scoreEl = document.getElementById('simTotalScore');
  const breakdownEl = document.getElementById('simBreakdownPills');

  function calculate() {
    if (!sevInput || !urgInput || !impInput || !accInput) return;
    const S = parseFloat(sevInput.value);
    const U = parseFloat(urgInput.value);
    const I = parseFloat(impInput.value);
    const A = parseFloat(accInput.value);

    if (sevVal) sevVal.textContent = S;
    if (urgVal) urgVal.textContent = U;
    if (impVal) impVal.textContent = I;
    if (accVal) accVal.textContent = A;

    const wS = 0.40 * S;
    const wU = 0.25 * U;
    const wI = 0.20 * I;
    const wA = 0.15 * A;
    const total = wS + wU + wI + wA;

    if (scoreEl) scoreEl.textContent = total.toFixed(1);

    if (breakdownEl) {
      breakdownEl.innerHTML = `
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:11px;text-align:center;">
          <div style="background:#fee2e2;color:#991b1b;padding:8px;border-radius:var(--radius-sm)">
            <div>Severity</div>
            <b>+${wS.toFixed(1)} pts</b>
          </div>
          <div style="background:#ffedd5;color:#9a3412;padding:8px;border-radius:var(--radius-sm)">
            <div>Urgency</div>
            <b>+${wU.toFixed(1)} pts</b>
          </div>
          <div style="background:#fef3c7;color:#92400e;padding:8px;border-radius:var(--radius-sm)">
            <div>Impact</div>
            <b>+${wI.toFixed(1)} pts</b>
          </div>
          <div style="background:#e0e7ff;color:#3730a3;padding:8px;border-radius:var(--radius-sm)">
            <div>Access</div>
            <b>+${wA.toFixed(1)} pts</b>
          </div>
        </div>
      `;
    }
  }

  [sevInput, urgInput, impInput, accInput].forEach(el => {
    if (el) el.addEventListener('input', calculate);
  });
  calculate();
}

// Multimodal Deduplication Center
function renderDuplicatesView() {
  const container = document.getElementById('duplicatesListContainer');
  if (!container) return;

  if (!state.duplicates.length) {
    container.innerHTML = `
      <div style="background:var(--bg-surface);border:1px dashed var(--border-strong);padding:30px;text-align:center;border-radius:var(--radius-md);color:var(--text-muted)">
        <p style="font-size:14px;font-weight:600">No candidate duplicates detected.</p>
        <p style="font-size:12px;">The multimodal algorithm (MiniLM text embedding + image dHash + GPS distance) monitors incoming submissions.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = state.duplicates.map(link => `
    <div style="background:#ffffff;border:1px solid var(--border-light);border-radius:var(--radius-md);padding:16px;margin-bottom:14px;box-shadow:var(--shadow-xs)">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
        <span style="font-size:13px;font-weight:700;color:var(--navy-900);">Candidate Pair: ${esc(link.incident_a)} ⇄ ${esc(link.incident_b)}</span>
        <span style="background:#fef3c7;color:#92400e;font-weight:700;padding:2px 8px;border-radius:var(--radius-full);font-size:11px;">
          Composite Score: ${link.composite_score}% (${esc(link.status)})
        </span>
      </div>

      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;font-size:12px;">
        <div style="background:var(--bg-subtle);padding:10px;border-radius:var(--radius-sm)">
          <div style="font-weight:700;color:var(--navy-900)">Report A (${esc(link.a_class || 'Severe')})</div>
          <p style="margin:4px 0;">${esc(link.a_desc || 'No description available')}</p>
        </div>
        <div style="background:var(--bg-subtle);padding:10px;border-radius:var(--radius-sm)">
          <div style="font-weight:700;color:var(--navy-900)">Report B (${esc(link.b_class || 'Severe')})</div>
          <p style="margin:4px 0;">${esc(link.b_desc || 'No description available')}</p>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;font-size:11px;margin-bottom:12px;background:#f8fafc;padding:8px 10px;border-radius:var(--radius-sm);border:1px solid var(--border-light)">
        <div><span>Text Sim (MiniLM):</span> <b>${link.text_sim}%</b></div>
        <div><span>Image Hash Sim:</span> <b>${link.image_sim}%</b></div>
        <div><span>GPS Proximity:</span> <b>${link.geo_dist_m.toFixed(0)}m</b></div>
        <div><span>Time Proximity:</span> <b>${link.time_sim}%</b></div>
      </div>

      <div style="display:flex;gap:8px;justify-content:flex-end">
        <button class="btn btn-sm btn-primary" onclick="handleDuplicateDecision('${esc(link.id)}', 'Confirmed')">✓ Confirm Duplicate</button>
        <button class="btn btn-sm btn-secondary" onclick="handleDuplicateDecision('${esc(link.id)}', 'Split')">✂ Split (Independent)</button>
      </div>
    </div>
  `).join('');
}

async function handleDuplicateDecision(linkId, decision) {
  try {
    const res = await fetch(`/api/duplicates/${linkId}/decision`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision: decision, actor: state.userRole })
    });
    if (!res.ok) throw new Error('Duplicate decision failed');
    showToast(`Duplicate link marked as ${decision}.`);
    await loadAllData();
  } catch (err) {
    showToast(err.message, 'error');
  }
}

// Chart.js Disaster Intelligence Analytics
function renderAnalyticsCharts() {
  if (typeof Chart === 'undefined') return;

  const destroyChart = key => {
    if (state.charts[key]) {
      state.charts[key].destroy();
      state.charts[key] = null;
    }
  };

  const s = state.stats;
  const incs = state.incidents;

  // 1. Severity Distribution Donut Chart
  const sevCtx = document.getElementById('chartSeverity');
  if (sevCtx) {
    destroyChart('severity');
    state.charts.severity = new Chart(sevCtx, {
      type: 'doughnut',
      data: {
        labels: ['Severe damage', 'Mild damage', 'Little-or-none', 'Abstained / Uncertain'],
        datasets: [{
          data: [s.severe || 0, s.mild || 0, s.little || 0, s.abstained || 0],
          backgroundColor: ['#dc2626', '#ea580c', '#16a34a', '#9333ea'],
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } }
        }
      }
    });
  }

  // 2. Hazard Category Distribution Bar Chart
  const catCtx = document.getElementById('chartCategories');
  if (catCtx) {
    destroyChart('categories');
    const byCat = s.by_category || {};
    state.charts.categories = new Chart(catCtx, {
      type: 'bar',
      data: {
        labels: Object.keys(byCat),
        datasets: [{
          label: 'Incidents Reported',
          data: Object.values(byCat),
          backgroundColor: '#2563eb',
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          y: { beginAtZero: true, ticks: { precision: 0 } }
        }
      }
    });
  }

  // 3. Verification Pipeline Bar Chart
  const pipeCtx = document.getElementById('chartPipeline');
  if (pipeCtx) {
    destroyChart('pipeline');
    state.charts.pipeline = new Chart(pipeCtx, {
      type: 'bar',
      data: {
        labels: ['Pending', 'Verified', 'Rejected', 'Assigned', 'Resolved'],
        datasets: [{
          label: 'Triage Stage',
          data: [s.pending || 0, s.verified || 0, s.rejected || 0, s.assigned || 0, s.resolved || 0],
          backgroundColor: ['#f59e0b', '#059669', '#ef4444', '#0284c7', '#16a34a'],
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
      }
    });
  }

  // 4. Priority Score Distribution Histogram
  const priCtx = document.getElementById('chartPriority');
  if (priCtx) {
    destroyChart('priority');
    let binLow = 0, binMed = 0, binHigh = 0, binCrit = 0;
    incs.forEach(i => {
      const p = i.priority_score;
      if (p >= 80) binCrit++;
      else if (p >= 60) binHigh++;
      else if (p >= 40) binMed++;
      else binLow++;
    });

    state.charts.priority = new Chart(priCtx, {
      type: 'bar',
      data: {
        labels: ['Low (<40)', 'Moderate (40-60)', 'High (60-80)', 'Critical (80-100)'],
        datasets: [{
          label: 'Incident Volume',
          data: [binLow, binMed, binHigh, binCrit],
          backgroundColor: ['#94a3b8', '#f59e0b', '#ea580c', '#dc2626'],
          borderRadius: 4
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
      }
    });
  }
}

// Admin & Audit View
async function renderAdminView() {
  const tableBody = document.getElementById('adminAuditTableBody');
  const healthEl = document.getElementById('adminHealthStatus');

  if (healthEl && state.systemHealth) {
    healthEl.innerHTML = `
      <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:10px;font-size:12px;">
        <div><b>System:</b> ${esc(state.systemHealth.app)} (${esc(state.systemHealth.status)})</div>
        <div><b>Vision Engine:</b> ${esc(state.systemHealth.vision_model)}</div>
        <div><b>NLP Engine:</b> ${esc(state.systemHealth.nlp_model)}</div>
        <div><b>Formula:</b> ${esc(state.systemHealth.priority_formula)}</div>
      </div>
    `;
  }

  try {
    const res = await fetch('/api/audit-logs');
    if (!res.ok) return;
    state.auditLogs = await res.json();

    if (tableBody) {
      if (!state.auditLogs.length) {
        tableBody.innerHTML = `<tr><td colspan="5" style="text-align:center;padding:20px;color:var(--text-muted)">No audit log records found.</td></tr>`;
        return;
      }
      tableBody.innerHTML = state.auditLogs.map(log => `
        <tr>
          <td>${formatDate(log.created_at)}</td>
          <td><span style="font-family:monospace;font-size:11px;">${esc(log.incident_id)}</span></td>
          <td><b>${esc(log.action)}</b></td>
          <td>${esc(log.previous_value)} → <b>${esc(log.new_value)}</b></td>
          <td><span class="role-tag" style="background:#e0f2fe;color:#0369a1">${esc(log.actor)}</span></td>
        </tr>
      `).join('');
    }
  } catch (err) {
    console.error('Audit log fetch error:', err);
  }
}

// Data Export Utilities (CSV & JSON)
function exportDataJSON() {
  const blob = new Blob([JSON.stringify(state.incidents, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `RescueIQ_Incidents_${new Date().toISOString().slice(0, 10)}.json`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Incidents exported to JSON successfully.');
}

function exportDataCSV() {
  if (!state.incidents.length) return showToast('No incidents to export', 'error');
  const headers = ['id', 'damage_class', 'priority_score', 'category', 'verification_status', 'incident_status', 'latitude', 'longitude', 'urgency_score', 'reporter', 'created_at'];
  const rows = state.incidents.map(i => headers.map(h => JSON.stringify(i[h] ?? '')).join(','));
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `RescueIQ_Incidents_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('Incidents exported to CSV successfully.');
}

// Scenario Simulation Seeder
async function seedRealisticScenario() {
  try {
    const res = await fetch('/api/demo-seed');
    const data = await res.json();
    showToast(data.message || 'Realistic scenario seeded successfully.');
    await loadAllData();
  } catch (err) {
    showToast('Failed to seed scenario: ' + err.message, 'error');
  }
}

// Role Simulation Switcher
function setRole(roleName) {
  state.userRole = roleName;
  const roleEl = document.getElementById('currentRoleLabel');
  if (roleEl) roleEl.textContent = roleName;
  showToast(`Operating mode switched to: ${roleName}`);
}

// Filter listeners setup
function setupFilterListeners() {
  const searchInput = document.getElementById('queueSearchInput');
  if (searchInput) {
    searchInput.addEventListener('input', e => {
      state.filters.search = e.target.value.trim();
      renderIncidentsQueue();
    });
  }

  const statusSel = document.getElementById('filterStatus');
  if (statusSel) {
    statusSel.addEventListener('change', e => {
      state.filters.verificationStatus = e.target.value;
      renderIncidentsQueue();
    });
  }

  const catSel = document.getElementById('filterCategory');
  if (catSel) {
    catSel.addEventListener('change', e => {
      state.filters.category = e.target.value;
      renderIncidentsQueue();
    });
  }

  const sevSel = document.getElementById('filterSeverity');
  if (sevSel) {
    sevSel.addEventListener('change', e => {
      state.filters.damageClass = e.target.value;
      renderIncidentsQueue();
    });
  }

  const sortSel = document.getElementById('sortIncidents');
  if (sortSel) {
    sortSel.addEventListener('change', e => {
      state.filters.sortBy = e.target.value;
      renderIncidentsQueue();
    });
  }
}

// Initialization on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  setupRouting();
  initMaps();
  setupCitizenForm();
  setupFilterListeners();
  setupLiveVisionPlayground();
  setupPrioritySimulator();
  setupDialogBehaviors();
  loadAllData();

  // Auto-refresh data every 30 seconds
  setInterval(() => {
    loadAllData();
  }, 30000);
});
