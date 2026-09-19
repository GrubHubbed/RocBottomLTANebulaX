/**
 * triage.js - LTA Smart Depot Active Train Pull-Out Triage & Work Order Generator
 */

const DepotTriage = {
  triageData: null,

  async init() {
    await this.refreshTriage();
  },

  async refreshTriage() {
    try {
      const res = await fetch("/api/triage");
      const json = await res.json();
      if (json.status === "success") {
        this.triageData = json.data;
        this.renderKPIs();
        this.renderTriageTable();
        this.renderBays();
      }
    } catch (err) {
      console.error("Failed to load triage data:", err);
    }
  },

  renderKPIs() {
    if (!this.triageData || !this.triageData.fleet_summary) return;
    const s = this.triageData.fleet_summary;

    const elTotal = document.getElementById("kpiTotalFleet");
    const elInService = document.getElementById("kpiInService");
    const elCritical = document.getElementById("kpiCriticalPullout");
    const elUrgent = document.getElementById("kpiUrgentPullout");
    const elBays = document.getElementById("kpiBayOccupancy");

    if (elTotal) elTotal.textContent = s.total_fleet;
    if (elInService) elInService.textContent = s.in_revenue_service;
    if (elCritical) elCritical.textContent = s.critical_pullout_required;
    if (elUrgent) elUrgent.textContent = s.urgent_offpeak_pullout;
    if (elBays) elBays.textContent = s.bay_occupancy;
  },

  renderTriageTable() {
    const tbody = document.getElementById("triageTableBody");
    if (!tbody || !this.triageData) return;

    tbody.innerHTML = this.triageData.triage_queue.map(train => {
      let priorityClass = "triage-priority-p4";
      if (train.triage_priority.includes("P1")) priorityClass = "triage-priority-p1";
      else if (train.triage_priority.includes("P2")) priorityClass = "triage-priority-p2";
      else if (train.triage_priority.includes("P3")) priorityClass = "triage-priority-p3";

      const thiColor = train.composite_thi < 50 ? "var(--accent-red)" : (train.composite_thi < 75 ? "var(--accent-amber)" : "var(--accent-emerald)");

      return `
        <tr>
          <td>
            <div style="font-weight: 800; font-size: 0.95rem; color: #fff;">${train.train_id}</div>
            <div style="font-size: 0.7rem; color: var(--text-muted);">${train.fleet_line} (${train.car_count} Cars)</div>
            <div style="font-size: 0.68rem; color: var(--accent-cyan);">${train.depot}</div>
          </td>
          <td>
            <div style="font-size: 1.25rem; font-weight: 900; font-family: var(--font-mono); color: ${thiColor};">
              ${train.composite_thi}%
            </div>
            <div style="font-size: 0.65rem; color: var(--text-muted);">Composite Health</div>
          </td>
          <td>
            <span class="${priorityClass}">${train.triage_priority}</span>
          </td>
          <td style="font-size: 0.72rem; line-height: 1.4; max-width: 380px;">
            <div style="color: #fff; font-weight: 600; margin-bottom: 4px;">${train.recommended_action}</div>
            <div style="color: var(--text-muted); font-size: 0.68rem;">
              <span style="color: ${train.door_score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">Door: ${train.door_score}%</span> | 
              <span style="color: ${train.acv_score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">ACV: ${train.acv_score}%</span> | 
              <span style="color: ${train.rail_score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">Rail: ${train.rail_score}%</span> | 
              <span style="color: ${train.shm_score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">SHM: ${train.shm_score}%</span>
            </div>
          </td>
          <td>
            <div style="font-family: var(--font-mono); font-size: 0.78rem; color: var(--accent-cyan); font-weight: 700;">
              ${train.allocated_bay}
            </div>
            <div style="font-size: 0.68rem; color: var(--text-muted);">${train.assigned_team}</div>
          </td>
          <td>
            <button class="hud-btn" style="padding: 6px 12px; font-size: 0.72rem;" onclick="DepotTriage.openWorkOrderModal('${train.train_id}')">
              📋 Work Order
            </button>
          </td>
        </tr>
      `;
    }).join("");
  },

  renderBays() {
    const container = document.getElementById("depotBaysGrid");
    if (!container || !this.triageData) return;

    container.innerHTML = this.triageData.bays.map(bay => {
      const isOccupied = bay.status === "OCCUPIED";
      const borderColor = isOccupied ? "var(--accent-red)" : "rgba(0, 255, 157, 0.4)";
      const bg = isOccupied ? "rgba(255, 0, 85, 0.08)" : "rgba(0, 255, 157, 0.05)";

      return `
        <div style="background: ${bg}; border: 1px solid ${borderColor}; border-radius: 8px; padding: 10px; display: flex; flex-direction: column; justify-content: space-between;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
            <strong style="font-size: 0.8rem; color: #fff;">${bay.bay_id}</strong>
            <span style="font-size: 0.65rem; padding: 2px 6px; border-radius: 4px; font-weight: 700; ${isOccupied ? "background: rgba(255,0,85,0.2); color: #ff3366;" : "background: rgba(0,255,157,0.2); color: #00ff9d;"}">
              ${bay.status}
            </span>
          </div>
          <div style="font-size: 0.72rem; color: var(--text-muted); margin-bottom: 6px;">${bay.type}</div>
          <div style="font-size: 0.7rem; font-family: var(--font-mono); color: #cbd5e1;">
            ${isOccupied ? `Occupant: <strong style="color: #fff;">${bay.train_id}</strong> (ETA ${bay.eta_hours}h)` : `<span style="color: var(--accent-emerald);">✔ Cleared for Ingress</span>`}
          </div>
        </div>
      `;
    }).join("");
  },

  async openWorkOrderModal(trainId) {
    const modalBg = document.getElementById("filePreviewModalBg");
    const modalTitle = document.getElementById("previewModalTitle");
    const gridContainer = document.getElementById("previewGridContainer");
    const statsContainer = document.getElementById("previewStatsContainer");
    const sheetTabs = document.getElementById("previewSheetTabs");

    if (!modalBg) return;

    modalBg.style.display = "flex";
    modalTitle.innerHTML = `LTA Depot Rolling Stock Maintenance Work Order: <span style="color: var(--accent-cyan);">${trainId}</span>`;
    sheetTabs.innerHTML = "";
    statsContainer.innerHTML = "";
    gridContainer.innerHTML = `<div style="padding: 20px; color: var(--accent-cyan);"><span class="status-pulse-live"></span> Compiling Telemetry Audit Trail & Directives...</div>`;

    try {
      const res = await fetch("/api/triage/work-order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ train_id: trainId })
      });
      const data = await res.json();
      if (res.ok && data.status === "success") {
        const wo = data.work_order;
        gridContainer.innerHTML = `
          <div style="background: #090e1a; border: 1px solid var(--border-hud); border-radius: 8px; padding: 24px; color: #f1f5f9; font-family: var(--font-hud);">
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid var(--accent-cyan); padding-bottom: 12px; margin-bottom: 16px;">
              <div>
                <h2 style="font-size: 1.15rem; color: #fff; letter-spacing: 1px;">SINGAPORE LAND TRANSPORT AUTHORITY (LTA)</h2>
                <div style="font-size: 0.75rem; color: var(--text-muted);">SMART DEPOT OPERATIONS CONTROL // ROLLING STOCK CORRECTIVE MAINTENANCE PULL-OUT SLIP</div>
              </div>
              <div style="text-align: right;">
                <div style="font-family: var(--font-mono); font-size: 0.95rem; font-weight: 800; color: var(--accent-cyan);">${wo.work_order_id}</div>
                <div style="font-size: 0.7rem; color: var(--text-muted);">${wo.timestamp}</div>
              </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; background: rgba(255,255,255,0.03); padding: 12px; border-radius: 6px; margin-bottom: 20px; font-size: 0.78rem;">
              <div>Trainset: <strong style="color: #fff;">${wo.train_id}</strong></div>
              <div>Operating Line: <strong style="color: #fff;">${wo.fleet_line}</strong></div>
              <div>Maintenance Base: <strong style="color: var(--accent-cyan);">${wo.depot}</strong></div>
              <div>Health Index (THI): <strong style="color: var(--accent-amber); font-family: var(--font-mono);">${wo.composite_thi}</strong></div>
            </div>

            <div style="margin-bottom: 18px;">
              <h3 style="font-size: 0.82rem; text-transform: uppercase; color: var(--accent-cyan); margin-bottom: 8px;">Subsystem Condition Monitoring Audit Trail</h3>
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                <div style="background: rgba(15,23,42,0.8); padding: 10px; border-radius: 6px; border-left: 3px solid ${wo.subsystem_diagnostics.door.score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">
                  <strong>Door System (${wo.subsystem_diagnostics.door.score}%)</strong>: ${wo.subsystem_diagnostics.door.finding}
                </div>
                <div style="background: rgba(15,23,42,0.8); padding: 10px; border-radius: 6px; border-left: 3px solid ${wo.subsystem_diagnostics.acv.score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">
                  <strong>ACV HVAC System (${wo.subsystem_diagnostics.acv.score}%)</strong>: ${wo.subsystem_diagnostics.acv.finding}
                </div>
                <div style="background: rgba(15,23,42,0.8); padding: 10px; border-radius: 6px; border-left: 3px solid ${wo.subsystem_diagnostics.rail.score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">
                  <strong>Rail Corrugation (${wo.subsystem_diagnostics.rail.score}%)</strong>: ${wo.subsystem_diagnostics.rail.finding}
                </div>
                <div style="background: rgba(15,23,42,0.8); padding: 10px; border-radius: 6px; border-left: 3px solid ${wo.subsystem_diagnostics.shm.score < 70 ? "var(--accent-red)" : "var(--accent-emerald)"};">
                  <strong>SHM Bogie Stress (${wo.subsystem_diagnostics.shm.score}%)</strong>: ${wo.subsystem_diagnostics.shm.finding}
                </div>
              </div>
            </div>

            <div style="background: rgba(255,170,0,0.08); border: 1px solid var(--accent-amber); padding: 14px; border-radius: 6px; margin-bottom: 20px;">
              <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--accent-amber); font-weight: 800; margin-bottom: 4px;">Operational Directive & Bay Routing</div>
              <div style="font-size: 0.85rem; color: #fff; font-weight: 600;">${wo.operational_directive}</div>
              <div style="margin-top: 8px; font-size: 0.75rem; color: var(--text-muted);">
                Destination: <strong style="color: var(--accent-cyan);">${wo.allocated_bay}</strong> | Assigned Unit: <strong style="color: #fff;">${wo.assigned_team}</strong>
              </div>
            </div>

            <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 14px; font-size: 0.72rem; color: var(--text-muted);">
              <div>LTA Interlock Safety Clearance: <strong style="color: var(--accent-emerald);">${wo.authorization.depot_master}</strong></div>
              <button class="hud-btn" onclick="window.print()">🖨 Print Official Work Order</button>
            </div>
          </div>
        `;
      }
    } catch (err) {
      gridContainer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Error generating work order: ${err}</div>`;
    }
  }
};
