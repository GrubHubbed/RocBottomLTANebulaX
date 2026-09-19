/**
 * app.js - Master Dashboard Controller for LTA Smart Depot CdM System
 * Handles customizable multi-solution layout, tab routing, schemas rendering, and telemetry loop.
 */

const DashboardApp = {
  currentTab: "dashboard",
  currentLayout: "solo", // 'quad' | 'dual' | 'triple' | 'solo'
  activeSubsystems: {
    door: true,
    acv: false,
    rail: false,
    shm: false
  },
  schemas: {},

  async init() {
    this.restoreUserPreferences();
    this.bindNavigation();
    this.bindLayoutControls();
    this.startHUDClock();
    
    await this.loadSchemas();
    await FileVault.init();
    await ModelRunner.init();
    await DepotTriage.init();

    this.applyLayout();
  },

  restoreUserPreferences() {
    try {
      const savedLayout = localStorage.getItem("lta_dashboard_layout");
      if (savedLayout) this.currentLayout = savedLayout;

      const savedSubs = localStorage.getItem("lta_active_subsystems");
      if (savedSubs) this.activeSubsystems = JSON.parse(savedSubs);
    } catch (e) {
      console.warn("Could not restore preferences from localStorage:", e);
    }
  },

  saveUserPreferences() {
    try {
      localStorage.setItem("lta_dashboard_layout", this.currentLayout);
      localStorage.setItem("lta_active_subsystems", JSON.stringify(this.activeSubsystems));
    } catch (e) {
      console.warn("Could not save preferences to localStorage:", e);
    }
  },

  bindNavigation() {
    const tabBtns = document.querySelectorAll(".hud-tab-btn");
    tabBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        const targetTab = btn.getAttribute("data-tab");
        this.switchTab(targetTab);
      });
    });
  },

  switchTab(tabName) {
    this.currentTab = tabName;
    document.querySelectorAll(".hud-tab-btn").forEach(b => {
      b.classList.toggle("active", b.getAttribute("data-tab") === tabName);
    });

    document.querySelectorAll(".tab-content-view").forEach(view => {
      view.style.display = view.id === `tab-view-${tabName}` ? "block" : "none";
    });

    if (tabName === "triage") {
      DepotTriage.refreshTriage();
    } else if (tabName === "vault") {
      FileVault.refreshFiles();
    } else if (tabName === "models") {
      ModelRunner.refreshModels();
    }
  },

  bindLayoutControls() {
    // Layout presets (Quad, Dual, Triple, Solo)
    document.querySelectorAll(".layout-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const preset = btn.getAttribute("data-preset");
        this.setLayoutPreset(preset);
      });
    });

    // Subsystem toggle pills (Door, ACV, Rail, SHM)
    document.querySelectorAll(".filter-toggle-pill").forEach(pill => {
      const checkbox = pill.querySelector("input[type='checkbox']");
      if (checkbox) {
        const sub = checkbox.getAttribute("data-sub");
        checkbox.checked = !!this.activeSubsystems[sub];
        pill.classList.toggle("checked", checkbox.checked);

        pill.addEventListener("click", (e) => {
          e.preventDefault();
          
          // Force exclusive selection (Solo mode)
          Object.keys(this.activeSubsystems).forEach(k => {
            this.activeSubsystems[k] = (k === sub);
          });
          
          // Update all pills visually
          document.querySelectorAll(".filter-toggle-pill").forEach(p => {
            const cb = p.querySelector("input[type='checkbox']");
            if (cb) {
              const s = cb.getAttribute("data-sub");
              cb.checked = this.activeSubsystems[s];
              p.classList.toggle("checked", cb.checked);
            }
          });
          
          this.currentLayout = "solo";
          this.applyLayout();
          this.saveUserPreferences();
        });
      }
    });
  },

  setLayoutPreset(preset) {
    this.currentLayout = preset;
    
    // Automatically configure active subsystems according to preset
    if (preset === "quad") {
      this.activeSubsystems = { door: true, acv: false, rail: false, shm: false };
    } else if (preset === "dual") {
      this.activeSubsystems = { door: true, acv: false, rail: false, shm: false };
    } else if (preset === "triple") {
      this.activeSubsystems = { door: true, acv: false, rail: false, shm: false };
    } else if (preset === "solo") {
      this.activeSubsystems = { door: true, acv: false, rail: false, shm: false };
    }

    // Update pill checkboxes visually
    document.querySelectorAll(".filter-toggle-pill").forEach(pill => {
      const cb = pill.querySelector("input[type='checkbox']");
      if (cb) {
        const sub = cb.getAttribute("data-sub");
        cb.checked = !!this.activeSubsystems[sub];
        pill.classList.toggle("checked", cb.checked);
      }
    });

    this.applyLayout();
    this.saveUserPreferences();
  },

  applyLayout() {
    const viewport = document.getElementById("solutionsViewport");
    if (!viewport) return;

    // Count how many subsystems are enabled
    const enabledSubs = Object.keys(this.activeSubsystems).filter(k => this.activeSubsystems[k]);
    const count = enabledSubs.length;

    // Remove all grid layout classes
    viewport.classList.remove("grid-quad", "grid-dual", "grid-triple", "grid-solo");

    if (count === 4) {
      viewport.classList.add("grid-quad");
    } else if (count === 3) {
      viewport.classList.add("grid-triple");
    } else if (count === 2) {
      viewport.classList.add("grid-dual");
    } else {
      viewport.classList.add("grid-solo");
    }

    // Toggle panel visibility
    ["door", "acv", "rail", "shm"].forEach(sub => {
      const panel = document.getElementById(`panel-${sub}`);
      if (panel) {
        panel.style.display = this.activeSubsystems[sub] ? "flex" : "none";
      }
    });

    // Update active preset button highlight
    document.querySelectorAll(".layout-btn").forEach(btn => {
      btn.classList.toggle("active", btn.getAttribute("data-preset") === this.currentLayout);
    });

    const activeCountEl = document.getElementById("activeSubsystemCount");
    if (activeCountEl) activeCountEl.textContent = `${count} of 4 Solutions Displayed`;
  },

  async loadSchemas() {
    try {
      const res = await fetch("/api/schemas");
      const data = await res.json();
      if (data.status === "success") {
        this.schemas = data.subsystems;
        this.renderHeadersTables();
      }
    } catch (e) {
      console.error("Failed to load schemas:", e);
    }
  },

  renderHeadersTables() {
    // Populate the official headers table in each panel
    ["door", "acv", "rail", "shm"].forEach(sub => {
      const container = document.getElementById(`headers-table-${sub}`);
      if (!container || !this.schemas[sub]) return;

      const subData = this.schemas[sub];
      const rowsHtml = subData.headers.map(h => `
        <tr>
          <td style="font-family: var(--font-mono); font-weight: 700; color: var(--accent-cyan);">${h.parameter}</td>
          <td><span style="font-size: 0.68rem; color: var(--accent-amber);">${h.unit}</span></td>
          <td style="font-size: 0.68rem; color: var(--text-muted);">${h.description}</td>
        </tr>
      `).join("");

      container.innerHTML = `
        <table class="hud-table">
          <thead>
            <tr>
              <th>Parameter</th>
              <th>Unit</th>
              <th>Operational Function</th>
            </tr>
          </thead>
          <tbody>
            ${rowsHtml}
          </tbody>
        </table>
      `;
    });
  },

  startHUDClock() {
    const clockEl = document.getElementById("hudClockTime");
    const updateTime = () => {
      const now = new Date();
      if (clockEl) {
        clockEl.textContent = now.toLocaleTimeString("en-SG", {
          timeZone: "Asia/Singapore",
          hour12: false,
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit"
        }) + " SGT";
      }
    };
    updateTime();
    setInterval(updateTime, 1000);
  }
};

window.addEventListener("DOMContentLoaded", () => {
  DashboardApp.init();
});
