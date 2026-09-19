/**
 * model_runner.js - Python Script Execution, Prediction Visualizer & Hackathon Submissions
 * Conforms to NebulaX Hackathon PS3 execution and packaging specifications.
 */

const ModelRunner = {
  availableModels: [],
  activeRunningSubsystem: null,

  async init() {
    this.bindEvents();
    await this.refreshModels();
    await this.validateSubmission(true);
  },

  bindEvents() {
    const uploadInput = document.getElementById("scriptUploadInput");
    if (uploadInput) {
      uploadInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.uploadCustomScript(e.target.files[0]);
        }
      });
    }

    // Bind card drag-and-drop zones for test datasets
    ["door", "acv", "rail", "shm"].forEach(sub => {
      const dropzone = document.getElementById(`dropzone-test-${sub}`);
      const fileInput = document.getElementById(`file-input-test-${sub}`);
      if (dropzone && fileInput) {
        dropzone.addEventListener("click", () => fileInput.click());
        fileInput.addEventListener("change", (e) => {
          if (e.target.files && e.target.files.length > 0) {
            this.uploadCardFile(sub, e.target.files[0]);
          }
        });

        dropzone.addEventListener("dragover", (e) => {
          e.preventDefault();
          dropzone.style.borderColor = "var(--accent-cyan)";
          dropzone.style.background = "rgba(0, 240, 255, 0.12)";
        });

        dropzone.addEventListener("dragleave", () => {
          dropzone.style.borderColor = "rgba(0, 240, 255, 0.4)";
          dropzone.style.background = "rgba(0, 240, 255, 0.03)";
        });

        dropzone.addEventListener("drop", (e) => {
          e.preventDefault();
          dropzone.style.borderColor = "rgba(0, 240, 255, 0.4)";
          dropzone.style.background = "rgba(0, 240, 255, 0.03)";
          if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            this.uploadCardFile(sub, e.dataTransfer.files[0]);
          }
        });
      }
    });
  },

  async refreshModels() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      if (data.status === "success") {
        this.availableModels = data.models;
        this.updateModelDropdowns();
        this.renderModelListTable();
      }
    } catch (err) {
      console.error("Failed to load models:", err);
    }
  },

  updateModelDropdowns() {
    const selects = document.querySelectorAll(".model-select-dropdown");
    selects.forEach(select => {
      const targetSub = select.getAttribute("data-subsystem");
      const currentVal = select.value;

      let relevant = this.availableModels.filter(m => m.subsystem === targetSub || m.subsystem === "custom");
      if (relevant.length === 0) relevant = this.availableModels;

      select.innerHTML = relevant.map(m => `
        <option value="${m.path}" ${m.path === currentVal || m.name.includes(targetSub) ? "selected" : ""}>
          ${m.name} [${m.type === "standin_baseline" ? "STAND-IN" : "CUSTOM"}]
        </option>
      `).join("");
    });
  },

  renderModelListTable() {
    const tbody = document.getElementById("modelsTableBody");
    if (!tbody) return;

    if (this.availableModels.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 15px;">No Python models available.</td></tr>`;
      return;
    }

    tbody.innerHTML = this.availableModels.map(m => `
      <tr>
        <td style="font-weight: 700; color: #fff;">🐍 ${m.name}</td>
        <td><span class="brand-tag">${m.subsystem.toUpperCase()}</span></td>
        <td>
          <span style="font-size: 0.75rem; color: ${m.type === "standin_baseline" ? "var(--accent-amber)" : "var(--accent-cyan)"}">
            ${m.type === "standin_baseline" ? "⚡ Stand-in Baseline" : "🚀 User Uploaded"}
          </span>
        </td>
        <td style="font-family: var(--font-mono);">${m.size_kb} KB</td>
        <td>
          <button class="hud-btn hud-btn-outline" style="padding: 4px 8px; font-size: 0.7rem;" onclick="ModelRunner.inspectCode('${m.path}')">
            📜 Inspect Code
          </button>
        </td>
      </tr>
    `).join("");
  },

  async uploadCardFile(subsystem, file) {
    const dropzone = document.getElementById(`dropzone-test-${subsystem}`);
    if (dropzone) {
      dropzone.innerHTML = `<span style="color: var(--accent-cyan);"><span class="status-pulse-live"></span> Uploading ${file.name}...</span>`;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/files/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        if (dropzone) {
          dropzone.innerHTML = `<span style="color: var(--accent-emerald);">✔ Selected: <strong>${file.name}</strong> (${data.size_kb} KB)</span>`;
        }
        await FileVault.refreshFiles();
        // Automatically select this file in the subsystem dropdown
        const select = document.getElementById(`file-select-${subsystem}`);
        if (select) {
          select.value = file.name;
        }
      } else {
        if (dropzone) dropzone.innerHTML = `<span style="color: var(--accent-red);">✖ Failed: ${data.detail}</span>`;
      }
    } catch (e) {
      if (dropzone) dropzone.innerHTML = `<span style="color: var(--accent-red);">✖ Upload error: ${e}</span>`;
    }
  },

  async uploadCustomScript(file) {
    if (!file.name.endsWith(".py")) {
      alert("Only Python .py script files can be uploaded.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const statusEl = document.getElementById("scriptUploadStatus");
    if (statusEl) {
      statusEl.innerHTML = `<span style="color: var(--accent-cyan);"><span class="status-pulse-live"></span> Registering model '${file.name}'...</span>`;
    }

    try {
      const res = await fetch("/api/models/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        if (statusEl) {
          statusEl.innerHTML = `<span style="color: var(--accent-emerald);">✔ Model '${file.name}' registered into Depot Model Registry!</span>`;
        }
        await this.refreshModels();
      } else {
        if (statusEl) {
          statusEl.innerHTML = `<span style="color: var(--accent-red);">✖ Failed: ${data.detail}</span>`;
        }
      }
    } catch (err) {
      if (statusEl) {
        statusEl.innerHTML = `<span style="color: var(--accent-red);">✖ Error: ${err}</span>`;
      }
    }
  },

  async runSubsystemModel(subsystem) {
    const fileSelect = document.getElementById(`file-select-${subsystem}`);
    const terminal = document.getElementById(`terminal-${subsystem}`);
    const runBtn = document.getElementById(`btn-run-${subsystem}`);
    const viewBtn = document.getElementById(`btn-view-${subsystem}`);
    const downloadBtn = document.getElementById(`btn-download-${subsystem}`);

    if (!fileSelect || !terminal) return false;

    let selectedOptions = [];
    if (fileSelect.multiple) {
      selectedOptions = Array.from(fileSelect.selectedOptions).map(opt => opt.value);
    } else {
      if (fileSelect.value) selectedOptions = [fileSelect.value];
    }

    if (selectedOptions.length === 0) {
      terminal.textContent = "[ERROR] Please select an input dataset (.xlsx/.csv).";
      return false;
    }

    runBtn.disabled = true;
    runBtn.innerHTML = `<span class="status-pulse-live"></span> ANALYSING...`;
    terminal.style.display = 'block';
    terminal.textContent = `[INITIALIZING]\n> Running ML Pipeline natively\n> Input Dataset(s): ${selectedOptions.join(", ")}\n> Subsystem: ${subsystem.toUpperCase()}\n`;

    try {
      const res = await fetch("/api/models/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          input_filenames: selectedOptions,
          subsystem: subsystem
        })
      });

      const data = await res.json();
      runBtn.disabled = false;
      runBtn.innerHTML = `⚡ RUN ANALYSIS`;

      if (data.status === "success") {
        terminal.textContent = `[STATUS: COMPLETED IN ${data.duration_sec}s]\nAnalysis completed successfully.\n[PREDICTION CSV WRITTEN] -> ${data.output_file}\n\n`;
        
        if (data.adapter_results && window.ResultRenderer) {
            window.ResultRenderer.render(subsystem, data, `results-${subsystem}`);
        }

        if (viewBtn) viewBtn.style.display = "inline-flex";
        if (downloadBtn) downloadBtn.style.display = "inline-flex";

        await this.validateSubmission(true);
        return true;
      } else {
        terminal.textContent = `[FAILED WITH ERROR]\n${data.message || data.stderr || "Unknown error."}`;
        return false;
      }
    } catch (err) {
      runBtn.disabled = false;
      runBtn.innerHTML = `⚡ RUN ANALYSIS`;
      terminal.textContent = `[EXECUTION EXCEPTION]: ${err}`;
      return false;
    }
  },

  async runAllSubsystems() {
    const masterBtn = document.getElementById("btnRunAllModels");
    const statusEl = document.getElementById("runAllStatusIndicator");
    if (masterBtn) {
      masterBtn.disabled = true;
      masterBtn.innerHTML = `<span class="status-pulse-live"></span> RUNNING ALL 4 MODELS...`;
    }
    if (statusEl) statusEl.textContent = "Executing models in sequence...";

    const subs = ["door", "acv", "rail", "shm"];
    let successCount = 0;

    for (let i = 0; i < subs.length; i++) {
      const sub = subs[i];
      if (statusEl) statusEl.textContent = `Running ${i + 1}/4: ${sub.toUpperCase()} model...`;
      const ok = await this.runSubsystemModel(sub);
      if (ok) successCount++;
    }

    if (masterBtn) {
      masterBtn.disabled = false;
      masterBtn.innerHTML = `⚡ RUN ALL 4 MODELS SEQUENTIALLY`;
    }
    if (statusEl) {
      statusEl.innerHTML = `<span style="color: var(--accent-emerald);">✔ Batch finished: ${successCount}/4 models completed successfully!</span>`;
    }

    await this.validateSubmission(false);
  },

  async viewPredictionOnScreen(subsystem) {
    const modalBg = document.getElementById("filePreviewModalBg");
    const modalTitle = document.getElementById("previewModalTitle");
    const gridContainer = document.getElementById("previewGridContainer");
    const statsContainer = document.getElementById("previewStatsContainer");
    const sheetTabs = document.getElementById("previewSheetTabs");

    if (!modalBg) return;

    modalBg.style.display = "flex";
    modalTitle.innerHTML = `Prediction Visualizer: <span style="color: var(--accent-cyan);">${subsystem.toUpperCase()} Predictions</span>`;
    sheetTabs.innerHTML = "";
    gridContainer.innerHTML = `<div style="padding: 30px; text-align: center; color: var(--accent-cyan);"><span class="status-pulse-live"></span> Loading prediction output...</div>`;

    try {
      const res = await fetch(`/api/predictions/preview/${subsystem}`);
      const data = await res.json();
      if (!res.ok) {
        gridContainer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">${data.detail || "Error loading prediction"}</div>`;
        return;
      }

      const v = data.validation;
      const isValid = v && v.valid;

      statsContainer.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(0,0,0,0.3); padding: 10px 14px; border-radius: 6px; font-size: 0.78rem;">
          <div style="display: flex; gap: 18px;">
            <div>Target File: <strong style="color: #fff; font-family: var(--font-mono);">${data.filename}</strong></div>
            <div>Total Output Rows: <strong style="color: var(--accent-cyan); font-family: var(--font-mono);">${data.row_count}</strong></div>
            <div>Columns: <strong style="color: var(--accent-amber);">${data.columns.join(", ")}</strong></div>
          </div>
          <div style="display: flex; gap: 10px; align-items: center;">
            <span style="padding: 3px 8px; border-radius: 4px; font-weight: 700; ${isValid ? "background: rgba(0,255,157,0.15); color: #00ff9d; border: 1px solid #00ff9d;" : "background: rgba(255,0,85,0.15); color: #ff3366; border: 1px solid #ff0055;"}">
              ${isValid ? "✔ PS3 SCHEMA VALID" : "✖ INVALID SCHEMA"}
            </span>
            <button class="hud-btn" style="padding: 4px 10px; font-size: 0.72rem;" onclick="ModelRunner.downloadPrediction('${subsystem}')">
              ⬇️ Download CSV
            </button>
          </div>
        </div>
      `;

      let tableHtml = `
        <table class="hud-table" style="font-size: 0.75rem;">
          <thead>
            <tr>
              <th style="width: 40px;">#</th>
              ${data.columns.map(c => `<th>${c}</th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${data.rows.map((row, idx) => `
              <tr>
                <td style="color: var(--text-muted); font-family: var(--font-mono);">${idx + 1}</td>
                ${row.map(val => `<td style="font-family: var(--font-mono);">${val !== null ? val : ""}</td>`).join("")}
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      gridContainer.innerHTML = tableHtml;

    } catch (e) {
      gridContainer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Error parsing predictions: ${e}</div>`;
    }
  },

  downloadPrediction(subsystem) {
    const outNames = {
      door: "door_predictions.csv",
      acv: "acv_predictions.csv",
      rail: "rail_predictions.csv",
      shm: "shm_predictions.csv"
    };
    const filename = outNames[subsystem.lower ? subsystem.lower() : subsystem] || `${subsystem}_predictions.csv`;
    window.location.href = `/api/predictions/download/${filename}`;
  },

  async validateSubmission(silent = false) {
    try {
      const res = await fetch("/api/predictions/validate");
      const data = await res.json();
      if (res.ok && data.status === "success") {
        const subs = data.subsystems;
        ["door", "acv", "rail", "shm"].forEach(sub => {
          const chip = document.getElementById(`submission-chip-${sub}`);
          const report = subs[sub];
          if (chip && report) {
            chip.className = "submission-status-chip";
            if (report.valid) {
              chip.classList.add("chip-ready");
              chip.innerHTML = `<span>✔</span> <strong>${report.filename}</strong> (${report.row_count} rows)`;
            } else if (report.status === "missing") {
              chip.classList.add("chip-missing");
              chip.innerHTML = `<span>○</span> <strong>${report.filename}</strong> (Pending Run)`;
            } else {
              chip.classList.add("chip-invalid");
              chip.innerHTML = `<span>✖</span> <strong>${report.filename}</strong> (Invalid Schema)`;
            }
          }
        });

        const overallIndicator = document.getElementById("submissionOverallStatus");
        if (overallIndicator) {
          if (data.all_valid) {
            overallIndicator.innerHTML = `<span style="color: var(--accent-emerald); font-weight: 700;">✔ ALL PREDICTIONS VALIDATED (READY TO ZIP)</span>`;
          } else if (data.has_predictions) {
            overallIndicator.innerHTML = `<span style="color: var(--accent-amber); font-weight: 700;">⚡ PARTIAL SUBMISSION READY (ONLY ATTEMPTED PREDICTIONS WILL BE ZIPPED)</span>`;
          } else {
            overallIndicator.innerHTML = `<span style="color: var(--text-muted);">No predictions generated yet. Run models to create submission CSVs.</span>`;
          }
        }

        if (!silent) {
          alert(data.all_valid ? "✔ All prediction CSVs match the official PS3 submission format perfectly!" : "Some predictions are pending or have schema discrepancies. Check the submission chips above.");
        }
      }
    } catch (e) {
      console.warn("Could not validate submission:", e);
    }
  },

  async inspectCode(scriptPath) {
    const modalBg = document.getElementById("filePreviewModalBg");
    const modalTitle = document.getElementById("previewModalTitle");
    const gridContainer = document.getElementById("previewGridContainer");
    const statsContainer = document.getElementById("previewStatsContainer");
    const sheetTabs = document.getElementById("previewSheetTabs");

    if (!modalBg) return;

    modalBg.style.display = "flex";
    modalTitle.innerHTML = `Python Script Inspector: <span style="color: var(--accent-cyan);">${scriptPath.split("/").pop().split("\\").pop()}</span>`;
    sheetTabs.innerHTML = "";
    statsContainer.innerHTML = `<div style="font-size: 0.75rem; color: var(--text-muted);">Python 3.12 Architecture - Stand-in Baseline / Custom Program</div>`;
    gridContainer.innerHTML = `<div style="padding: 20px; color: var(--accent-cyan);"><span class="status-pulse-live"></span> Loading source code...</div>`;

    try {
      const res = await fetch(`/api/models/code?script=${encodeURIComponent(scriptPath)}`);
      const data = await res.json();
      if (res.ok) {
        gridContainer.innerHTML = `
          <pre style="background: #05080f; color: #a5f3fc; padding: 18px; border-radius: 8px; font-family: var(--font-mono); font-size: 0.8rem; overflow: auto; height: 100%; line-height: 1.5; border: 1px solid rgba(0,240,255,0.2);"><code>${this.escapeHtml(data.code)}</code></pre>
        `;
      } else {
        gridContainer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">${data.detail}</div>`;
      }
    } catch (err) {
      gridContainer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Error: ${err}</div>`;
    }
  },

  async packageAndDownloadZip() {
    const statusEl = document.getElementById("zipStatusIndicator");
    if (statusEl) statusEl.innerHTML = `<span class="status-pulse-live"></span> Packaging predictions.zip...`;

    try {
      const res = await fetch("/api/predictions/package", { method: "POST" });
      const data = await res.json();
      if (res.ok && data.status === "success") {
        if (statusEl) {
          statusEl.innerHTML = `✔ ${data.zip_filename} ready (${data.size_kb} KB, ${data.total_files} files: ${data.files_included.join(", ")})`;
        }
        window.location.href = `/api/predictions/download/predictions.zip`;
      } else {
        if (statusEl) statusEl.innerHTML = `✖ Failed: ${data.message || "Unknown error"}`;
      }
    } catch (err) {
      if (statusEl) statusEl.innerHTML = `✖ Error: ${err}`;
    }
  },

  escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
};
