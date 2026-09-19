/**
 * file_vault.js - Excel & CSV Storage Vault and Sheet Data Inspector
 */

const FileVault = {
  currentFiles: [],
  selectedFileForPreview: null,
  activePreviewSheet: null,

  async init() {
    this.bindEvents();
    await this.refreshFiles();
  },

  bindEvents() {
    const dropzone = document.getElementById("vaultDropzone");
    const fileInput = document.getElementById("vaultFileInput");

    if (dropzone && fileInput) {
      dropzone.addEventListener("click", () => fileInput.click());
      
      fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          this.uploadFile(e.target.files[0]);
        }
      });

      dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.classList.add("drag-hover");
      });

      dropzone.addEventListener("dragleave", () => {
        dropzone.classList.remove("drag-hover");
      });

      dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-hover");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          this.uploadFile(e.dataTransfer.files[0]);
        }
      });
    }
  },

  async refreshFiles() {
    try {
      const res = await fetch("/api/files");
      const data = await res.json();
      if (data.status === "success") {
        this.currentFiles = data.files;
        this.renderFilesTable();
        this.updateFileDropdowns();
      }
    } catch (err) {
      console.error("Failed to load stored files:", err);
    }
  },

  async uploadFile(file) {
    const formData = new FormData();
    formData.append("file", file);

    const statusEl = document.getElementById("vaultUploadStatus");
    if (statusEl) {
      statusEl.innerHTML = `<span style="color: var(--accent-cyan);"><span class="status-pulse-live"></span> Uploading '${file.name}' to Depot Vault...</span>`;
    }

    try {
      const res = await fetch("/api/files/upload", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        if (statusEl) {
          statusEl.innerHTML = `<span style="color: var(--accent-emerald);">✔ File '${file.name}' stored successfully (${data.size_kb} KB)</span>`;
        }
        await this.refreshFiles();
      } else {
        if (statusEl) {
          statusEl.innerHTML = `<span style="color: var(--accent-red);">✖ Upload failed: ${data.detail || "Error"}</span>`;
        }
      }
    } catch (err) {
      if (statusEl) {
        statusEl.innerHTML = `<span style="color: var(--accent-red);">✖ Network error during upload</span>`;
      }
    }
  },

  async deleteFile(filename) {
    if (!confirm(`Are you sure you want to delete '${filename}' from the depot storage vault?`)) {
      return;
    }
    try {
      const res = await fetch(`/api/files/${encodeURIComponent(filename)}`, {
        method: "DELETE"
      });
      if (res.ok) {
        await this.refreshFiles();
      }
    } catch (err) {
      alert("Error deleting file: " + err);
    }
  },

  renderFilesTable() {
    const tbody = document.getElementById("vaultTableBody");
    if (!tbody) return;

    if (this.currentFiles.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted); padding: 20px;">No Excel or CSV files stored in depot vault yet.</td></tr>`;
      return;
    }

    tbody.innerHTML = this.currentFiles.map(file => {
      const isExcel = file.extension === ".xlsx" || file.extension === ".xls";
      const icon = isExcel ? "📊" : "📄";
      const badgeClass = file.subsystem === "acv" ? "kpi-warning" : (file.subsystem === "door" ? "kpi-danger" : "kpi-success");
      
      return `
        <tr>
          <td style="font-weight: 600; color: #fff;">
            ${icon} ${file.filename}
          </td>
          <td>
            <span class="brand-tag" style="text-transform: uppercase;">${file.subsystem}</span>
          </td>
          <td style="font-family: var(--font-mono);">${file.size_kb} KB</td>
          <td style="color: var(--text-muted);">${file.modified}</td>
          <td>
            ${isExcel ? `<span style="color: var(--accent-cyan); font-size: 0.75rem;">${file.sheet_count} Sheet(s)</span>` : `<span style="color: var(--text-muted); font-size: 0.75rem;">Flat CSV</span>`}
          </td>
          <td>
            <div style="display: flex; gap: 6px;">
              <button class="hud-btn hud-btn-outline" style="padding: 4px 8px; font-size: 0.7rem;" onclick="FileVault.openPreviewModal('${file.filename}')">
                🔍 Preview Sheet
              </button>
              <button class="hud-btn hud-btn-danger" style="padding: 4px 8px; font-size: 0.7rem;" onclick="FileVault.deleteFile('${file.filename}')">
                🗑
              </button>
            </div>
          </td>
        </tr>
      `;
    }).join("");
  },

  updateFileDropdowns() {
    // Updates all input file dropdowns in each subsystem panel
    const selects = document.querySelectorAll(".file-select-dropdown");
    selects.forEach(select => {
      const currentVal = select.value;
      const targetSub = select.getAttribute("data-subsystem");
      
      let relevantFiles = this.currentFiles;
      if (targetSub) {
        relevantFiles = this.currentFiles.filter(f => f.subsystem === targetSub || f.subsystem === "general");
        if (relevantFiles.length === 0) relevantFiles = this.currentFiles;
      }

      select.innerHTML = relevantFiles.map(f => 
        `<option value="${f.filename}" ${f.filename === currentVal ? "selected" : ""}>${f.filename} (${f.size_kb} KB)</option>`
      ).join("");
    });
  },

  async openPreviewModal(filename, sheetName = null) {
    this.selectedFileForPreview = filename;
    this.activePreviewSheet = sheetName;
    
    const modalBg = document.getElementById("filePreviewModalBg");
    const modalTitle = document.getElementById("previewModalTitle");
    const sheetTabs = document.getElementById("previewSheetTabs");
    const gridContainer = document.getElementById("previewGridContainer");
    const statsContainer = document.getElementById("previewStatsContainer");

    if (!modalBg) return;

    modalBg.style.display = "flex";
    modalTitle.innerHTML = `Inspection: <span style="color: var(--accent-cyan);">${filename}</span>`;
    gridContainer.innerHTML = `<div style="padding: 30px; text-align: center; color: var(--accent-cyan);"><span class="status-pulse-live"></span> Loading Excel Worksheet Data Grid...</div>`;

    try {
      const url = `/api/files/preview?filename=${encodeURIComponent(filename)}${sheetName ? `&sheet=${encodeURIComponent(sheetName)}` : ""}`;
      const res = await fetch(url);
      const data = await res.json();

      if (!res.ok) {
        gridContainer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Failed: ${data.detail}</div>`;
        return;
      }

      // Render Sheet Tabs
      if (data.sheets && data.sheets.length > 0) {
        sheetTabs.innerHTML = data.sheets.map(s => `
          <button class="sheet-tab ${s === data.current_sheet ? "active" : ""}" onclick="FileVault.openPreviewModal('${filename}', '${s}')">
            📑 ${s}
          </button>
        `).join("");
      } else {
        sheetTabs.innerHTML = "";
      }

      // Render Statistics Deck
      statsContainer.innerHTML = `
        <div style="display: flex; gap: 20px; font-size: 0.75rem; color: var(--text-muted); background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 6px;">
          <div>Sheet: <strong style="color: #fff;">${data.current_sheet}</strong></div>
          <div>Total Rows: <strong style="color: var(--accent-cyan);">${data.total_rows}</strong></div>
          <div>Total Columns: <strong style="color: var(--accent-emerald);">${data.total_cols}</strong></div>
          <div>Previewing First: <strong style="color: var(--accent-amber);">${data.preview_row_count} rows</strong></div>
        </div>
      `;

      // Render Table Grid
      let tableHtml = `
        <table class="hud-table" style="font-size: 0.72rem;">
          <thead>
            <tr>
              <th style="width: 40px;">#</th>
              ${data.headers.map(h => `<th>${h}</th>`).join("")}
            </tr>
          </thead>
          <tbody>
            ${data.rows.map((row, idx) => `
              <tr>
                <td style="color: var(--text-muted); font-family: var(--font-mono);">${idx + 1}</td>
                ${row.map(cell => `<td>${cell !== null && cell !== undefined ? cell : ""}</td>`).join("")}
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      gridContainer.innerHTML = tableHtml;

    } catch (err) {
      gridContainer.innerHTML = `<div style="color: var(--accent-red); padding: 20px;">Error parsing sheet data: ${err}</div>`;
    }
  },

  closePreviewModal() {
    const modalBg = document.getElementById("filePreviewModalBg");
    if (modalBg) modalBg.style.display = "none";
  }
};
