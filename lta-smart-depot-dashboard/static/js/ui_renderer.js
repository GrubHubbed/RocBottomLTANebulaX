const ResultRenderer = {
  render(subsystem, data, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.style.display = 'block';
    
    if (subsystem === 'door') {
      this.renderDoor(data.adapter_results, container);
    } else if (subsystem === 'acv') {
      this.renderACV(data.adapter_results, container);
    } else if (subsystem === 'rail') {
      this.renderRail(data.adapter_results, container);
    } else if (subsystem === 'shm') {
      this.renderSHM(data.adapter_results, container);
    } else {
      container.innerHTML = `<pre>${JSON.stringify(data.adapter_results, null, 2)}</pre>`;
    }
  },

  renderDoor(results, container) {
    const events = results.events || [];
    const trend = results.trend || {};
    const summary = results.summary || {};

    let html = `
      <div style="margin-bottom: 15px;">
        <h4 style="color: var(--accent-cyan); font-size: 1rem; margin-bottom: 5px;">Door Maintenance Assessment</h4>
        <div style="display: flex; gap: 15px; font-size: 0.8rem; color: var(--text-muted);">
          <div>Analyzed: <strong style="color: #fff;">${summary.n_segments || 0} cycles</strong></div>
          <div>Abnormal: <strong style="color: var(--accent-amber);">${summary.n_abnormal || 0}</strong></div>
        </div>
      </div>
    `;

    if (events.length > 0) {
      events.forEach(ev => {
        const sev = parseFloat(ev.severity || 0);
        let chip = "Low";
        let color = "#888";
        let action = "Clean and lubricate the roller tracks.";
        let bg = "rgba(136, 136, 136, 0.1)";

        if (sev > 4.0) {
          chip = "Critical";
          color = "#ff3366";
          action = "Check the power supply and current sensor for short circuits.";
          bg = "rgba(255, 51, 102, 0.15)";
        } else if (sev >= 2.5) {
          chip = "High";
          color = "#ff9900";
          action = "Inspect the actuator and guide rail for physical obstruction.";
          bg = "rgba(255, 153, 0, 0.15)";
        } else if (sev >= 1.5) {
          chip = "Medium";
          color = "#ffcc00";
          action = "Check the encoder and belt tension.";
          bg = "rgba(255, 204, 0, 0.15)";
        } else if (sev >= 1.0) {
          chip = "Low";
          color = "#a5f3fc";
          action = "Clean and lubricate the roller tracks.";
          bg = "rgba(165, 243, 252, 0.15)";
        }

        if (sev >= 1.0) {
          html += `
            <div style="background: rgba(0,0,0,0.3); border-left: 4px solid ${color}; padding: 12px; margin-bottom: 10px; border-radius: 4px;">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="font-weight: bold; color: #fff;">Event Duration: ${(ev.duration_s || 0).toFixed(1)}s</div>
                <div style="background: ${bg}; color: ${color}; padding: 4px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: bold; border: 1px solid ${color};">
                  ${chip} Severity
                </div>
              </div>
              <div style="font-size: 0.85rem; color: var(--text-muted);">
                <strong style="color: #fff;">Engineer View:</strong> ${action}
              </div>
            </div>
          `;
        }
      });
    } else {
      html += `<div style="color: var(--accent-emerald); font-weight: bold;">✔ No immediate mechanical faults detected.</div>`;
    }

    if (trend.cycles_to_threshold) {
      const days = (trend.cycles_to_threshold / 300).toFixed(1);
      html += `
        <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
          <div style="font-size: 0.85rem;">
            Estimated time to failure: <strong style="color: var(--accent-amber); font-size: 1rem;">~${days} days</strong> <span style="color: var(--text-muted);">(${Math.round(trend.cycles_to_threshold)} cycles remaining)</span>
          </div>
        </div>
      `;
    }

    container.innerHTML = html;
  },

  renderACV(results, container) {
    const batch = results.batch_results || [];
    let html = `
      <div style="margin-bottom: 15px;">
        <h4 style="color: var(--accent-cyan); font-size: 1rem; margin-bottom: 5px;">ACV Degassing Analysis</h4>
      </div>
    `;

    batch.forEach(res => {
      let carListHtml = "";
      if (res.ranked_cars) {
        // If ranked_cars is a pipe-separated string like "06|08|04...", split it
        const carsArray = typeof res.ranked_cars === 'string' ? res.ranked_cars.split('|') : res.ranked_cars;
        
        if (carsArray.length > 0) {
            const topCar = carsArray[0];
            
            carListHtml += `
              <div style="margin-bottom: 10px;">
                <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 4px;">Most Likely Affected Car:</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: #ff3366; background: rgba(255,51,102,0.1); display: inline-block; padding: 6px 16px; border-radius: 6px; border: 1px solid #ff3366;">
                  CAR ${topCar}
                </div>
              </div>
              
              <div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 8px;">Ranked list:</div>
              <div style="display: flex; gap: 6px; flex-wrap: wrap;">
            `;
            
            carsArray.forEach((car, i) => {
              const isTop = i === 0;
              const bg = isTop ? 'rgba(255,51,102,0.2)' : 'rgba(255,255,255,0.05)';
              const col = isTop ? '#ff3366' : '#ccc';
              carListHtml += `<div style="background: ${bg}; color: ${col}; padding: 3px 8px; border-radius: 4px; font-family: var(--font-mono); font-size: 0.75rem;">${i+1}. Car ${car}</div>`;
            });
            carListHtml += `</div>`;
        }
      } else {
        carListHtml = `<div style="color: var(--accent-emerald); font-weight: bold;">✔ Normal - No significant undercharge detected.</div>`;
      }

      html += `
        <div style="background: rgba(0,0,0,0.3); padding: 12px; margin-bottom: 12px; border-radius: 4px; border-left: 2px solid var(--accent-cyan);">
          <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #a5f3fc; margin-bottom: 10px;">📄 ${res.file_id}</div>
          ${carListHtml}
        </div>
      `;
    });

    container.innerHTML = html;
  },

  renderRail(results, container) {
    const batch = results.batch_results || [];
    const counts = results.counts || {};
    
    let html = `
      <div style="margin-bottom: 15px;">
        <h4 style="color: var(--accent-cyan); font-size: 1rem; margin-bottom: 5px;">Rail Corrugation Batch Summary</h4>
        <div style="display: flex; gap: 15px; font-size: 0.8rem;">
          <div style="color: #fff;">Total Files: <strong>${batch.length}</strong></div>
          <div style="color: var(--accent-emerald);">Normal: <strong>${counts['Normal'] || 0}</strong></div>
          <div style="color: #ff9900;">Side I Fault: <strong>${counts['Side I'] || 0}</strong></div>
          <div style="color: #ff3366;">Side II Fault: <strong>${counts['Side II'] || 0}</strong></div>
        </div>
      </div>
      <div style="max-height: 250px; overflow-y: auto;">
        <table class="hud-table" style="font-size: 0.75rem; margin-top: 10px;">
          <thead>
            <tr><th>File</th><th>Localization</th></tr>
          </thead>
          <tbody>
    `;

    batch.forEach(res => {
      let color = "var(--accent-emerald)";
      if (res.prediction === "Side I") color = "#ff9900";
      if (res.prediction === "Side II") color = "#ff3366";
      
      html += `
        <tr>
          <td style="font-family: var(--font-mono); color: #ccc;">${res.file_id}</td>
          <td style="color: ${color}; font-weight: bold;">${res.prediction}</td>
        </tr>
      `;
    });

    html += `</tbody></table></div>`;
    container.innerHTML = html;
  },

  renderSHM(results, container) {
    const batch = results.batch_results || [];
    
    let html = `
      <div style="margin-bottom: 15px;">
        <h4 style="color: var(--accent-cyan); font-size: 1rem; margin-bottom: 5px;">SHM Fatigue Damage Batch Overview</h4>
      </div>
    `;

    batch.forEach(res => {
      const pred = res.prediction.toFixed(4);
      const limit = (res.pct_of_fatigue_limit * 100).toFixed(1);
      const segments = res.segments_to_limit.toLocaleString();
      const interp = res.interpretation || {};
      
      let badgeColor = "var(--accent-emerald)";
      let badgeBg = "rgba(0, 255, 157, 0.1)";
      if (interp.status === "Critical") {
        badgeColor = "#ff3366";
        badgeBg = "rgba(255, 51, 102, 0.1)";
      } else if (interp.status === "Warning") {
        badgeColor = "#ff9900";
        badgeBg = "rgba(255, 153, 0, 0.1)";
      }
      
      html += `
        <div style="background: rgba(0,0,0,0.3); padding: 12px; margin-bottom: 12px; border-radius: 4px; border-left: 4px solid ${badgeColor};">
          <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
            <div style="font-family: var(--font-mono); font-size: 0.75rem; color: #a5f3fc;">📄 ${res.file_id}</div>
            <div style="background: ${badgeBg}; color: ${badgeColor}; padding: 3px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: bold;">${interp.status || "Unknown"}</div>
          </div>
          
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 10px;">
            <div>
              <div style="font-size: 0.7rem; color: var(--text-muted);">Cumulative Damage (D):</div>
              <div style="font-size: 1.1rem; color: #fff; font-weight: bold;">${pred}</div>
            </div>
            <div>
              <div style="font-size: 0.7rem; color: var(--text-muted);">Budget Used:</div>
              <div style="font-size: 1.1rem; color: ${badgeColor}; font-weight: bold;">${limit}%</div>
            </div>
          </div>
          
          <div style="font-size: 0.8rem; color: var(--text-muted); background: rgba(255,255,255,0.02); padding: 8px; border-radius: 4px;">
            ${interp.message || ""}
          </div>
        </div>
      `;
    });

    container.innerHTML = html;
  }
};

window.ResultRenderer = ResultRenderer;
