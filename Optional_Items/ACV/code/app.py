"""ACV Thermal Integrity Depot Console.

Audience: LTA smart-depot duty controllers deciding which trainsets to withdraw
and which carriages fitters open first.

Every ranking shown here comes from acv_core.  Severity bands, colours, icons
and sort controls are presentation only: the submission CSV is built from the
same `ranked_cars` string the engine returned, so the console and the CLI cannot
disagree.
"""

from __future__ import annotations

import hashlib
import io
import os
import zipfile

import pandas as pd
import streamlit as st

import acv_core as core

st.set_page_config(
    page_title="ACV Thermal Integrity Depot Console",
    page_icon="=",
    layout="wide",
)

BUNDLED = [os.path.join("data", "Train"), os.path.join("data", "Test")]

SEVERITY_COLOUR = {
    "CRITICAL": "#ff4d6d",
    "HIGH": "#ff9f45",
    "ELEVATED": "#ffd93d",
    "NORMAL": "#3ddc97",
    # Deliberately off the red-amber-green leak scale: a carriage that sent no
    # telemetry is a different KIND of finding, not a quieter version of a leak,
    # and it must not read as "nothing to report".
    "NO DATA": "#a77bff",
}

PRIME_COLOUR = "#5aa2ff"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

.stApp {
  background:
    radial-gradient(1200px 700px at 12% -8%, #16264a 0%, rgba(22,38,74,0) 62%),
    radial-gradient(1000px 600px at 92% 4%, #0f2f44 0%, rgba(15,47,68,0) 58%),
    radial-gradient(900px 900px at 50% 108%, #131a38 0%, rgba(19,26,56,0) 60%),
    #05070f;
  color: #dbe6f5;
  font-family: 'Inter', system-ui, sans-serif;
}
section.main > div { padding-top: 1.2rem; }

.acv-mono, .acv-num { font-family: 'JetBrains Mono', ui-monospace, monospace; }

/* Header ------------------------------------------------------------- */
.acv-header {
  border: 1px solid rgba(120,170,255,.22);
  border-radius: 14px;
  padding: 20px 26px;
  margin-bottom: 20px;
  background: linear-gradient(135deg, rgba(20,34,68,.94), rgba(8,14,30,.94));
  box-shadow: 0 0 44px rgba(56,132,255,.16), inset 0 1px 0 rgba(255,255,255,.05);
}
.acv-header h1 {
  margin: 0; font-size: 1.62rem; font-weight: 700; letter-spacing: .022em;
  color: #f2f7ff;
}
.acv-header .sub {
  font-family: 'JetBrains Mono', monospace; font-size: .78rem;
  color: #7fa6da; margin-top: 7px; letter-spacing: .08em; text-transform: uppercase;
}

/* KPI ---------------------------------------------------------------- */
.acv-kpis { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 22px; }
.acv-kpi {
  flex: 1 1 172px;
  border: 1px solid rgba(120,170,255,.16);
  border-left: 3px solid var(--accent, #3884ff);
  border-radius: 11px; padding: 15px 17px;
  background: linear-gradient(160deg, rgba(17,28,56,.92), rgba(7,12,26,.92));
  box-shadow: 0 0 22px rgba(56,132,255,.09);
}
.acv-kpi .v {
  font-family: 'JetBrains Mono', monospace; font-size: 1.85rem; font-weight: 700;
  color: var(--accent, #eaf2ff); line-height: 1.1;
}
.acv-kpi .k {
  font-family: 'JetBrains Mono', monospace; font-size: .67rem; letter-spacing: .1em;
  text-transform: uppercase; color: #7e95b8; margin-top: 6px;
}

/* Section titles ------------------------------------------------------ */
.acv-sect {
  font-family: 'JetBrains Mono', monospace; font-size: .8rem; letter-spacing: .15em;
  text-transform: uppercase; color: #8fb4e6; margin: 26px 0 12px;
  border-bottom: 1px solid rgba(120,170,255,.16); padding-bottom: 7px;
}

/* Rows / cards -------------------------------------------------------- */
.acv-row, .acv-card {
  border: 1px solid rgba(120,170,255,.14);
  border-left: 4px solid var(--sev, #8193ad);
  border-radius: 10px; padding: 13px 17px; margin-bottom: 9px;
  background: linear-gradient(160deg, rgba(16,26,52,.9), rgba(7,12,26,.9));
  box-shadow: 0 0 20px -6px var(--sev, transparent);
}
.acv-row .top { display: flex; align-items: baseline; gap: 13px; flex-wrap: wrap; }
.acv-rank {
  font-family: 'JetBrains Mono', monospace; font-size: 1.15rem; font-weight: 700;
  color: #6f8bb5; min-width: 2.1rem;
}
.acv-file { font-weight: 600; color: #eef4ff; font-size: .97rem; }
.acv-meta {
  font-family: 'JetBrains Mono', monospace; font-size: .72rem; color: #7a90b4;
}
.acv-badge {
  font-family: 'JetBrains Mono', monospace; font-size: .67rem; font-weight: 700;
  letter-spacing: .09em; padding: 3px 9px; border-radius: 5px;
  border: 1px solid var(--sev); color: var(--sev); background: rgba(255,255,255,.035);
  white-space: nowrap;
}
.acv-say { color: #b7c8e2; font-size: .87rem; margin-top: 7px; line-height: 1.5; }

/* Train strip --------------------------------------------------------- */
.acv-train { display: flex; gap: 9px; flex-wrap: wrap; margin: 6px 0 4px; }
.acv-carbox {
  flex: 1 1 108px; min-width: 104px;
  border: 1px solid rgba(120,170,255,.16); border-top: 3px solid var(--sev);
  border-radius: 9px; padding: 12px 10px; text-align: center;
  background: linear-gradient(170deg, rgba(17,28,56,.92), rgba(7,12,26,.92));
  box-shadow: 0 0 18px -7px var(--sev);
}
.acv-carbox .id {
  font-family: 'JetBrains Mono', monospace; font-size: 1.28rem; font-weight: 700;
  color: #edf3ff;
}
.acv-carbox .ic { font-family: 'JetBrains Mono', monospace; font-size: .95rem; color: var(--sev); }
.acv-carbox .sf {
  font-family: 'JetBrains Mono', monospace; font-size: .95rem; color: var(--sev); margin-top: 3px;
}
.acv-carbox .bd {
  font-family: 'JetBrains Mono', monospace; font-size: .6rem; letter-spacing: .07em;
  color: #8296b8; margin-top: 4px;
}
.acv-carbox .rk {
  font-family: 'JetBrains Mono', monospace; font-size: .6rem; color: #64789c; margin-top: 2px;
}
.acv-carbox .sub {
  font-family: 'JetBrains Mono', monospace; font-size: .64rem; color: #7d92b5; margin-top: 2px;
}
.acv-carbox .z {
  font-family: 'JetBrains Mono', monospace; font-size: .62rem; color: #8ea6c9; margin-top: 2px;
}
/* No telemetry: dashed, off-scale colour, never mistakable for "OK". */
.acv-carbox.nodata { border-style: dashed; border-width: 1px; border-top-width: 3px; }
/* Rank 1 is always marked, whatever its band. */
.acv-carbox.prime { box-shadow: 0 0 0 2px var(--prime), 0 0 22px -6px var(--sev); }
.acv-carbox .prime-tag {
  font-family: 'JetBrains Mono', monospace; font-size: .55rem; font-weight: 700;
  letter-spacing: .08em; color: var(--prime); margin-bottom: 4px;
}
.acv-note {
  border-left: 3px solid rgba(120,170,255,.35); padding: 9px 14px; margin: 10px 0 4px;
  background: rgba(20,34,68,.5); border-radius: 0 8px 8px 0;
  color: #a9bdda; font-size: .85rem; line-height: 1.55;
}

/* Alerts -------------------------------------------------------------- */
.acv-alert {
  border: 1px solid rgba(120,170,255,.14); border-left: 4px solid var(--sev);
  border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
  background: linear-gradient(160deg, rgba(16,26,52,.9), rgba(7,12,26,.9));
  box-shadow: 0 0 22px -8px var(--sev);
}
.acv-alert .h {
  font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: .84rem;
  letter-spacing: .07em; color: var(--sev);
}
.acv-alert p { margin: 7px 0 0; color: #b9cae4; font-size: .87rem; line-height: 1.55; }
.acv-alert .rank-note { color: #8aa2c6; font-style: italic; font-size: .82rem; }

div[data-testid="stDataFrame"] { border-radius: 10px; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Analysis, cached on file bytes so re-rendering never re-runs the model
# --------------------------------------------------------------------------

def _engine_fingerprint():
    """Hash of the engine source, so editing acv_core invalidates the cache.

    Streamlit keys cache_data on this function's own body and its arguments.
    Without the fingerprint an edit to acv_core.py leaves the console serving
    results produced by the previous version of the engine.
    """
    try:
        with open(core.__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:16]
    except OSError:
        return "unknown"


ENGINE_FINGERPRINT = _engine_fingerprint()


@st.cache_data(show_spinner=False)
def analyse(payload: bytes, file_id: str, engine: str):
    return core.analyze_workbook(payload, file_id=file_id)


def severity_of(result):
    """Worst band present on a trainset, for the fleet view."""
    for band in core.SEVERITY_ORDER:
        if any(info["severity"] == band for info in result["cars"]):
            return band
    return "NO DATA"


def fmt(value, spec="{:+.2f}"):
    return "--" if value is None else spec.format(value)


def esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------

st.markdown(
    '<div class="acv-header">'
    '<h1>ACV Thermal Integrity Depot Console</h1>'
    '<div class="sub">Refrigerant-leak triage &middot; 8-car trainsets &middot; '
    'ranking is physically derived, not learned</div></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Data source")
    uploads = st.file_uploader(
        "Trainset workbooks", type=["xlsx", "xlsm"], accept_multiple_files=True)
    use_bundled = st.checkbox(
        "Load the bundled dataset", value=not uploads,
        help="Reads every workbook under data/Train and data/Test.")
    st.markdown("---")
    st.caption(
        "Ranking basis is chosen by the schema, not by this panel. "
        "Nothing on this page can reorder a car."
    )

payloads = []
if uploads:
    for upload in uploads:
        payloads.append((upload.name, upload.getvalue()))
if use_bundled:
    seen = {name for name, _ in payloads}
    for folder in BUNDLED:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith((".xlsx", ".xlsm")) or name.startswith("~$"):
                continue
            if name in seen:
                continue
            with open(os.path.join(folder, name), "rb") as handle:
                payloads.append((name, handle.read()))
            seen.add(name)

if not payloads:
    st.info("Upload one or more trainset workbooks, or tick "
            "**Load the bundled dataset** in the sidebar.")
    st.stop()

results = []
progress = st.progress(0.0, text="Reading trainsets ...")
for position, (name, payload) in enumerate(payloads, start=1):
    progress.progress((position - 1) / len(payloads),
                      text="Reading {} ({}/{}) ...".format(name, position, len(payloads)))
    results.append(analyse(payload, name, ENGINE_FINGERPRINT))
progress.empty()


# --------------------------------------------------------------------------
# 2 -- KPI strip
# --------------------------------------------------------------------------

worst = {result["file_id"]: severity_of(result) for result in results}
withdraw = sum(1 for band in worst.values() if band in ("CRITICAL", "HIGH"))
watch = sum(1 for band in worst.values() if band == "ELEVATED")
serviceable = sum(1 for band in worst.values() if band == "NORMAL")
dark_cars = sum(1 for result in results for info in result["cars"]
                if info["status"] != "scored")

kpis = [
    ("Trainsets analysed", len(results), "#5aa2ff"),
    ("Withdraw / schedule", withdraw, SEVERITY_COLOUR["HIGH"]),
    ("Watchlist", watch, SEVERITY_COLOUR["ELEVATED"]),
    ("Serviceable", serviceable, SEVERITY_COLOUR["NORMAL"]),
    ("Carriages with no telemetry", dark_cars, SEVERITY_COLOUR["NO DATA"]),
]
st.markdown(
    '<div class="acv-kpis">' + "".join(
        '<div class="acv-kpi" style="--accent:{}"><div class="v">{}</div>'
        '<div class="k">{}</div></div>'.format(colour, value, esc(label))
        for label, value, colour in kpis
    ) + "</div>",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# 3 -- Fleet withdrawal priority
# --------------------------------------------------------------------------

st.markdown('<div class="acv-sect">Fleet withdrawal priority</div>',
            unsafe_allow_html=True)
st.caption("Worst carriage first. This is the pull-out order; the rank column "
           "is the console's recommendation, not a score.")

ordered = sorted(
    results,
    key=lambda r: (core.SEVERITY_ORDER.index(severity_of(r)),
                   -(r["cars"][0]["shortfall"] if r["cars"] and
                     r["cars"][0]["shortfall"] is not None else -99)),
)

for position, result in enumerate(ordered, start=1):
    band = severity_of(result)
    colour = SEVERITY_COLOUR[band]
    lead_note = result["prime_qualifier"]
    top = result["cars"][0] if result["cars"] else None
    carriage = top["car"] if top else "--"
    shortfall = (
        "{:.0f}% circuit divergence".format((top["asymmetry"] or 0.0) * 100)
        if top is not None and result["basis_metric"] == "asymmetry"
        else fmt(top["shortfall"] if top else None) + " C")
    action = top["action"] if top else "Check the data feed before judging"
    st.markdown(
        '<div class="acv-row" style="--sev:{colour}">'
        '<div class="top">'
        '<span class="acv-rank">{pos:02d}</span>'
        '<span class="acv-file">{file}</span>'
        '<span class="acv-badge">{icon} {band}</span>'
        '<span class="acv-meta">model {model} &middot; train {train} &middot; '
        '{basis}</span>'
        '</div>'
        '<div class="acv-say">PRIME SUSPECT carriage <b class="acv-mono">{car}'
        '</b> &middot; <span class="acv-num">{sf}</span> &middot; '
        '{lead} ({leadval}) &mdash; {action}</div>'
        '</div>'.format(
            colour=colour, pos=position, file=esc(result["file_id"]),
            icon=core.SEVERITY_ICON[band], band=band,
            model=esc(result["car_model"]), train=esc(result["train_number"]),
            basis=esc(result["ranking_basis"]), car=esc(carriage), sf=shortfall,
            lead=lead_note, leadval=fmt(result["lead"], "{:+.3f}"),
            action=esc(action)),
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# 4 -- Sensor health
# --------------------------------------------------------------------------

st.markdown('<div class="acv-sect">Sensor health</div>', unsafe_allow_html=True)
st.caption("Missing telemetry is a maintenance finding in its own right. "
           "Severity here is separate from the leak ranking.")

ALERT_RANK = {"fault": 0, "advisory": 1, "informational": 2}
ALERT_STYLE = {
    "fault": (SEVERITY_COLOUR["CRITICAL"], "!!!"),
    "advisory": (SEVERITY_COLOUR["ELEVATED"], "!"),
    "informational": (SEVERITY_COLOUR["NO DATA"], "i"),
}

alerts = [(result["file_id"], alert)
          for result in results for alert in result["alerts"]]
alerts.sort(key=lambda item: (ALERT_RANK[item[1]["severity"]],
                              item[0], item[1]["kind"]))

if not alerts:
    st.markdown(
        '<div class="acv-alert" style="--sev:{}"><div class="h">OK  NO SENSOR '
        'FAULTS</div><p>Every carriage reported its air-conditioning telemetry '
        'throughout.</p></div>'.format(SEVERITY_COLOUR["NORMAL"]),
        unsafe_allow_html=True)
else:
    for file_id, alert in alerts:
        colour, icon = ALERT_STYLE[alert["severity"]]
        st.markdown(
            '<div class="acv-alert" style="--sev:{colour}">'
            '<div class="h">{icon}  {head}</div>'
            '<p class="acv-meta">{file}</p>'
            '<p>{seen}</p><p>{means}</p><p><b>{action}</b></p>'
            '<p class="rank-note">Leak ranking: {note}</p></div>'.format(
                colour=colour, icon=icon, head=esc(alert["headline"]),
                file=esc(file_id), seen=esc(alert["seen"]),
                means=esc(alert["means"]), action=esc(alert["action"]),
                note=esc(alert["ranking_note"])),
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------------------
# 5 -- Carriage inspection
# --------------------------------------------------------------------------

st.markdown('<div class="acv-sect">Carriage inspection</div>',
            unsafe_allow_html=True)

choice = st.selectbox("Trainset", [result["file_id"] for result in results],
                      key="inspect")
selected = next(r for r in results if r["file_id"] == choice)

st.caption("Carriages in physical order. Severity is measured against this "
           "trainset's own fleet, so it does not drift with the weather or the "
           "setpoint. Colour is never the only cue: each tile carries an icon "
           "and a written band.")

on_asymmetry = selected["basis_metric"] == "asymmetry"

tiles = []
for car in sorted(selected["cars_by_id"]):
    info = selected["cars_by_id"][car]
    band = info["severity"]
    scored = info["status"] == "scored"
    is_prime = info["rank"] == 1

    if not scored:
        headline, secondary = "NO TELEMETRY", "not assessed"
    elif on_asymmetry:
        # The headline metric must be the one the file was ranked on, or the
        # displayed numbers contradict the displayed order.
        headline = "{:.0f}%".format((info["asymmetry"] or 0.0) * 100)
        secondary = "circuit divergence"
    else:
        headline, secondary = fmt(info["shortfall"]) + " C", "vs own setpoint"

    extra = ""
    if scored and on_asymmetry:
        extra = '<div class="sub">{} C thermal</div>'.format(fmt(info["shortfall"]))

    tiles.append(
        '<div class="acv-carbox{cls}" style="--sev:{colour};--prime:{prime}">'
        '{tag}'
        '<div class="id">{car}</div>'
        '<div class="ic">{icon}</div>'
        '<div class="sf">{headline}</div>'
        '<div class="sub">{secondary}</div>'
        '{extra}'
        '<div class="z">{z}</div>'
        '<div class="bd">{band}</div>'
        '<div class="rk">rank {rank} of 8</div>'
        '</div>'.format(
            cls=(" prime" if is_prime else "") + ("" if scored else " nodata"),
            colour=SEVERITY_COLOUR[band], prime=PRIME_COLOUR,
            tag=('<div class="prime-tag">PRIME SUSPECT</div>' if is_prime
                 else '<div class="prime-tag">&nbsp;</div>'),
            car=esc(car), icon=core.SEVERITY_ICON[band],
            headline=headline, secondary=secondary, extra=extra,
            z=("{:.1f}&sigma; vs fleet".format(info["fleet_z"])
               if info["fleet_z"] is not None else "&nbsp;"),
            band=band, rank=info["rank"]))
st.markdown('<div class="acv-train">' + "".join(tiles) + "</div>",
            unsafe_allow_html=True)

prime = selected["prime_car"]
if prime:
    st.markdown(
        '<div class="acv-note"><b class="acv-mono">PRIME SUSPECT &middot; '
        'carriage {}</b> &mdash; {}. Severity answers <i>how bad</i>; rank '
        'answers <i>which one</i>. A carriage can be the prime suspect and '
        'still sit inside its own fleet.</div>'.format(
            esc(prime), esc(selected["prime_qualifier"])),
        unsafe_allow_html=True)

if on_asymmetry:
    st.markdown(
        '<div class="acv-note">Ranked on within-car circuit divergence. A '
        'refrigerant leak starves one of a car\'s two refrigeration circuits '
        'while the other takes up the load, so the two circuits diverge. A car '
        'that is heavily loaded on <i>both</i> circuits equally is a '
        'cooling-demand or condenser issue, not a leak &mdash; which is why the '
        'thermal shortfall is shown as a secondary line here and is not what '
        'this trainset was ranked on.</div>',
        unsafe_allow_html=True)

if selected["peak_unavailable_cars"]:
    st.markdown(
        '<div class="acv-note">Peak-load corroboration unavailable for '
        'carriage{} {} &mdash; this trainset\'s outdoor temperature sensor is '
        'not reporting on {}. The hot-weather cross-check could not be run for '
        'them.</div>'.format(
            "" if len(selected["peak_unavailable_cars"]) == 1 else "s",
            esc(", ".join(selected["peak_unavailable_cars"])),
            "any car" if len(selected["peak_unavailable_cars"]) >= 7
            else "those carriages"),
        unsafe_allow_html=True)

st.markdown(
    '<div class="acv-meta">Basis: <b>{}</b> &mdash; {}<br>Fleet baseline: '
    'median {} , MAD {} &mdash; bands are z &ge; 2 elevated, &ge; 4 high, '
    '&ge; 8 critical.</div>'.format(
        esc(selected["ranking_basis"]), esc(selected["basis_reason"]),
        fmt(selected["fleet_median"], "{:+.3f}"),
        fmt(selected["fleet_mad"], "{:.4f}")),
    unsafe_allow_html=True)


# --------------------------------------------------------------------------
# 6 -- Priority carriages
# --------------------------------------------------------------------------

st.markdown('<div class="acv-sect">Priority carriages</div>',
            unsafe_allow_html=True)

# The prime suspect always appears here, even when its band is NORMAL: a
# trainset whose cars all sit inside their own fleet still has a rank 1, and the
# controller needs to see which carriage that is.
priority = [info for info in selected["cars"]
            if info["severity"] in ("CRITICAL", "HIGH", "ELEVATED", "NO DATA")
            or info["rank"] == 1]
if all(info["severity"] in ("NORMAL",) for info in priority):
    st.info("No carriage on this trainset stands outside its own fleet. The "
            "prime suspect below is the model's best candidate, not a fault.")

for info in priority:
    band = info["severity"]
    colour = PRIME_COLOUR if info["rank"] == 1 else SEVERITY_COLOUR[band]
    st.markdown(
        '<div class="acv-card" style="--sev:{colour}">'
        '<div class="top"><span class="acv-rank">{car}</span>'
        '{prime}'
        '<span class="acv-badge" style="--sev:{bandcolour}">{icon} {band}</span>'
        '<span class="acv-meta">rank {rank} of 8 &middot; {z}</span></div>'
        '<div class="acv-say">{action}</div></div>'.format(
            colour=colour, bandcolour=SEVERITY_COLOUR[band],
            car=esc(info["car"]),
            prime=('<span class="acv-badge" style="--sev:{}">PRIME SUSPECT '
                   '&middot; {}</span>'.format(
                       PRIME_COLOUR, esc(selected["prime_qualifier"]))
                   if info["rank"] == 1 else ""),
            icon=core.SEVERITY_ICON[band], band=band, rank=info["rank"],
            z=("{:.1f}&sigma; outside its own fleet".format(info["fleet_z"])
               if info["fleet_z"] is not None else "not measured"),
            action=esc(info["action"])),
        unsafe_allow_html=True)

    with st.expander("Evidence for carriage {}".format(info["car"])):
        peak = info["peak_shortfall"]
        agrees = info["peak_agrees"]
        lines = [
            ("Shortfall against its own setpoint",
             "{} C over {} eligible readings".format(
                 fmt(info["shortfall"]), info["samples"])),
            ("Cabin / setpoint means",
             "{} C held against {} C commanded".format(
                 fmt(info["outcome_mean"], "{:.2f}"),
                 fmt(info["setpoint_mean"], "{:.2f}"))),
            ("Against its own fleet",
             "--" if info["fleet_z"] is None else
             "{:.2f} standard deviations outside the fleet median of {} C "
             "(MAD {})".format(
                 info["fleet_z"], fmt(selected["fleet_median"], "{:+.3f}"),
                 fmt(selected["fleet_mad"], "{:.4f}"))),
            ("Under peak thermal load",
             "unavailable -- this trainset's outdoor sensor is not reporting "
             "for this carriage" if peak is None
             else "{} C over {} readings ({})".format(
                 fmt(peak), info["peak_samples"],
                 "agrees" if agrees else "disagrees")),
            ("Cross-car agreement",
             "--" if info["anomaly"] is None else
             "mean |z| {:.3f}, {} of the train".format(
                 info["anomaly"],
                 "most anomalous" if info.get("anomaly_rank") == 1
                 else "rank {}".format(info.get("anomaly_rank")))),
            ("Filters applied",
             ", ".join(info["filters_applied"]) or "none survived the guard"),
            ("Status", info["status"]),
        ]
        if info["asymmetry"] is not None:
            channels = info["asymmetry_detail"].get("channels", {})
            lines.insert(1, (
                "Circuit asymmetry",
                "{:.3f} overall ({})".format(
                    info["asymmetry"],
                    ", ".join("{} {:.2f}".format(core.CIRCUIT_LABEL.get(k, k), v)
                              for k, v in sorted(channels.items())))))
        if info["duty_quarters"]:
            lines.append((
                "Per-circuit duty by quarter",
                "  ".join("Q{} {}/{}".format(
                    i + 1,
                    "--" if a is None else "{:.2f}".format(a),
                    "--" if b is None else "{:.2f}".format(b))
                    for i, (a, b) in enumerate(info["duty_quarters"]))))
        st.table(pd.DataFrame(lines, columns=["Evidence", "Reading"]))


# --------------------------------------------------------------------------
# 7 -- Audit
# --------------------------------------------------------------------------

st.markdown('<div class="acv-sect">Audit</div>', unsafe_allow_html=True)

with st.expander("Per-car numbers -- {}".format(selected["file_id"])):
    st.dataframe(pd.DataFrame([{
        "rank": info["rank"],
        "car": info["car"],
        "status": info["status"],
        "band": info["severity"],
        "shortfall_C": info["shortfall"],
        "fleet_z": info["fleet_z"],
        "peak_shortfall_C": info["peak_shortfall"],
        "asymmetry": info["asymmetry"],
        "mean_abs_z": info["anomaly"],
        "eligible_rows": info["samples"],
        "live_rows": info["live_rows"],
        "dropouts": info["dropouts"],
        "setpoint_mean_C": info["setpoint_mean"],
        "cabin_mean_C": info["outcome_mean"],
    } for info in selected["cars"]]), width="stretch", hide_index=True)
    st.caption(
        "Ranking basis **{}** -- {}.  Read {} of {} columns over {} rows in "
        "{:.1f}s.  Cross-car margin {}.".format(
            selected["ranking_basis"], selected["basis_reason"],
            selected["columns_read"], selected["columns_total"],
            selected["rows"], selected["elapsed"],
            "--" if selected["z_margin"] is None
            else "{:.2f}x".format(selected["z_margin"])))

with st.expander("Schema resolution -- {}".format(selected["file_id"])):
    schema = core.sniff(io.BytesIO(
        dict(payloads)[selected["file_id"]])) if selected["file_id"] in dict(payloads) else None
    if schema is not None:
        st.dataframe(pd.DataFrame([{
            "column": column.header,
            "car": column.car or "-",
            "parameter": column.param,
            "nature": column.nature,
            "role": column.role or "-",
            "matched_by": column.matched_by or "-",
            "circuit": "{} {}".format(column.circuit_channel, column.circuit_index)
                       if column.circuit_channel else "-",
        } for column in schema.columns]), width="stretch", hide_index=True)
    st.markdown("**Resolution log**")
    st.code("\n".join(selected["schema_log"]) or "(empty)", language="text")


# --------------------------------------------------------------------------
# 8 -- Export
# --------------------------------------------------------------------------

st.markdown('<div class="acv-sect">Export</div>', unsafe_allow_html=True)

submission = core.submission_csv(results)
work_order = core.work_order_csv(results)

buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
    archive.writestr("acv_predictions.csv", submission)

left, middle, right = st.columns(3)
left.download_button("acv_predictions.csv", submission, "acv_predictions.csv",
                     "text/csv", width="stretch")
middle.download_button("predictions.zip", buffer.getvalue(), "predictions.zip",
                       "application/zip", width="stretch")
right.download_button("depot work order", work_order, "acv_work_order.csv",
                      "text/csv", width="stretch")

st.dataframe(pd.DataFrame(core.submission_rows(results)),
             width="stretch", hide_index=True)
st.caption("Two columns, `file_id` lowercase. Built from the same ranked_cars "
           "string the engine returned -- no view setting on this page can "
           "change it.")
