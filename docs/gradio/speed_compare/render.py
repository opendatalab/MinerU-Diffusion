from html import escape
from speed_compare.parsers import build_panel_state, format_stats


APP_CSS = """
:root,
body.light,
.light,
.gradio-container {
  --stage-height: calc(100vh - 170px);
  --page-bloom-a: rgba(255, 120, 0, 0.10);
  --page-bloom-b: rgba(77, 163, 255, 0.08);
  --page-bg-start: #fbfbfd;
  --page-bg-end: #eef2f8;
  --page-text: #12131a;
  --panel-border: rgba(18, 19, 26, 0.10);
  --panel-bg:
    radial-gradient(circle at 50% 36%, rgba(255,255,255,0.75), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,0.94), rgba(245,247,251,0.98));
  --panel-shadow: 0 28px 60px rgba(31, 41, 55, 0.10);
  --muted-text: #68707f;
  --input-border: rgba(18, 19, 26, 0.12);
  --chip-bg: rgba(18, 19, 26, 0.04);
  --chip-text: #181b24;
  --divider: rgba(18, 19, 26, 0.12);
  --status-text: #222632;
  --stats-text: #39404f;
  --timeline-bg: rgba(18, 19, 26, 0.10);
  --output-bg: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,253,0.98));
  --output-border: rgba(18, 19, 26, 0.10);
  --output-shadow: inset 0 0 0 1px rgba(255,255,255,0.6), 0 0 30px rgba(15,23,42,0.05);
  --output-text: #181b24;
  --empty-text: #7a8291;
  --placeholder-text: rgba(18, 19, 26, 0.14);
}

.dark,
body.dark,
.dark .gradio-container,
[data-theme="dark"],
[data-theme="dark"] .gradio-container {
  --page-bloom-a: rgba(255, 95, 210, 0.08);
  --page-bloom-b: rgba(255, 255, 255, 0.05);
  --page-bg-start: #000000;
  --page-bg-end: #030303;
  --page-text: #f5f5f5;
  --panel-border: rgba(255,255,255,0.14);
  --panel-bg:
    radial-gradient(circle at 50% 36%, rgba(255,255,255,0.05), transparent 28%),
    linear-gradient(180deg, rgba(12,12,12,0.96), rgba(4,4,4,0.98));
  --panel-shadow: 0 28px 60px rgba(0, 0, 0, 0.55);
  --muted-text: #8d8d8d;
  --input-border: rgba(255,255,255,0.16);
  --chip-bg: rgba(255,255,255,0.04);
  --chip-text: #f5f5f5;
  --divider: rgba(255,255,255,0.14);
  --status-text: #d9d9d9;
  --stats-text: #d9d9d9;
  --timeline-bg: rgba(255,255,255,0.12);
  --output-bg: linear-gradient(180deg, rgba(10,10,10,0.98), rgba(5,5,5,0.98));
  --output-border: rgba(255,255,255,0.14);
  --output-shadow: inset 0 0 0 1px rgba(255,255,255,0.02), 0 0 40px rgba(255,255,255,0.08);
  --output-text: #f2f2f2;
  --empty-text: #8d8d8d;
  --placeholder-text: rgba(255,255,255,0.12);
}

@media (prefers-color-scheme: dark) {
  :root:not(.light):not([data-theme="light"]) {
    --page-bloom-a: rgba(255, 95, 210, 0.08);
    --page-bloom-b: rgba(255, 255, 255, 0.05);
    --page-bg-start: #000000;
    --page-bg-end: #030303;
    --page-text: #f5f5f5;
    --panel-border: rgba(255,255,255,0.14);
    --panel-bg:
      radial-gradient(circle at 50% 36%, rgba(255,255,255,0.05), transparent 28%),
      linear-gradient(180deg, rgba(12,12,12,0.96), rgba(4,4,4,0.98));
    --panel-shadow: 0 28px 60px rgba(0, 0, 0, 0.55);
    --muted-text: #8d8d8d;
    --input-border: rgba(255,255,255,0.16);
    --chip-bg: rgba(255,255,255,0.04);
    --chip-text: #f5f5f5;
    --divider: rgba(255,255,255,0.14);
    --status-text: #d9d9d9;
    --stats-text: #d9d9d9;
    --timeline-bg: rgba(255,255,255,0.12);
    --output-bg: linear-gradient(180deg, rgba(10,10,10,0.98), rgba(5,5,5,0.98));
    --output-border: rgba(255,255,255,0.14);
    --output-shadow: inset 0 0 0 1px rgba(255,255,255,0.02), 0 0 40px rgba(255,255,255,0.08);
    --output-text: #f2f2f2;
    --empty-text: #8d8d8d;
    --placeholder-text: rgba(255,255,255,0.12);
  }
}

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  margin: 0 !important;
  padding-left: 20px !important;
  padding-right: 20px !important;
  background:
    radial-gradient(circle at 20% 10%, var(--page-bloom-a), transparent 22%),
    radial-gradient(circle at 80% 8%, var(--page-bloom-b), transparent 18%),
    linear-gradient(180deg, var(--page-bg-start) 0%, var(--page-bg-end) 100%);
  color: var(--page-text);
  font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}

footer,
.footer,
.built-with-gradio,
[data-testid="footer"],
button[aria-label="settings"] {
  display: none !important;
}

#app-shell {
  gap: 24px;
  align-items: stretch;
  flex-wrap: wrap;
}

#left-col,
#mineru-col,
#diffusion-col {
  min-height: var(--stage-height);
  border: 1px solid var(--panel-border);
  background: var(--panel-bg);
  box-shadow: var(--panel-shadow);
  padding: 20px;
  border-radius: 22px;
  overflow: hidden;
}

#mineru-col,
#diffusion-col {
  padding: 0 0 12px;
}

.panel-title {
  margin: 0 0 10px 0 !important;
  letter-spacing: 0.04em;
}

.panel-note {
  color: var(--muted-text);
  margin: 0 0 16px 0;
  line-height: 1.6;
}

#left-col .gr-form,
#left-col .gr-box,
#left-col .gr-panel {
  background: transparent !important;
  border: none !important;
}

#left-col .gradio-image,
#left-col .gradio-radio {
  border-color: var(--input-border) !important;
  color: var(--page-text) !important;
}

#left-col label,
#left-col span,
#left-col p,
#left-col .gradio-image button,
#left-col .gradio-image [role="button"] {
  color: var(--page-text) !important;
}

#left-col .gradio-image,
#left-col .gradio-image > div,
#left-col .gradio-radio,
#left-col .gradio-radio > div {
  background: transparent !important;
}

#left-col .gradio-image .wrap,
#left-col .gradio-image .image-container,
#left-col .gradio-image .image-frame,
#left-col .gradio-image .upload-container,
#left-col .gradio-radio .wrap {
  background: transparent !important;
  border-color: var(--input-border) !important;
}

#prompt-grid {
  margin-top: 10px;
}

#prompt-grid .gradio-radio {
  background: transparent !important;
  border: none !important;
}

#prompt-grid .wrap {
  display: grid !important;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px !important;
}

#prompt-grid label {
  margin: 0 !important;
  min-width: 0 !important;
}

#prompt-grid label > span,
#prompt-grid label[role="radio"] {
  width: 100%;
  min-height: 56px;
}

#prompt-grid label {
  display: grid !important;
  place-items: center;
  height: 56px;
  padding: 0 !important;
  border: 1px solid var(--input-border);
  border-radius: 14px;
  background: var(--chip-bg);
  color: var(--chip-text);
  text-align: center;
  font-size: 18px;
  line-height: 1;
  font-weight: 500;
}

#prompt-grid label > span,
#prompt-grid label > div,
#prompt-grid label p {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 100%;
  height: 100%;
  margin: 0 !important;
  text-align: center;
}

#prompt-grid input[type="radio"] {
  display: none;
}

#prompt-grid label:has(input[type="radio"]:checked) {
  border-color: rgba(123, 97, 255, 0.95);
  background: linear-gradient(90deg, rgba(77,163,255,0.18), rgba(123,97,255,0.24) 55%, rgba(180,108,255,0.18));
  box-shadow: 0 0 0 1px rgba(123,97,255,0.26) inset;
}

#start-button {
  border-radius: 16px !important;
  background: linear-gradient(90deg, #4da3ff, #7b61ff 55%, #b46cff) !important;
  color: #ffffff !important;
  border: none !important;
}

#mineru-render-toggle,
#diffusion-render-toggle {
  margin: 12px 18px 12px !important;
  border-radius: 14px !important;
  width: calc(100% - 36px) !important;
  max-width: calc(100% - 36px) !important;
  box-sizing: border-box !important;
  background: linear-gradient(90deg, #4da3ff, #7b61ff 55%, #b46cff) !important;
  border: none !important;
  color: #ffffff !important;
}

#mineru-render-toggle[disabled],
#diffusion-render-toggle[disabled] {
  background: var(--chip-bg) !important;
  border: 1px solid var(--input-border) !important;
  color: var(--chip-text) !important;
}

.replay-stream {
  position: relative;
  height: var(--stage-height);
  padding: 26px 24px 22px;
  display: flex;
  flex-direction: column;
  border-radius: 22px;
  overflow: hidden;
}

.replay-stream::before {
  content: "";
  position: absolute;
  inset: 22px 18px auto 18px;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--divider), transparent);
  opacity: 0.35;
}

.label {
  flex: 0 0 20px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  font-size: 12px;
  color: #8a8a8a;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 16px;
}

.label.done {
  color: #67f59b;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 999px;
  display: inline-block;
  background: #7a7a7a;
}

.label.done .dot {
  background: #67f59b;
  box-shadow: 0 0 12px rgba(103,245,155,0.45);
}

.stream-head {
  flex: 0 0 118px;
  text-align: center;
  margin-bottom: 20px;
  min-height: 118px;
  display: flex;
  flex-direction: column;
  justify-content: center;
}

.name {
  margin: 0;
  font-size: clamp(22px, 1.8vw, 30px);
  line-height: 1;
  letter-spacing: 0.04em;
  font-weight: 400;
}

.stream-meta {
  flex: 0 0 52px;
  display: flex;
  justify-content: space-between;
  align-items: end;
  gap: 18px;
  margin: 0 0 14px;
  color: var(--muted-text);
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.stream-status {
  font-size: clamp(20px, 2vw, 32px);
  line-height: 1;
  color: #9a9a9a;
  font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, "Liberation Mono", monospace;
  letter-spacing: 0.08em;
  background: linear-gradient(90deg, #707070 0%, #bdbdbd 30%, #7c7c7c 55%, #d0d0d0 78%, #747474 100%);
  background-size: 240% 100%;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: statusPulse 2.6s linear infinite;
}

.stream-status:not(.done) {
  color: #9a9a9a;
}

.stream-status.done {
  color: #67f59b;
  background: none;
  -webkit-text-fill-color: initial;
  animation: none;
}

.stream-stats {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  justify-content: center;
  gap: 2px;
  text-transform: none;
  letter-spacing: 0.04em;
}

.stream-stats span {
  font-size: clamp(13px, 1.2vw, 16px);
  line-height: 1;
  color: var(--stats-text);
}

.stream-progress {
  flex: 0 0 52px;
  margin: 0 0 18px;
}

.timeline {
  position: relative;
  height: 8px;
  border-radius: 999px;
  background: var(--timeline-bg);
  overflow: hidden;
}

.timeline-fill {
  position: absolute;
  inset: 0 auto 0 0;
  width: 0%;
  background: linear-gradient(90deg, #4da3ff, #7b61ff 55%, #b46cff);
  box-shadow: 0 0 20px rgba(109,107,255,0.34);
}

.stream-clock {
  flex: 0 0 88px;
  margin: 10px 0 6px;
  text-align: center;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stream-clock strong {
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: clamp(38px, 3.4vw, 56px);
  line-height: 1;
  letter-spacing: 0.04em;
  font-weight: 500;
  margin: 0;
}

.stream-clock span {
  display: block;
  margin-top: 0;
  font-size: 12px;
  color: var(--muted-text);
  text-transform: uppercase;
  letter-spacing: 0.14em;
}

.stream-output {
  min-height: 0;
  flex: 1 1 auto;
  overflow: auto;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  font-size: 13px;
  line-height: 1.7;
  padding: 20px 18px;
  border-radius: 14px;
  background: var(--output-bg);
  border: 1px solid var(--output-border);
  box-shadow: var(--output-shadow);
}

#mineru-col .replay-stream,
#diffusion-col .replay-stream {
  height: calc(var(--stage-height) - 62px);
  padding: 16px 18px 12px;
}

#mineru-col .stream-head,
#diffusion-col .stream-head {
  flex: 0 0 52px;
  min-height: 52px;
  margin-bottom: 2px;
}

#mineru-col .stream-meta,
#diffusion-col .stream-meta {
  flex: 0 0 36px;
  margin: 0 0 8px;
}

#mineru-col .stream-progress,
#diffusion-col .stream-progress {
  flex: 0 0 20px;
  margin: 0;
}

#mineru-col .stream-clock,
#diffusion-col .stream-clock {
  flex: 0 0 42px;
  margin: 0;
}

#mineru-col .stream-clock strong,
#diffusion-col .stream-clock strong {
  font-size: clamp(28px, 2.6vw, 40px);
}

#mineru-col .stream-output,
#diffusion-col .stream-output {
  padding: 22px 20px;
}

.stream-output.empty {
  color: var(--empty-text);
  font-style: italic;
}

.rendered-output {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  white-space: normal;
}

.rendered-pre {
  width: 100%;
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  color: var(--output-text);
  font: inherit;
}

.rendered-error {
  width: 100%;
  border-radius: 12px;
  border: 1px solid rgba(255, 95, 95, 0.28);
  background: rgba(255, 95, 95, 0.08);
  color: var(--output-text);
  padding: 14px 16px;
  line-height: 1.6;
}

.rendered-table-wrap,
.rendered-formula-wrap,
.rendered-layout-wrap {
  width: 100%;
}

.rendered-table-wrap {
  overflow: auto;
}

.rendered-table {
  width: 100%;
  border-collapse: collapse;
  background: transparent;
  color: var(--output-text);
  font-size: 13px;
}

.rendered-table th,
.rendered-table td {
  border: 1px solid var(--output-border);
  padding: 8px 10px;
  text-align: left;
  vertical-align: top;
}

.rendered-table th {
  background: rgba(127, 127, 127, 0.10);
}

.rendered-image {
  display: block;
  width: 100%;
  height: auto;
  border-radius: 12px;
  background: #ffffff;
}

.formula-image {
  margin-bottom: 14px;
}

.formula-source {
  margin-top: 0;
  font-size: 12px;
  line-height: 1.6;
  opacity: 0.82;
}

.chunk {
  color: var(--output-text);
}

.new-chunk {
  color: #ffffff;
  background: rgba(255,95,210,0.16);
  box-shadow: 0 0 12px rgba(255,95,210,0.18);
}

.placeholder {
  color: var(--placeholder-text);
}

.cursor {
  display: inline-block;
  width: 9px;
  height: 1.1em;
  margin-left: 2px;
  vertical-align: -0.18em;
  background: currentColor;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  50% { opacity: 0.15; }
}

@keyframes statusPulse {
  0% { background-position: 200% 0; }
  100% { background-position: -40% 0; }
}

@media (max-width: 1200px) {
  #app-shell {
    gap: 18px;
  }

  #left-col {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 100% !important;
    gap: 18px;
  }

  #mineru-col,
  #diffusion-col {
    flex: 1 1 calc(50% - 9px) !important;
    width: calc(50% - 9px) !important;
    min-width: 320px !important;
  }

  #left-col,
  #mineru-col,
  #diffusion-col {
    min-height: auto;
  }

  .replay-stream {
    height: auto;
    min-height: 680px;
  }

  #mineru-col,
  #diffusion-col {
    padding: 0;
  }
}

@media (max-width: 860px) {
  .gradio-container {
    padding-left: 12px !important;
    padding-right: 12px !important;
  }

  #app-shell {
    gap: 14px;
  }

  #left-col,
  #mineru-col,
  #diffusion-col {
    flex: 1 1 100% !important;
    width: 100% !important;
    min-width: 100% !important;
    border-radius: 18px;
  }

  .replay-stream {
    min-height: 560px;
    padding: 16px 16px 14px;
    border-radius: 18px;
  }

  .stream-head {
    flex: 0 0 48px !important;
    min-height: 48px !important;
    margin-bottom: 2px !important;
  }

  .name {
    font-size: clamp(20px, 5vw, 24px);
  }

  .stream-meta {
    flex: 0 0 auto !important;
    margin: 0 0 8px !important;
    gap: 10px;
    align-items: center;
  }

  .stream-status {
    font-size: clamp(18px, 4.5vw, 24px);
  }

  .stream-stats span {
    font-size: 13px;
  }

  .stream-progress {
    flex: 0 0 18px !important;
  }

  .stream-clock {
    flex: 0 0 34px !important;
  }

  .stream-clock strong {
    font-size: clamp(22px, 5vw, 30px) !important;
  }

  .stream-output {
    padding: 18px 16px !important;
    font-size: 12.5px;
    line-height: 1.65;
  }

  #prompt-grid .wrap {
    gap: 10px !important;
  }

  #prompt-grid label {
    height: 52px;
    font-size: 16px;
  }
}

@media (max-width: 560px) {
  .panel-title {
    font-size: 22px !important;
  }

  #left-col,
  #mineru-col,
  #diffusion-col {
    border-radius: 16px;
    padding: 14px;
  }

  #mineru-col,
  #diffusion-col {
    padding: 0;
  }

  .replay-stream {
    min-height: 500px;
    padding: 14px 14px 12px;
    border-radius: 16px;
  }

  .stream-meta {
    flex-wrap: wrap;
    justify-content: space-between;
  }

  .stream-stats {
    align-items: flex-start;
  }

  .stream-output {
    padding: 16px 14px !important;
  }

  #mineru-render-toggle,
  #diffusion-render-toggle {
    width: calc(100% - 28px) !important;
    max-width: calc(100% - 28px) !important;
    margin: 12px 14px 12px !important;
  }

  #prompt-grid .wrap {
    grid-template-columns: 1fr;
  }
}
"""


def _render_output(output_state: dict, done: bool) -> str:
    fragments = output_state["fragments"]
    if not fragments:
        suffix = "" if done else '<span class="cursor"></span>'
        return f'<div class="stream-output empty">Waiting for generation complete{suffix}</div>'

    body = "".join(
        f'<span class="{class_name}">{escape(text)}</span>'
        for class_name, text in fragments
    )
    cursor = "" if done else '<span class="cursor"></span>'
    return f'<div class="stream-output">{body}{cursor}</div>'


def _render_stats(stats: str) -> str:
    parts = [escape(part.strip()) for part in stats.split("/", 1)]
    if len(parts) == 1:
        parts.append("")
    return f'<div class="stream-stats"><span>{parts[0]}</span><span>{parts[1]}</span></div>'


def render_panel(title: str, state: dict, output_html: str | None = None) -> str:
    done = bool(state["done"])
    status_class = "stream-status done" if done else "stream-status"
    output_block_html = output_html or _render_output(state["output"], done)
    stats_html = _render_stats(state["stats"])

    return f"""
<section class="replay-stream">
  <div class="stream-head"><h2 class="name">{escape(title)}</h2></div>
  <div class="stream-meta">
    <div class="{status_class}">{escape(state["status"])}</div>
    {stats_html}
  </div>
  <div class="stream-progress">
    <div class="timeline"><div class="timeline-fill" style="width: {state["progress"]:.2f}%"></div></div>
  </div>
  <div class="stream-clock">
    <strong>{escape(state["clock"])}</strong>
  </div>
  {output_block_html}
</section>
"""


def build_initial_view() -> tuple[str, str]:
    mineru_panel = render_panel(
        "MinerU 2.5",
        build_panel_state(status="Ready", mode="append"),
    )
    diffusion_panel = render_panel(
        "MinerU-Diffusion",
        build_panel_state(status="Ready", mode="positional"),
    )
    return mineru_panel, diffusion_panel


def build_waiting_panel(title: str, status: str, mode: str) -> str:
    return render_panel(
        title,
        build_panel_state(status=status, stats="-- tokens / -- tok/s", mode=mode),
    )


def build_result_panel(title: str, result: dict, events: list[dict], active_duration: float, mode: str, local_time: float, status: str) -> str:
    return render_panel(
        title,
        build_panel_state(
            events=events,
            active_duration=active_duration,
            local_time=local_time,
            status=status,
            stats=format_stats(result),
            mode=mode,
        ),
    )


def build_rendered_panel(title: str, result: dict, active_duration: float, status: str, output_html: str) -> str:
    return render_panel(
        title,
        build_panel_state(
            active_duration=active_duration,
            local_time=active_duration,
            status=status,
            stats=format_stats(result),
            mode="append",
        ),
        output_html=output_html,
    )
