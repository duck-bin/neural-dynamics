"""Inline data/*.json into web/viewer_template.html -> a self-contained web/index.html.

Self-contained (data embedded, no fetch, no external libs) so it works identically
as a local file, on Hugging Face Spaces / GitHub Pages, and as a Claude Artifact
(whose CSP blocks external requests).
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
tpl = (REPO / "web" / "viewer_template.html").read_text()
brain = (REPO / "data" / "brain.json").read_text()
rnn = (REPO / "data" / "rnn.json").read_text()

body = tpl.replace("__BRAIN_JSON__", brain).replace("__RNN_JSON__", rnn)

html = (
    "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<meta name=\"color-scheme\" content=\"dark\">\n"
    "<title>Motor Cortex Dynamics — jPCA &amp; Fixed Points</title>\n"
    "</head>\n<body>\n" + body + "\n</body>\n</html>\n"
)
out = REPO / "web" / "index.html"
out.write_text(html)
print(f"wrote {out} ({len(html)//1024} KB)")

# body-only fragment for the Claude Artifact (skeleton added at publish time)
frag = REPO / "web" / "artifact_body.html"
frag.write_text(body)
print(f"wrote {frag} ({len(body)//1024} KB)")
