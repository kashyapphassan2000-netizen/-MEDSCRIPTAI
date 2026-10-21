"""Build outputs/reports/REPORT.md (paste it into your Hugging Face model card and your CV/portfolio)."""
import json
from pathlib import Path
from common import OUT

reports = OUT / "reports"
metrics = json.loads((reports / "metrics.json").read_text())
cols = ["n", "accuracy", "format_ok", "json_valid", "rougeL", "bertscore_f1"]
lines = ["# MedScriptAI evaluation report", "", "## Automatic metrics", "",
         "| task/model | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
for name, m in metrics.items():
    lines.append(f"| {name} | " + " | ".join(str(m.get(c, "-")) for c in cols) + " |")
lines += ["", "## LLM-as-a-judge", ""]
for path in sorted(reports.glob("judge_*.json")):
    lines.append(f"- **{path.stem.removeprefix('judge_')}**: `{json.dumps(json.loads(path.read_text()))}`")
lines += ["", "Judge: Qwen2.5-14B-Instruct. Pairwise results use position swapping (a win must win in both orders).",
          "", "Built with Llama. Educational use only; not a medical device and not medical advice."]
(reports / "REPORT.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
