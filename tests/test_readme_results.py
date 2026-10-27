import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_results_match_sample_analysis():
    """Verify that README results table agrees with examples/sample_analysis.json."""
    analysis_file = ROOT / "examples" / "sample_analysis.json"
    readme_file = ROOT / "README.md"
    assert analysis_file.exists(), "sample_analysis.json must exist"
    assert readme_file.exists(), "README.md must exist"

    # Run scripts/render_results.py in check mode
    script = ROOT / "scripts" / "render_results.py"
    assert script.exists(), "scripts/render_results.py must exist"

    proc = subprocess.run([sys.executable, str(script), "--check"],
                          cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"README results table is out of date:\n{proc.stdout}\n{proc.stderr}"
