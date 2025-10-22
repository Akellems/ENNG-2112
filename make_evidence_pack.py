
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤
# File: make_evidence_pack.py
# Purpose: Gather metrics & figures into evidence/ with an index README
# Output: evidence/README.md
# ©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤©¤

import os, shutil
from pathlib import Path

SECTIONS = [
    ('metrics', ['metrics_baseline.json','metrics_cross_dataset.json','metrics_calibration.json','unsure_sweep.json','latency_size_comparison.csv']),
    ('evidence/heatmaps', []),
    ('evidence/feature_maps', []),
]

def main():
    ev = Path('evidence'); ev.mkdir(exist_ok=True)
    lines = ["# Evidence Pack\n"]
    for folder, files in SECTIONS:
        p = Path(folder)
        if p.exists():
            dest = ev / p.name
            if p.is_dir():
                # copy tree selectively
                if not dest.exists():
                    shutil.copytree(p, dest)
                lines.append(f"- {p}/ -> evidence/{p.name}/")
            else:
                dest_file = ev / p.name
                shutil.copy2(p, dest_file)
                lines.append(f"- {p} -> evidence/{p.name}")
        for f in files:
            src = Path(folder)/f
            if src.exists():
                shutil.copy2(src, ev/src.name)
                lines.append(f"- {src} -> evidence/{src.name}")
    (ev/"README.md").write_text("\n".join(lines), encoding='utf-8')
    print('Evidence pack compiled -> evidence/README.md')

if __name__ == '__main__':
    main()
