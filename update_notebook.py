import json
import io

with io.open("c:/Users/faris3/MagicFace/process.ipynb", "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        src = cell.get('source', [])
        for i, line in enumerate(src):
            if "Testing AU12 (Slight Smile)" in line:
                src[i] = line.replace("Testing AU12 (Slight Smile)", "Testing AU4+AU5 (Angry)")
    elif cell.get('cell_type') == 'code':
        src = cell.get('source', [])
        for i, line in enumerate(src):
            if "inference.py" in line and "AU12" in line:
                src[i] = line.replace("AU12", "AU4+AU5").replace("5 --saved_path", "5+5 --saved_path")
            if "Edited Result (AU12)" in line:
                src[i] = line.replace("Edited Result (AU12)", "Edited Result (Angry)")

with io.open("c:/Users/faris3/MagicFace/process.ipynb", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)
