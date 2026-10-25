"""Read-only evidence audit: hashes and ranges, never import upstream code."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def audit():
    sources = json.loads((ROOT / 'sources.json').read_text())
    commit = json.loads((ROOT / 'book.json').read_text())['commit']
    checked = []
    for key, item in sources.items():
        path = ROOT / 'reference' / 'DeepInfer' / item['path']
        content = path.read_bytes()
        assert hashlib.sha256(content).hexdigest() == item['sha256'], key
        lines = content.decode().splitlines()
        assert 1 <= item['start_line'] <= item['end_line'] <= len(lines), key
        assert '/blob/' + commit + '/' in item['url'], key
        checked.append(key)
    return {'anchors': len(checked), 'hashes_and_ranges': 'passed',
            'upstream_imported': False, 'gpu_correctness': 'not tested'}

if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
