#!/usr/bin/env python3
"""Build a deterministic public source archive from an explicit allowlist."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/Learn-DeepInfer-source.zip'
ROOT_FILES = ('.gitignore', 'LICENSE', 'NOTICE.md', 'README.md', 'Makefile',
              'book.json', 'sources.json', 'upstream.lock.json')
SOURCE_DIRS = ('.github', 'assets', 'docs', 'examples', 'lessons', 'licenses',
               'reference', 'scripts', 'tests')

# A reference subset must be present in a clean clone. Never download, import,
# execute, or silently re-lock an upstream file during packaging.
sources = json.loads((ROOT/'sources.json').read_text())
for source in sources.values():
    path = ROOT/'reference/DeepInfer'/source['path']
    if hashlib.sha256(path.read_bytes()).hexdigest() != source['sha256']:
        raise SystemExit('Missing or changed source reference: ' + str(path))

files = [ROOT/name for name in ROOT_FILES]
for name in SOURCE_DIRS:
    for path in (ROOT/name).rglob('*'):
        if path.is_symlink():
            raise SystemExit('Symlinks are not included: ' + str(path))
        if path.is_file() and '__pycache__' not in path.parts and path.suffix not in ('.pyc', '.pdf'):
            files.append(path)
OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(files):
        relative = path.relative_to(ROOT)
        info = zipfile.ZipInfo((Path('Learn-DeepInfer')/relative).as_posix(),
                               (2026, 10, 25, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o100644 << 16
        archive.writestr(info, path.read_bytes(), compresslevel=6)
manifest = {
    'archive': OUT.name,
    'files': len(files),
    'bytes': OUT.stat().st_size,
    'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest(),
    'contains': 'Markdown, cumulative Python lessons, assets, tests, build scripts, GitHub Pages workflow, source locks and audited Apache reference subset',
    'excluded': 'generated site, PDFs, output, reports, full upstream checkout, research archives, .git and caches',
}
(ROOT/'output/source-package-manifest.json').write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
