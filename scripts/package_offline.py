"""Package the validated website and complete rebuildable source together."""
from pathlib import Path
import hashlib
import json
import zipfile
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'output/Learn-DeepInfer-source.zip'
if not source.is_file():
    raise SystemExit('Run scripts/package.py first')
entries=[(p, p.relative_to(ROOT/'site').as_posix()) for p in (ROOT/'site').rglob('*') if p.is_file()]
entries.append((source, 'source/Learn-DeepInfer-source.zip'))
out=ROOT/'output/Learn-DeepInfer-offline.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p,name in sorted(entries,key=lambda v:v[1]):
        if p.is_symlink() or p.suffix=='.pdf':
            raise SystemExit('Unexpected file: '+str(p))
        info=zipfile.ZipInfo('Learn-DeepInfer/'+name,(2026,10,25,0,0,0))
        info.compress_type=zipfile.ZIP_DEFLATED
        info.external_attr=0o100644<<16
        archive.writestr(info,p.read_bytes(),compresslevel=6)
manifest={'archive':out.name,'files':len(entries),'bytes':out.stat().st_size,
          'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
          'entry':'Learn-DeepInfer/index.html',
          'markdown':'Learn-DeepInfer/downloads/markdown/',
          'source':'Learn-DeepInfer/source/Learn-DeepInfer-source.zip',
          'scope':'Offline website, Markdown, code, fixed-version source references; no GPU execution claim'}
(ROOT/'output/package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
