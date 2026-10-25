#!/usr/bin/env python3
"""Read-only validation of provenance, static links, and codebook/source parity."""
from pathlib import Path
import hashlib,json,re,subprocess
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
ROOT=Path(__file__).resolve().parents[1]
class Page(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.links=[];self.images=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and 'href' in a:self.links.append(a['href'])
        if tag in ('img','script') and 'src' in a:self.images.append(a['src'])
        if tag=='link' and 'href' in a:self.images.append(a['href'])
errors=[];cfg=json.loads((ROOT/'book.json').read_text());sources=json.loads((ROOT/'sources.json').read_text())
for key,s in sources.items():
    base=ROOT/('reference/DeepInfer' if (ROOT/'reference/DeepInfer').exists() else 'upstream/DeepInfer')
    file=base/s['path']
    if not file.exists():errors.append(f'missing source {key}');continue
    raw=file.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=s['sha256']:errors.append(f'changed source {key}')
    if not 1<=s['start_line']<=s['end_line']<=len(raw.decode().splitlines()):errors.append(f'bad lines {key}')
    if cfg['commit'] not in s['url']:errors.append(f'unpinned URL {key}')
pages={}
for file in (ROOT/'site').rglob('*.html'):
    page=Page();page.feed(file.read_text());pages[file.resolve()]=page
    if len(page.ids)!=len(set(page.ids)):errors.append(f'duplicate ids {file.name}')
    if '[[source:' in file.read_text() or '[[TRACE_' in file.read_text():errors.append(f'unexpanded marker {file.name}')
for file,page in pages.items():
    for url in page.links+page.images:
        parsed=urlsplit(url)
        if parsed.scheme or parsed.netloc:continue
        path=(file.parent/unquote(parsed.path)).resolve() if parsed.path else file
        if not path.exists():errors.append(f'broken asset/link {file.name}: {url}');continue
        if parsed.fragment and path in pages and unquote(parsed.fragment) not in pages[path].ids:errors.append(f'broken anchor {file.name}: {url}')
for ch in cfg['chapters']:
    if not (ROOT/'site'/f"{ch['slug']}.html").exists():errors.append('missing chapter '+ch['slug'])
codebook=(ROOT/'docs/12-code.md').read_text()
for file in (ROOT/'lessons').glob('*/code.py'):
    if file.read_text() not in codebook:errors.append(f'codebook mismatch {file.parent.name}')
    compile(file.read_text(),str(file),'exec')
reports={'source_anchors_checked':len(sources),'html_pages_checked':len(pages),'chapters':len(cfg['chapters']),'standalone_snapshots':len(list((ROOT/'lessons').glob('*/code.py'))),'source_and_codebook_integrity':'passed' if not errors else 'failed','errors':errors,'pandoc':subprocess.check_output(['pandoc','--version'],text=True).splitlines()[0],'upstream_executed':False,'device_inference_tested':False}
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/'reports/validation.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(reports,ensure_ascii=False,indent=2))
raise SystemExit(bool(errors))
