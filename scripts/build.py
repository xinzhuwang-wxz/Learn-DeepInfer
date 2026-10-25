#!/usr/bin/env python3
"""Small stdlib orchestration around installed Pandoc; never imports upstream code."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def render(text: str) -> str:
    return subprocess.run(['pandoc', '-f', 'markdown+fenced_divs+link_attributes', '-t', 'html5', '--section-divs', '--highlight-style=pygments'], input=text, text=True, capture_output=True, check=True).stdout

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', default='site')
    args = parser.parse_args()
    cfg = json.loads((ROOT / 'book.json').read_text())
    sources = json.loads((ROOT / 'sources.json').read_text()) if (ROOT / 'sources.json').exists() else {}
    out = ROOT / args.site
    out.mkdir(exist_ok=True, parents=True)
    shutil.copytree(ROOT / 'assets', out / 'assets', dirs_exist_ok=True)
    if (ROOT/'lessons').exists():
        shutil.copytree(ROOT/'lessons', out/'downloads/lessons', ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    shutil.copytree(ROOT/'licenses', out/'licenses', dirs_exist_ok=True)
    shutil.copy2(ROOT/'LICENSE',out/'LICENSE.txt')
    shutil.copy2(ROOT/'NOTICE.md',out/'NOTICE.txt')
    for provenance in ('sources.json','upstream.lock.json'):
        if (ROOT/provenance).exists():shutil.copy2(ROOT/provenance,out/provenance)
    for folder in ('examples', 'reference', 'licenses'):
        shutil.copytree(ROOT/folder, out/'downloads'/folder, ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    # Markdown downloads use the same source text, with adjacent SVG assets.
    (out/'downloads/markdown').mkdir(parents=True,exist_ok=True)
    (out/'downloads/assets').mkdir(parents=True,exist_ok=True)
    for asset in (ROOT/'assets').glob('*.svg'):
        shutil.copy2(asset,out/'downloads/assets'/asset.name)
    # Old local exports must never leak into a subsequent default site build.
    for stale_pdf in (out/'downloads').glob('*.pdf'):
        stale_pdf.unlink()
    chapters = cfg['chapters']
    records, bodies, canon = [], [], []
    for ch in chapters:
        raw = (ROOT / ch['file']).read_text()
        raw = re.sub(r'\[\[source:([^\]]+)\]\]', lambda m: '[源码 · '+sources[m[1]]['label']+']('+sources[m[1]]['url']+')', raw)
        (out/'downloads/markdown'/Path(ch['file']).name).write_text(raw)
        def local_link(match):
            label, target = match.groups()
            if target.startswith(('https:', 'http:', '#')):
                return match.group(0)
            path, _, fragment = target.partition('#')
            if path.startswith('../assets/'):
                mapped = path.removeprefix('../')
            elif path.startswith('../'):
                relative = path.removeprefix('../')
                if Path(relative).suffix in ('.py', '.hpp', '.cpp'):
                    source = ROOT/relative
                    viewer = out/'source'/Path(relative + '.html')
                    viewer.parent.mkdir(parents=True, exist_ok=True)
                    lines = ''.join('<span class="source-line" id="L'+str(i)+'"><a href="#L'+str(i)+'">'+str(i)+'</a> '+html.escape(line)+'</span>\n' for i,line in enumerate(source.read_text().splitlines(), 1))
                    viewer.write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(relative)+'</title><style>body{font:16px monospace;background:#f5faf8;color:#183b36}pre{line-height:1.65}.source-line:target{background:#fff1a8}.source-line a{color:#54756c}a{scroll-margin-top:30px}</style><h1>'+html.escape(relative)+'</h1><pre>'+lines+'</pre></html>')
                    mapped = 'source/'+relative+'.html'
                    fragment = fragment.split('-')[0]
                else:
                    mapped = 'downloads/'+relative
            else:
                mapped = next((c['slug']+'.html' for c in chapters if Path(c['file']).name == path), path)
            return '['+label+']('+mapped+('#'+fragment if fragment else '')+')'
        page_md = re.sub(r'\[([^\]]*)\]\(([^\s)]+)\)', local_link, raw)
        page_md = page_md.replace('<!-- TRACE_EXPLORER -->', (ROOT/'assets/trace-explorer.html').read_text())
        body = render(page_md)
        records.append({**ch, 'url': ch['slug']+'.html', 'text': re.sub(r'<[^>]+>', ' ', body)})
        bodies.append(body)
        canon.append(raw)
    (out/'assets/search-data.js').write_text('window.LEARN_SEARCH = '+json.dumps(records, ensure_ascii=False)+';\n')
    nav = ''.join('<a class="nav-link" data-chapter="'+html.escape(ch['slug'])+'" href="'+ch['slug']+'.html"><span>'+ch.get('number',f'{i:02d}')+'</span>'+html.escape(ch['title'])+'</a>' for i,ch in enumerate(chapters))
    def shell(title: str, body: str, active: str = '', pager: str = '') -> str:
        reading = '<a href="book.html">整本阅读 ↗</a>'
        return '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="'+html.escape(cfg['description'])+'"><title>'+html.escape(title+' · '+cfg['title'])+'</title><link rel="stylesheet" href="assets/style.css"><script defer src="assets/search-data.js"></script><script defer src="assets/site.js"></script><script defer src="assets/trace-data.js"></script><script defer src="assets/trace.js"></script></head><body data-book="'+html.escape(cfg['title'])+'" data-active="'+active+'"><a class="skip" href="#main">跳至正文</a><header class="topbar"><a class="brand" href="index.html"><span class="mark">L/</span>'+html.escape(cfg['title'])+'</a><div class="top-actions">'+reading+'<button id="theme" aria-label="切换深浅主题">◐</button><button id="menu" aria-controls="sidebar" aria-expanded="false">目录</button></div></header><aside id="sidebar"><div class="nav-label">LEARNING PATH / 学习路径</div><nav aria-label="章节">'+nav+'</nav><div class="source-lock">SOURCE LOCK<br><a href="'+cfg['repo']+'/tree/'+cfg['commit']+'">'+cfg['commit'][:12]+'</a><br>教学实现 ≠ 生产运行</div></aside><main id="main"><div class="search-wrap"><label for="search">搜索课程</label><input id="search" type="search" placeholder="输入关键词，如 调度、缓存…  /" autocomplete="off"><div id="search-results" aria-live="polite"></div></div>'+body+pager+'<footer>独立教学项目，非上游官方教程。正文、可运行代码与固定版本来源。<br>Source-locked, inspectable, deliberately small. <a href="LICENSE.txt">MIT 许可</a> · <a href="NOTICE.txt">来源声明</a> · <a href="licenses/DeepInfer-Apache-2.0.txt">上游 Apache-2.0</a></footer></main></body></html>'
    cards = ''.join('<a class="chapter-card" href="'+ch['slug']+'.html"><span class="card-num">'+ch.get('number',f'{i:02d}')+'</span><div><h2>'+html.escape(ch['title'])+'</h2><p>'+html.escape(ch.get('summary',''))+'</p><small>'+html.escape(ch.get('time',''))+'</small></div><span class="arrow">↗</span></a>' for i,ch in enumerate(chapters))
    intro = '<section class="hero"><p class="eyebrow">SOURCE → CONCEPT → EXPERIMENT → EVIDENCE</p><h1>'+html.escape(cfg['title'])+'</h1><p class="hero-tagline">'+html.escape(cfg['tagline'])+'</p><p>'+html.escape(cfg['description'])+'</p><div class="hero-meta"><span>'+str(len(chapters))+' 个学习单元</span><span>无模型下载的 CPU 实验</span><span>固定源码版本</span></div></section><section class="chapter-grid" aria-label="课程">'+cards+'</section>'
    (out/'index.html').write_text(shell('课程首页',intro))
    for i,(ch,body) in enumerate(zip(chapters,bodies)):
        pager='<div class="lesson-actions"><button class="complete" data-complete="'+ch['slug']+'">标记学完</button><span class="progress" aria-live="polite"></span></div><nav class="pager" aria-label="前后章节">'
        if i: pager+='<a href="'+chapters[i-1]['slug']+'.html">← '+html.escape(chapters[i-1]['title'])+'</a>'
        if i+1<len(chapters): pager+='<a href="'+chapters[i+1]['slug']+'.html">'+html.escape(chapters[i+1]['title'])+' →</a>'
        pager+='</nav>'
        download = '<div class="lesson-downloads"><a href="downloads/markdown/'+Path(ch['file']).name+'" download>下载本章 Markdown</a></div>'
        if ch.get('lesson'):
            base = 'downloads/lessons/' + ch['lesson'] + '/'
            download += '<div class="lesson-downloads"><a href="'+base+'code.py" download>下载本课完整 code.py</a><a href="'+base+'delta.patch" download>相对上一课的代码差异</a></div>'
        (out/(ch['slug']+'.html')).write_text(shell(ch['title'],download+'<article>'+body+'</article>',ch['slug'],pager))
    toc='<section class="book-title"><p class="eyebrow">'+html.escape(cfg['title'])+'</p><h1>'+html.escape(cfg['tagline'])+'</h1><p>'+html.escape(cfg['description'])+'</p><p>基线 '+cfg['commit']+'<br>编制与核验日期：2026-10-25<br>原创讲解与教学代码；上游来源及许可见附录。</p></section><section class="book-toc"><h1>目录</h1><ol>'+''.join('<li><a href="#chapter-'+ch['slug']+'">'+html.escape(ch['title'])+'</a></li>' for ch in chapters)+'</ol></section>'
    def book_body(ch, body):
        prefix = 'bk-' + ch['slug'] + '-'
        body = re.sub(r'id="([^"]+)"', lambda m: 'id="'+(m[1] if m[1].startswith('trace-') else prefix+m[1])+'"', body)
        body = re.sub(r'href="#([^"]+)"', lambda m: 'href="#'+prefix+m[1]+'"', body)
        # Interactive control IDs are not used in print; preserve the same static evidence.
        return body
    book=toc+''.join('<article class="book-chapter" id="chapter-'+ch['slug']+'">'+book_body(ch,body)+'</article>' for ch,body in zip(chapters,bodies))
    (out/'book.html').write_text(shell('完整教程',book))
    (ROOT/'output').mkdir(exist_ok=True)
    (ROOT/'output/book.md').write_text('\n\n'.join(canon))
    manifest={'title':cfg['title'],'upstream_commit':cfg['commit'],'chapter_count':len(chapters),'canonical_sha256':hashlib.sha256(('\n\n'.join(canon)).encode()).hexdigest(),'chapter_files':[ch['file'] for ch in chapters]}
    (out/'build-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
