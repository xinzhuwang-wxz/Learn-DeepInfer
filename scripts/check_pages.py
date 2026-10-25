#!/usr/bin/env python3
"""Check root and project-subpath static hosting without starting a browser."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, unquote
import argparse, json, re

ROOT = Path(__file__).resolve().parents[1]

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []
    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ('href', 'src') and value:
                self.urls.append(value)

def check_site(site):
    errors, checked = [], 0
    pages = sorted(site.rglob('*.html'))
    if not pages:
        return ['No built HTML pages; run make build first.'], 0
    for prefix in ('/', '/Learn-DeepInfer/'):
        origin = 'https://example.invalid' + prefix
        for page in pages:
            parser = Links(); parser.feed(page.read_text())
            for url in parser.urls:
                if urlsplit(url).scheme or urlsplit(url).netloc:
                    continue
                checked += 1
                resolved = urlsplit(urljoin(origin + page.relative_to(site).as_posix(), url))
                if not resolved.path.startswith(prefix) or url.startswith('/'):
                    errors.append(f'Not project-relative: {page.name}: {url}')
                    continue
                local = site / unquote(resolved.path[len(prefix):])
                if not local.is_file():
                    errors.append(f'Missing asset: {page.name}: {url}')
        for css in (site/'assets').glob('*.css'):
            for url in re.findall(r'url\([\'"]?([^\)\'\"]+)', css.read_text()):
                if url.startswith('/'):
                    errors.append(f'Root-relative CSS URL: {url}')
    if list(site.rglob('*.pdf')):
        errors.append('Default web output must not include a stale/local PDF.')
    for path in pages:
        if '下载 PDF' in path.read_text():
            errors.append(f'Stale PDF download in {path.name}')
    data = (site/'assets/search-data.js').read_text()
    records = json.loads(data.removeprefix('window.LEARN_SEARCH = ').strip().removesuffix(';'))
    for record in records:
        if urlsplit(record['url']).scheme or record['url'].startswith('/') or not (site/record['url']).is_file():
            errors.append('Invalid project-relative search target: ' + record['url'])
    return errors, checked

def check_markdown():
    errors, checked = [], 0
    for path in [ROOT/'README.md', *sorted((ROOT/'docs').glob('*.md'))]:
        text = re.sub(r'```.*?```', '', path.read_text(), flags=re.S)
        if '[[source:' in text:
            errors.append(f'Unrendered source marker: {path.name}')
        for url in re.findall(r'\]\(([^\s\)]+)(?:\s+"[^"]*")?\)', text):
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            checked += 1
            local = (path.parent/unquote(parsed.path)).resolve()
            if not local.exists():
                errors.append(f'Broken Markdown path: {path.name}: {url}')
    return errors, checked


def check_workflow():
    errors = []
    lock = json.loads((ROOT/'.github/actions-lock.json').read_text())['actions']
    workflow = (ROOT/'.github/workflows/pages.yml').read_text()
    used = dict(re.findall(r'uses:\s*([\w-]+/[\w-]+)@([0-9a-f]{40})', workflow))
    if len(re.findall(r'uses:', workflow)) != len(used):
        errors.append('Every Action must use a unique full commit pin.')
    if used != {name: item['commit'] for name, item in lock.items()}:
        errors.append('Actions do not match the official source lock.')
    if 'pull_request_target:' in workflow or 'secrets.' in workflow:
        errors.append('Unexpected privileged PR trigger or extra secret.')
    if 'make check' not in workflow or 'persist-credentials: false' not in workflow:
        errors.append('Missing required checks or credential boundary.')
    return errors

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', default='site')
    args = parser.parse_args()
    errors, links = check_site(ROOT/args.site)
    md_errors, md_links = check_markdown(); errors += md_errors
    errors += check_workflow()
    report = {'status':'passed' if not errors else 'failed', 'hosting_prefixes':['/', '/Learn-DeepInfer/'], 'html_local_links_checked':links, 'markdown_local_links_checked':md_links, 'official_action_pins_checked':5, 'pdf_in_default_site':False if not list((ROOT/args.site).rglob('*.pdf')) else True, 'errors':errors}
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/pages-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    raise SystemExit(bool(errors))
