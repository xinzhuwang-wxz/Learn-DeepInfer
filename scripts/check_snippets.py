#!/usr/bin/env python3
"""Check marked Python excerpts against committed sources without rewriting either.

Only layout, parentheses, and comments may differ: the parsed AST must match.
This detects stale excerpts, not correctness of the lesson or its explanation.
"""
from __future__ import annotations
import ast
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parents[1]
MARKER = re.compile(
    r'<!-- code: ([^\s]+):L(\d+)-L(\d+) -->\s*'
    r'```(python|cpp)\n(.*?)\n```', re.S
)


def check(paths):
    errors, entries = [], []
    for chapter in paths:
        text = chapter.read_text()
        matches = list(MARKER.finditer(text))
        if text.count('<!-- code:') != len(matches):
            errors.append(f'{chapter.name}: malformed code marker')
        for match in matches:
            relative, first, last, language, excerpt = match.groups()
            first, last = int(first), int(last)
            source = (ROOT / relative).resolve()
            if not source.is_relative_to(ROOT) or not source.is_file():
                errors.append(f'{chapter.name}: invalid source {relative}')
                continue
            lines = source.read_text().splitlines()
            if not 1 <= first <= last <= len(lines):
                errors.append(f'{chapter.name}: invalid range {relative}:{first}-{last}')
                continue
            original = textwrap.dedent('\n'.join(lines[first - 1:last]))
            try:
                if language == 'python':
                    expected = ast.dump(ast.parse(original), include_attributes=False)
                    actual = ast.dump(ast.parse(textwrap.dedent(excerpt)), include_attributes=False)
                else:
                    expected = original.strip()
                    actual = textwrap.dedent(excerpt).strip()
                if expected != actual:
                    errors.append(f'{chapter.name}: stale excerpt {relative}:{first}-{last}')
            except SyntaxError as exc:
                errors.append(f'{chapter.name}: unparsable excerpt {relative}:{first}-{last}: {exc}')
            entries.append({'chapter': str(chapter.relative_to(ROOT)), 'source': relative,
                            'start_line': first, 'end_line': last, 'language': language})
    return errors, entries


if __name__ == '__main__':
    errors, entries = check(sorted((ROOT / 'docs').glob('*.md')))
    print(json.dumps({'status': 'failed' if errors else 'passed',
                      'marked_excerpts_checked': len(entries), 'errors': errors},
                     ensure_ascii=False, indent=2))
    raise SystemExit(bool(errors))
