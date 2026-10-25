.PHONY: test build verify check package pdf regenerate-lessons regenerate-evidence regenerate-diagrams

test:
	python -m unittest discover -s tests -v

build:
	python scripts/check_snippets.py
	python scripts/build.py

# Explicit authoring commands. These overwrite the named checked-in artifacts;
# normal build/check/Pages never runs them or overwrites learner edits.
regenerate-lessons:
	@echo "Rewriting all lessons/*/code.py from the snapshot generator."
	python scripts/make_lessons.py

regenerate-evidence:
	@echo "Rewriting deltas, trace assets, docs/09-evidence.md and docs/12-code.md."
	python scripts/make_evidence.py

regenerate-diagrams:
	@echo "Rewriting generated SVG assets."
	python scripts/make_diagrams.py
	python scripts/make_async_diagrams.py
	python scripts/make_core_diagrams.py
	python scripts/make_state_diagrams.py
	python scripts/make_compute_diagrams.py

verify:
	python scripts/verify.py
	python scripts/check_pages.py

check: build test verify

package: check
	python scripts/package.py
	python scripts/package_offline.py

# Optional local export only. Neither CI nor the default site uses this PDF.
pdf: build
	node scripts/export_pdf.mjs
	python scripts/normalize_pdf_text.py output/pdf/Learn-DeepInfer.pdf
