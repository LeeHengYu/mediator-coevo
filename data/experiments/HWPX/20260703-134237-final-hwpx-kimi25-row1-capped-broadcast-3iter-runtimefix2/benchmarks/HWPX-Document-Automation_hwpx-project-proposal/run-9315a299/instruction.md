# Task Instruction

You must produce `/root/project_proposal_ready.hwpx` from the template and JSON data. Follow these steps exactly:

### 1 — Inspect inputs
```bash
cd /root
ls -la
cat project_proposal.json
```
Unzip the template to examine its structure:
```bash
mkdir -p /tmp/hwpx_template
cp project_proposal_template.hwpx /tmp/hwpx_template/template.zip
cd /tmp/hwpx_template
unzip template.zip -d template_contents
find template_contents -type f | sort
```
Read every XML section file (typically `Contents/section*.xml`) to understand the placeholder layout and XML tag structure:
```bash
for f in template_contents/Contents/section*.xml; do echo "=== $f ==="; cat "$f"; echo; done
```
Also inspect any other XML files (e.g., `content.hpf`, `header.xml`) for possible placeholders.

### 2 — Understand the contracts
- Every `{{...}}` placeholder in any XML file must be replaced with the corresponding value from the JSON.
- Placeholders may be **fragmented across multiple `<hp:t>` tags** within one `<hp:run>` or `<hp:p>`. You must handle this: collect all `<hp:t>` text within a paragraph/run, find placeholders spanning tags, and after replacement consolidate or rewrite the tags so no `{{` or `}}` fragments remain.
- **Budget normalization**: If a JSON value contains commas in a number (e.g., `₩50,000,000`), remove the commas but keep the currency symbol (e.g., `₩50000000`).
- **Phase month spans**: For lines containing `단계1`, `단계2`, or `단계3`, compute the month span from the date range written on that same line (format like `2025.01 ~ 2025.03`). Calculate months = (end_year*12+end_month) - (start_year*12+start_month). Append ` (N개월)` to the end of that line's text (in the last `<hp:t>` of the paragraph).
- **Layout cache removal**: For every `<hp:p>` element whose text content you modify, remove any `<hp:lineSegArray>` child element (and its descendants). This prevents overlapping-character rendering issues.
- **Preserve all Korean labels and static note lines unchanged.**

### 3 — Write and run a Python script
Write a single Python script `/tmp/process_hwpx.py` that:
1. Copies the template to a working zip.
2. Extracts it.
3. Loads the JSON data.
4. For each XML file in `Contents/`:
   a. Parses the raw XML text.
   b. For each `<hp:p>` block, collects all `<hp:t>` text, joins it, checks for `{{...}}` patterns.
   c. If a placeholder is found spanning multiple `<hp:t>` tags, consolidates those tags' text into one `<hp:t>` (or rewrites them) and performs the replacement.
   d. Applies budget normalization (remove commas from numeric values while keeping currency symbol).
   e. For paragraphs containing `단계1`, `단계2`, or `단계3`, finds the date range pattern (e.g., `2025.01 ~ 2025.03`), computes month difference, and appends ` (N개월)` to the last `<hp:t>` in that paragraph.
   f. For any modified `<hp:p>`, removes `<hp:lineSegArray>` elements.
   g. Writes the modified XML back.
5. Re-zips the contents into `/root/project_proposal_ready.hwpx`.
6. As a final validation step, re-extracts the output hwpx and scans ALL XML files for any remaining `{{` or `}}` patterns, printing PASS or FAIL.

Run the script:
```bash
python3 /tmp/process_hwpx.py
```

### 4 — Verify
```bash
# Check the output exists and is a valid zip
file /root/project_proposal_ready.hwpx
python3 -c "import zipfile; z=zipfile.ZipFile('/root/project_proposal_ready.hwpx'); print('Valid zip, entries:', z.namelist())"

# Scan for leftover placeholders
mkdir -p /tmp/verify
cd /tmp/verify && unzip -o /root/project_proposal_ready.hwpx
grep -r '{{' . || echo 'No leftover {{ found'
grep -r '}}' . || echo 'No leftover }} found'

# Print section XMLs to visually confirm replacements, month spans, budget format
for f in Contents/section*.xml; do echo "=== $f ==="; cat "$f"; echo; done
```

### 5 — Run the verifier if present
```bash
cd /root
if [ -f test_output.py ]; then python3 -m pytest test_output.py -v; fi
if [ -d tests ]; then python3 -m pytest tests/ -v; fi
```

### Key pitfalls to avoid
- Do NOT assume placeholders fit neatly in one `<hp:t>` tag. Always handle fragmentation.
- Do NOT leave `{{` or `}}` fragments in any tag.
- Do NOT forget to remove `<hp:lineSegArray>` from modified paragraphs.
- Do NOT add commas back into normalized budget values.
- Make sure the month calculation is correct: e.g., 2025.01 ~ 2025.03 = 3-1 = 2? No — count inclusive months or use the difference. Check: Jan to Mar is 3 months if inclusive of start month, or 2 months if pure subtraction. The benchmark says 단계1 -> (3개월), 단계2 -> (3개월), 단계3 -> (1개월). So verify your formula against the actual dates in the template. The expected values suggest the formula is `end_month - start_month + 1` when same year, or `(end_year-start_year)*12 + end_month - start_month + 1` cross-year. But FIRST check the actual date ranges in the template to confirm which formula yields the expected 3, 3, 1 results. If the dates don't match those expected outputs with any simple formula, just read the dates and compute carefully.
- Ensure the re-zipped hwpx preserves the original directory structure exactly.

# Executor Policy

---
name: executor
description: Portable executor policy for workflow, verification, resource use, and failure handling across task runtimes.
---

## Executor Policy

Use this skill as execution policy, not as domain-specific task knowledge. When
task-local curated skills or resources are available, prefer them for domain
details and use this policy for workflow control.

## Task Execution

1. Read the task instruction, task resources, and verifier contract before editing.
2. Identify the scoring mechanism and the smallest command that can reproduce the
   failure or verify the expected behavior.
3. Inspect existing files and task-local resources before making changes.
4. Make the smallest source change that satisfies the task and verifier contract.
5. Keep a compact record of the concrete evidence behind the change: observed
   failure, files inspected, edit made, and verifier result.
6. Run targeted verification before broad verification when practical.

## File Editing

1. Read the actual current file contents immediately before making any edit.
   Never rely on memory, prior snapshots, or assumed content.
2. Prefer direct in-place edits over patch or diff application when the exact
   current context is uncertain.
3. If using a patch or diff, confirm that every context line exists verbatim in
   the file before applying it.
4. If a patch hunk fails to apply, re-read the affected file region and perform
   the edit directly instead of retrying the same patch.
5. After any edit, re-read the affected region to confirm the change landed.

## Build and Test Fixes

When a task requires fixing a broken build, failing test, or generated artifact:

1. Run the relevant build, test, or verifier command first to capture the
   baseline failure.
2. Identify the specific error message, file, line, or expected output before
   editing.
3. Apply the smallest fix, then re-run the same targeted command.
4. Treat newly introduced failures as separate sub-tasks and resolve them in
   order.
5. Do not mark the task complete until the verifier-relevant command succeeds or
   the remaining failure is clearly outside the task boundary.

## Artifact-Contract Handling

Do not treat artifacts as ordinary text files. Treat them as contract-bearing
interfaces between input data, generated output, verifier checks, and downstream
consumers.

When a task requires reading, modifying, or generating an artifact such as JSON,
DOT, reports, configs, generated source, schemas, datasets, or parsed outputs:

1. Identify the artifact contract first: format, schema, required fields,
   identifiers, references, ordering, examples, verifier assertions, and
   consuming code.
2. Inspect representative source artifacts directly before deciding how to
   transform or preserve them.
3. Determine whether the task calls for preservation, transformation, repair,
   generation, or validation.
4. Preserve required literals, identifiers, references, ordering, and
   representative content unless the contract explicitly requires a change.
5. Do not invent, drop, rename, normalize, collapse, expand, or repair artifact
   elements unless the verifier or consumer contract requires that behavior.
6. Prefer structured parsers, serializers, validators, or existing consumer code
   over ad hoc string manipulation when they are available.
7. After producing the artifact, run targeted checks for parseability, required
   keys or IDs, reference consistency, expected counts, preserved content, and
   format-specific validity.
8. If targeted checks regress or become unusable after a change, stop expanding
   the solution. Re-inspect the source contract and narrow the edit before trying
   a broader repair.

A plausible-looking artifact is not sufficient evidence. The artifact is only
correct when it satisfies the task contract under the verifier or consuming
code.

## Constraints

- Do not bypass, remove, or weaken tests, verifier scripts, fixtures, or expected
  output checks.
- Do not treat this policy as overriding task-specific instructions or verifier
  requirements.
- On tool or environment errors, retry once when the retry is safe, then report
  the failure with the command and error output.
- On ambiguous instructions, make a conservative assumption and continue.

# Task Resources

Inspect the task files, environment, tests, and expected outputs directly.

# Verifier Contract

Success is judged by the SkillFlow verifier for this task.
Do not bypass, remove, or weaken verifier scripts, tests, fixtures, or expected-output checks.
Run the provided tests or verifier command when practical before finalizing.
Task metadata: author_email=catpaw@example.com, author_name=CatPaw Task Engineer, category=document-editing, difficulty=medium, tags=[hwpx, xml-editing, document-processing, latent-method-reuse].
Verifier config: timeout_sec=600.0.