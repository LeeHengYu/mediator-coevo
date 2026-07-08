# Task Instruction

## Task: Prepare Warehouse Safety Audit Brief (HWPX)

You need to fill in a template HWPX document using data from two JSON files and save the result.

### Step-by-step Plan

#### 1. Inspect the workspace
```bash
find /root -maxdepth 3 -type f | head -60
ls -la /root/
```
Identify the location of `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

#### 2. Read the JSON data files
```bash
cat <path>/audit_overview.json
cat <path>/corrective_actions.json
```
Understand all fields and values you'll need to substitute.

#### 3. Unzip the HWPX template
HWPX is a ZIP-based format containing XML files.
```bash
mkdir -p /tmp/hwpx_work
cp <path>/safety_audit_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
```

#### 4. Inspect the XML content files
```bash
find /tmp/hwpx_work/template_contents -type f
```
Look for the main content XML (likely `Contents/section0.xml` or similar). Read it fully:
```bash
cat /tmp/hwpx_work/template_contents/Contents/section0.xml
```
Identify:
- All `{{...}}` placeholders and their exact spelling
- Whether placeholders are split across multiple XML elements (e.g., `<hp:t>{{</hp:t><hp:t>field}}</hp:t>`)
- Section titles and row labels (must be preserved)
- The audit table structure
- Corrective-action lines
- All occurrences of the risk tier placeholder
- All occurrences of the inspection date placeholder

#### 5. Write a Python script to perform all substitutions

Create `/tmp/hwpx_work/process.py` that does the following:

**a. Load JSON data:**
- Read `audit_overview.json` and `corrective_actions.json`.

**b. Read the XML content as a raw string.**

**c. Handle split placeholders:**
- First, collapse split `{{...}}` patterns. Use a regex to remove XML tags that appear between `{{` and `}}` so that placeholders become contiguous. Specifically, replace patterns like `{{</hp:t></hp:run><hp:run ...><hp:t>field_name}}` with `{{field_name}}` while preserving the surrounding XML structure. A robust approach: use regex to find all `\{\{[^}]*\}\}` patterns after stripping internal XML tags.
- Strategy: Use `re.sub` to collapse any XML markup within `{{...}}` blocks: `re.sub(r'\{\{((?:(?!\}\}).)*?)\}\}', lambda m: '{{' + re.sub(r'<[^>]+>', '', m.group(1)) + '}}', xml_text)` — but do this carefully, potentially in a loop or with a DOTALL-aware pattern.

**d. Perform substitutions:**
- Replace overview field placeholders with values from `audit_overview.json`.
- Replace audit table value cell placeholders.
- Fill the three corrective-action lines in the order they appear in `corrective_actions.json`.
- For the risk tier: replace ALL occurrences of the risk tier placeholder with the actual risk tier value.
- **Severity note**: Immediately after each risk tier value, append a severity note using this mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`. For example, if risk tier is "High", every occurrence should read "High 즉시조치" (with a space separator).
- **Date reformatting**: The inspection date from the JSON is in `YYYY-MM-DD` format. Rewrite it to `YYYY.MM.DD` everywhere it appears. Replace ALL occurrences in the XML.
- Ensure NO `{{...}}` placeholders remain anywhere.

**e. Remove stale layout-cache elements from modified paragraphs:**
- For every `<hp:p>` paragraph that was modified (or conservatively, for ALL paragraphs), remove `<hp:lineSegArray>...</hp:lineSegArray>` elements. Use: `re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', xml_text, flags=re.DOTALL)`

**f. Write the modified XML back.**

**g. Verify no `{{` remains:**
```python
assert '{{' not in modified_xml, f'Remaining placeholders found'
```

#### 6. Repackage the HWPX
```bash
cd /tmp/hwpx_work/template_contents
zip -r /root/safety_audit_brief_final.hwpx . -x '*.DS_Store'
```
Use `zip` with the correct structure (files at root of zip, preserving directory structure).

#### 7. Validate the output
```bash
# Check it's a valid zip
unzip -t /root/safety_audit_brief_final.hwpx

# Check no placeholders remain
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        content = z.read(name)
        try:
            text = content.decode('utf-8')
            if '{{' in text:
                print(f'PLACEHOLDER FOUND in {name}: ' + text[text.index('{{'):text.index('}}')+2])
        except: pass
    print('Validation complete')
"

# Check date format is YYYY.MM.DD not YYYY-MM-DD
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        try:
            text = z.read(name).decode('utf-8')
            dates_dash = re.findall(r'\d{4}-\d{2}-\d{2}', text)
            dates_dot = re.findall(r'\d{4}\.\d{2}\.\d{2}', text)
            if dates_dash: print(f'WARNING: YYYY-MM-DD dates still in {name}: {dates_dash}')
            if dates_dot: print(f'OK: YYYY.MM.DD dates in {name}: {dates_dot}')
        except: pass
"

# Check severity note is present
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        try:
            text = z.read(name).decode('utf-8')
            for note in ['즉시조치', '계획보완', '모니터링']:
                if note in text: print(f'Found severity note: {note} in {name}')
        except: pass
"
```

#### 8. Run any provided test/verifier
```bash
# Check if there's a test file
find /root -name 'test_*' -o -name '*test*.py' | head -10
# If found, run it
cd /root && python3 -m pytest test_output.py -v 2>&1 || true
```

### Critical Reminders
- Placeholders may be split across XML elements — you MUST handle this.
- Remove `<hp:lineSegArray>` from any modified paragraph to prevent rendering issues.
- The severity note must appear immediately after the risk tier text (e.g., "High 즉시조치"), at EVERY occurrence of the risk tier.
- The date must be reformatted from `YYYY-MM-DD` to `YYYY.MM.DD` at EVERY occurrence.
- Section titles and row labels must be preserved exactly.
- Output must be at exactly `/root/safety_audit_brief_final.hwpx`.
- The output must be a valid ZIP (HWPX package).

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