# Task Instruction

You must produce `/root/safety_audit_brief_final.hwpx` by filling the template with data from the two JSON files.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls /root/
ls /root/HWPX-Document-Automation/hwpx-safety-audit-brief/
```
Identify all provided files: `safety_audit_template.hwpx`, `audit_overview.json`, `corrective_actions.json`, and any test files.

### 2. Read the JSON data files
```bash
cat /root/HWPX-Document-Automation/hwpx-safety-audit-brief/audit_overview.json
cat /root/HWPX-Document-Automation/hwpx-safety-audit-brief/corrective_actions.json
```
Note every field value. Pay special attention to the risk tier value (e.g., "High", "Medium", or "Low") and the inspection date (in `YYYY-MM-DD` format).

### 3. Examine the HWPX template structure
A `.hwpx` file is a ZIP archive containing XML files.
```bash
cd /root/HWPX-Document-Automation/hwpx-safety-audit-brief/
python3 -c "
import zipfile, os
with zipfile.ZipFile('safety_audit_template.hwpx', 'r') as z:
    for name in z.namelist():
        print(name)
"
```
Then extract and read each XML file that contains document content (typically under `Contents/` — look for files like `section0.xml`, `section1.xml`, etc.):
```python
import zipfile
with zipfile.ZipFile('safety_audit_template.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            print(f'=== {name} ===')
            print(z.read(name).decode('utf-8'))
```
Read ALL XML content carefully. Identify:
- Every `{{...}}` placeholder and what it maps to
- Every occurrence of the risk tier placeholder
- Every occurrence of the inspection date placeholder
- The three corrective-action line placeholders
- Any layout-cache elements (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar elements that cache glyph/character positions)

### 4. Write the Python transformation script
Write a single Python script that:

a) Reads both JSON files.

b) Copies the HWPX zip, modifying only the XML content files.

c) For each content XML file:
   - Replaces ALL `{{...}}` placeholders with the corresponding JSON values.
   - Converts the inspection date from `YYYY-MM-DD` to `YYYY.MM.DD` format **everywhere** it appears (both in placeholders and after placeholder substitution — do a global search-replace of the date string).
   - Replaces every standalone occurrence of the risk tier value with `{RiskTier} ({SeverityNote})` using this mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`.
     **CRITICAL FORMAT**: The severity note MUST be in parentheses with a space before the opening paren. Example: if risk tier is "High", every occurrence becomes `High (즉시조치)`. Do NOT omit the parentheses.
   - Fills the three corrective-action lines in the same order as they appear in `corrective_actions.json`.
   - Removes layout-cache elements that could cause overlapping characters. Specifically, remove all `<hp:linesegarray>...</hp:linesegarray>` elements (case-insensitive tag matching). Use regex or XML parsing to strip these elements from any paragraph whose text content was modified.
   - Preserves all section titles and row labels unchanged.
   - Ensures no `{{` or `}}` placeholder markers remain.

d) Writes the result to `/root/safety_audit_brief_final.hwpx`.

**Implementation approach for the HWPX rewrite:**
```python
import zipfile, json, re, os, shutil

# Read JSON data
with open('audit_overview.json') as f:
    overview = json.load(f)
with open('corrective_actions.json') as f:
    actions = json.load(f)

# Determine severity note
severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}
risk_tier = overview.get('risk_tier') or overview.get('riskTier') or ...  # adapt to actual key
severity_note = severity_map[risk_tier]

# Date conversion
inspection_date = overview.get('inspection_date') or ...  # adapt to actual key
date_reformatted = inspection_date.replace('-', '.')

# Build replacement dict from overview fields
# ... map each {{placeholder}} to its value from the JSON

# Process HWPX
with zipfile.ZipFile('safety_audit_template.hwpx', 'r') as zin:
    with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx', 'w') as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith('.xml'):
                text = data.decode('utf-8')
                original_text = text
                # 1. Replace all {{...}} placeholders
                # 2. Fill corrective actions in order
                # 3. Replace date format YYYY-MM-DD -> YYYY.MM.DD
                text = text.replace(inspection_date, date_reformatted)
                # 4. Replace risk tier with "RiskTier (SeverityNote)"
                # Be careful: replace the standalone risk tier AFTER placeholders are filled
                # Use word-boundary or context-aware replacement to avoid partial matches
                text = text.replace(risk_tier, f'{risk_tier} ({severity_note})')
                # But then fix double-replacements if risk_tier appears in the severity string
                # e.g., avoid "High (즉시조치) (즉시조치)"
                text = text.replace(f'{risk_tier} ({severity_note}) ({severity_note})', f'{risk_tier} ({severity_note})')
                # 5. Remove linesegarray from modified paragraphs
                # Simple approach: remove all linesegarray elements
                text = re.sub(r'<hp:linesegarray>.*?</hp:linesegarray>', '', text, flags=re.DOTALL|re.IGNORECASE)
                text = re.sub(r'<linesegarray>.*?</linesegarray>', '', text, flags=re.DOTALL|re.IGNORECASE)
                # Also handle lineSegArray variant
                text = re.sub(r'<[^>]*lineSegArray[^>]*>.*?</[^>]*lineSegArray[^>]*>', '', text, flags=re.DOTALL|re.IGNORECASE)
                # 6. Verify no {{...}} remain
                assert '{{' not in text, f'Unfilled placeholder in {item.filename}: ' + re.findall(r'\{\{.*?\}\}', text)[0]
                data = text.encode('utf-8')
            zout.writestr(item, data)
```

**Adapt the above skeleton** to the actual JSON keys and placeholder names you discover in steps 2-3. Do NOT guess the placeholder names — read them from the XML.

### 5. Run the script
```bash
cd /root/HWPX-Document-Automation/hwpx-safety-audit-brief/
python3 transform.py
```

### 6. Validate the output

a) Verify the output is a valid ZIP:
```bash
python3 -c "import zipfile; zipfile.ZipFile('/root/safety_audit_brief_final.hwpx', 'r').testzip()"
```

b) Check all content XML for the critical patterns:
```python
import zipfile, re
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            text = z.read(name).decode('utf-8')
            # Check no placeholders remain
            placeholders = re.findall(r'\{\{.*?\}\}', text)
            if placeholders:
                print(f'ERROR: unfilled placeholders in {name}: {placeholders}')
            # Check severity note format
            # e.g., for High risk tier:
            if 'High' in text:
                assert 'High (즉시조치)' in text, 'Severity note format wrong!'
            # Check date format
            # Should have YYYY.MM.DD, not YYYY-MM-DD
            print(f'{name}: OK')
            # Print relevant snippets for manual verification
            for line in text.split('>'):
                if '즉시조치' in line or '계획보완' in line or '모니터링' in line:
                    print(f'  SEVERITY: ...{line[:200]}')
```

c) Run the test suite:
```bash
cd /root/HWPX-Document-Automation/hwpx-safety-audit-brief/
python3 -m pytest test_output.py -v
```

### 7. If tests fail
- Read the exact assertion error message.
- Extract and inspect the specific XML content the test is checking.
- Fix the transformation script accordingly.
- Re-run until all tests pass.

## CRITICAL REMINDERS
- **Severity note format**: `{RiskTier} ({SeverityNote})` — with a SPACE before the opening parenthesis and the note inside parentheses. E.g., `High (즉시조치)`. This was the failure in the previous run.
- **Date format**: `YYYY.MM.DD` with dots, not dashes.
- **Order**: Corrective actions must be filled in the same order as the JSON array.
- **No leftover placeholders**: Every `{{...}}` must be replaced.
- **Layout cache removal**: Strip `linesegarray` (and similar layout cache) elements from modified paragraphs to prevent overlapping characters.
- Read the test file (`test_output.py`) before writing the script to understand exactly what assertions will be checked.

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