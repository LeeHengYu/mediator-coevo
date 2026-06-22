# Task Instruction

You must produce the file `/root/safety_audit_brief_final.hwpx` by filling a template HWPX document with data from two JSON files. Follow every step precisely.

## Step 0 – Explore the workspace
```bash
find /root -maxdepth 3 -type f | head -80
```
Identify the exact paths of `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

## Step 1 – Understand the data files
```bash
cat <path>/audit_overview.json
cat <path>/corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The `risk_tier` (or equivalent) value – e.g. `High`, `Medium`, `Low`.
- The `inspection_date` (or equivalent) – in `YYYY-MM-DD` format.
- The severity mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`.

## Step 2 – Understand the HWPX template structure
An `.hwpx` file is a ZIP archive containing XML files. Unzip it to a temporary directory:
```bash
mkdir -p /tmp/hwpx_work
cp <path>/safety_audit_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -o template.hwpx -d template_contents
find template_contents -type f
```
Then inspect every XML file, especially the main content XML (likely `Contents/section0.xml` or similar):
```bash
cat template_contents/Contents/section0.xml
```
Also check for any other XML files that might contain `{{` placeholders:
```bash
grep -rl '{{' template_contents/
```
Read each file that contains placeholders.

## Step 3 – Write a Python script to perform all substitutions
Create `/tmp/hwpx_work/fill_template.py` with the following logic:

1. **Load JSON data** from both `audit_overview.json` and `corrective_actions.json`.
2. **Copy** the template HWPX to the output path.
3. **Open** the output HWPX as a ZIP (use `zipfile.ZipFile`).
4. **For each XML entry** in the ZIP that contains `{{` placeholders, perform substitutions.
5. **Substitution rules** (apply in this order):
   a. Replace every `{{placeholder}}` with the corresponding value from the JSON data. Map placeholder names to JSON keys carefully by inspecting both.
   b. For the inspection date: after substituting the date value, find every occurrence of the date in `YYYY-MM-DD` format and rewrite it to `YYYY.MM.DD` (replace hyphens with dots).
   c. For the risk tier: after substituting the risk tier value, find every occurrence of the risk tier text (e.g. `High`) and append the severity note **in parentheses** using the format `High (즉시조치)`. Use the mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`.
   
   **CRITICAL**: The severity note MUST be in parentheses. The expected format is `High (즉시조치)`, NOT `High 즉시조치`. This was the cause of the previous failure.
   
   d. For corrective actions: fill the three corrective-action lines in the same order they appear in `corrective_actions.json`.
   e. Verify no `{{` or `}}` remain in any XML content.
6. **Remove stale layout-cache elements**: In every paragraph (`<hp:p>` or similar) whose text content was modified, remove any `<hp:linesegarray>` or `<hp:lineSegArray>` child elements (these are layout caches that cause overlapping characters). Search for the actual element name by inspecting the XML – it might be under a namespace. Remove ALL such layout cache elements from modified paragraphs.
7. **Write** the modified ZIP to `/root/safety_audit_brief_final.hwpx`.

IMPORTANT implementation details:
- Use `zipfile.ZipFile` to read all entries, modify XML content in memory, then write a new ZIP with the same entries.
- Preserve all non-XML entries (images, etc.) byte-for-byte.
- Use `xml.etree.ElementTree` for XML parsing to properly handle namespaces.
- When doing text replacements in XML, work at the text node level (element.text and element.tail), not on raw XML strings, to avoid breaking XML structure.
- However, if the placeholders span multiple `<run>` or `<t>` elements, you may need to work on the serialized XML string. In that case, parse → serialize to string → do text replacements → parse back → remove layout caches → serialize final.
- Actually, the safest approach: read the raw XML bytes from the zip, decode to string, do all `{{...}}` text replacements on the string, then parse as XML, remove layout cache elements from modified paragraphs, serialize back.

Here is a sketch (adapt based on actual file structure):
```python
import json, zipfile, os, re, shutil, copy
from xml.etree import ElementTree as ET

# Load data
with open('<path>/audit_overview.json') as f:
    overview = json.load(f)
with open('<path>/corrective_actions.json') as f:
    actions = json.load(f)

# Build replacement map from placeholders to values
# (You MUST inspect the actual placeholders in the XML and map them to JSON keys)
# Example:
# replacements = {
#     '{{facility_name}}': overview['facility_name'],
#     '{{inspection_date}}': overview['inspection_date'].replace('-', '.'),
#     '{{risk_tier}}': f"{overview['risk_tier']} ({severity_map[overview['risk_tier']]})",
#     ...
# }

severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}

# ... build full replacements dict ...

# Process HWPX
template_path = '<path>/safety_audit_template.hwpx'
output_path = '/root/safety_audit_brief_final.hwpx'

with zipfile.ZipFile(template_path, 'r') as zin:
    with zipfile.ZipFile(output_path, 'w') as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith('.xml'):
                text = data.decode('utf-8')
                original_text = text
                # Apply all placeholder replacements
                for placeholder, value in replacements.items():
                    text = text.replace(placeholder, value)
                # Date format: replace any remaining YYYY-MM-DD dates
                # Risk tier: ensure format is "Value (Severity)"
                # Verify no {{ remain
                assert '{{' not in text, f'Unreplaced placeholder in {item.filename}: {re.findall(r"{{.*?}}", text)}'
                
                if text != original_text:
                    # Remove layout cache from modified paragraphs
                    # Parse XML, find and remove linesegarray elements
                    # ... (adapt to actual namespace and element names)
                    pass
                
                data = text.encode('utf-8')
            zout.writestr(item, data)
```

## Step 4 – Run the script
```bash
cd /tmp/hwpx_work
python3 fill_template.py
```

## Step 5 – Validate the output
```bash
# Check it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx'); print(z.namelist()); z.close()"

# Check no placeholders remain
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            if '{{' in content:
                print(f'FAIL: placeholder in {name}')
                import re
                print(re.findall(r'\{\{.*?\}\}', content))
            else:
                print(f'OK: {name}')
"

# Check risk tier format includes parentheses
python3 -c "
import zipfile, re
severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            for tier, sev in severity_map.items():
                pattern = f'{tier} ({sev})'
                if tier in content:
                    if pattern in content:
                        print(f'OK: Found \"{pattern}\" in {name}')
                    else:
                        print(f'WARNING: Found \"{tier}\" but not \"{pattern}\" in {name}')
"

# Check date format is YYYY.MM.DD not YYYY-MM-DD
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            dates_hyphen = re.findall(r'\d{4}-\d{2}-\d{2}', content)
            dates_dot = re.findall(r'\d{4}\.\d{2}\.\d{2}', content)
            if dates_hyphen:
                print(f'FAIL: YYYY-MM-DD dates in {name}: {dates_hyphen}')
            if dates_dot:
                print(f'OK: YYYY.MM.DD dates in {name}: {dates_dot}')
"

# Check no layout cache in modified paragraphs
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            if 'lineseg' in content.lower() or 'LineSegArray' in content or 'lineSegArray' in content:
                print(f'WARNING: layout cache may remain in {name}')
            else:
                print(f'OK: no layout cache in {name}')
"
```

## Step 6 – Run the test suite if available
```bash
cd /root
# Look for test files
find . -name 'test_output.py' -o -name 'test_*.py' | head -5
# Run if found
python3 -m pytest test_output.py -v 2>&1 | tail -40
```
If any test fails, read the error carefully, fix the script, and re-run.

## Key Reminders
- **Parentheses around severity**: `High (즉시조치)` NOT `High 즉시조치`. This is the #1 priority fix from previous feedback.
- **Date format**: `2024.03.15` NOT `2024-03-15`.
- **All placeholders removed**: No `{{...}}` anywhere.
- **Layout cache removed**: Remove `lineSegArray` / `linesegarray` elements from modified paragraphs.
- **Valid HWPX ZIP**: The output must be openable as a ZIP with the same structure as the template.
- **Corrective actions in order**: Same order as in `corrective_actions.json`.
- **Preserve section titles and row labels**: Don't modify existing Korean text that isn't a placeholder.

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