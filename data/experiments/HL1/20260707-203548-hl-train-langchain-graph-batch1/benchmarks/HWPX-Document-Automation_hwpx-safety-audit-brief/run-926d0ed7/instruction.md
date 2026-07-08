# Task Instruction

You must prepare a warehouse safety audit brief by filling a HWPX template with data from two JSON files, then saving the result.

## Step-by-step plan

### 1. Explore the workspace
```bash
cd /root
find . -maxdepth 3 -type f | head -80
```
Identify the task directory containing `safety_audit_template.hwpx`, `audit_overview.json`, `corrective_actions.json`, and any test/verifier files.

### 2. Read the JSON data files
```bash
cat <task_dir>/audit_overview.json
cat <task_dir>/corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The inspection date (in `YYYY-MM-DD` format — you must convert every occurrence to `YYYY.MM.DD`).
- The risk tier value (e.g., "High", "Medium", or "Low").
- The three corrective actions and their order.

### 3. Inspect the HWPX template
HWPX is a ZIP-based format. Unzip it to a temp directory:
```bash
mkdir /tmp/hwpx_work
cp <task_dir>/safety_audit_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f
```
Identify all XML files inside (likely under `Contents/`). The main document body is usually in a file like `section0.xml` or `content.hpf` — look for `<hp:t>` tags containing `{{...}}` placeholders.

### 4. Read and understand the XML content
```bash
cat template_extracted/Contents/section0.xml
```
(or whichever file contains the body text). Carefully note:
- All `{{placeholder}}` tokens. **Crucially**, placeholders may be split across multiple `<hp:t>` tags (e.g., `<hp:t>{{</hp:t><hp:t>field_name}}</hp:t>`). You must handle this.
- Section titles and row labels (preserve them exactly).
- The structure of the summary section and the audit table.
- Any `<hp:lineSegArray>` or `<hp:lineSeg>` elements (layout cache).

### 5. Write a Python script to perform all replacements
Create `/tmp/hwpx_work/fill_template.py` that does the following:

```python
import json, os, re, shutil, zipfile

# Paths
TASK_DIR = '<task_dir>'  # fill in actual path
TEMPLATE = os.path.join(TASK_DIR, 'safety_audit_template.hwpx')
OUTPUT = '/root/safety_audit_brief_final.hwpx'
WORK = '/tmp/hwpx_fill'

# Load JSON data
with open(os.path.join(TASK_DIR, 'audit_overview.json')) as f:
    overview = json.load(f)
with open(os.path.join(TASK_DIR, 'corrective_actions.json')) as f:
    actions = json.load(f)

# Build replacement map from overview + actions
# Map every {{key}} to its value
# For the date field: convert YYYY-MM-DD to YYYY.MM.DD
# For the risk tier: append severity note using mapping High->즉시조치, Medium->계획보완, Low->모니터링
#   e.g., if risk_tier is "High", replace {{risk_tier}} with "High 즉시조치"
#   AND update every OTHER occurrence of the risk tier similarly

severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}

# Extract and unzip
if os.path.exists(WORK):
    shutil.rmtree(WORK)
os.makedirs(WORK)
with zipfile.ZipFile(TEMPLATE, 'r') as z:
    z.extractall(WORK)

# Find all XML files
xml_files = []
for root, dirs, files in os.walk(WORK):
    for fname in files:
        if fname.endswith('.xml') or fname.endswith('.hpf'):
            xml_files.append(os.path.join(root, fname))

for xml_path in xml_files:
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Step A: Merge split placeholders
    # Remove XML tags between {{ and }} so that split placeholders become whole
    # Use iterative regex to collapse <hp:t>{{</hp:t>...<hp:t>key}}</hp:t> patterns
    # Strategy: find all text within <hp:run> elements, concatenate <hp:t> children,
    # then if the concatenated text contains a placeholder, put it all in one <hp:t>
    # 
    # Simpler approach: repeatedly collapse adjacent </hp:t> and <hp:t> tags within
    # the same run that are splitting a placeholder.
    
    # Collapse split placeholders by removing tags between {{ and }}
    # This regex finds {{ ... }} where ... may contain XML tags
    prev = None
    while prev != content:
        prev = content
        content = re.sub(
            r'(\{\{[^}]*?)</hp:t>(\s*</hp:run>)?\s*(<hp:run[^>]*>\s*)?(<[^>]*>)*\s*<hp:t[^>]*>([^<]*?\}\})',
            r'\1\5',
            content
        )
        # Also handle simpler splits within same run
        content = re.sub(
            r'(\{\{[^}]*?)</hp:t>\s*<hp:t[^>]*>([^<]*?\}\})',
            r'\1\2',
            content
        )

    # Step B: Build replacement dict
    replacements = {}
    # Add all overview fields
    for key, value in overview.items():
        replacements[key] = str(value)
    # Add corrective action fields (figure out naming from template)
    # The placeholder names will be visible from the XML inspection
    # e.g., {{action_1}}, {{action_2}}, {{action_3}} or similar
    # Fill them in order from corrective_actions.json
    
    # Perform replacements
    for key, value in replacements.items():
        # Date conversion: if value matches YYYY-MM-DD, convert to YYYY.MM.DD
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value):
            value = value.replace('-', '.')
        content = content.replace('{{' + key + '}}', value)
    
    # Step C: Handle risk tier + severity note
    # After placeholder replacement, find all occurrences of the risk tier text
    # and append the severity note if not already present
    risk_tier = overview.get('risk_tier', overview.get('risk_level', ''))
    if risk_tier in severity_map:
        note = severity_map[risk_tier]
        # Replace standalone risk tier occurrences (that don't already have the note)
        # Be careful not to double-append
        content = re.sub(
            rf'({re.escape(risk_tier)})(?!\s*{re.escape(note)})',
            rf'\1 {note}',
            content
        )
    
    # Step D: Convert any remaining YYYY-MM-DD dates to YYYY.MM.DD
    date_val = overview.get('inspection_date', overview.get('audit_date', ''))
    if date_val:
        content = content.replace(date_val, date_val.replace('-', '.'))
    
    # Step E: Remove layout cache elements (lineSegArray, lineSeg)
    content = re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL)
    content = re.sub(r'<hp:lineSeg[^/]*?/>', '', content)
    content = re.sub(r'<hp:lineSeg[^>]*>.*?</hp:lineSeg>', '', content, flags=re.DOTALL)
    
    # Step F: Verify no remaining placeholders
    remaining = re.findall(r'\{\{.*?\}\}', content)
    if remaining:
        print(f'WARNING: remaining placeholders in {xml_path}: {remaining}')
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(content)

# Repackage as HWPX (ZIP)
with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(WORK):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, WORK)
            zout.write(fpath, arcname)

print(f'Output written to {OUTPUT}')
```

**IMPORTANT**: The above is a template script. You MUST adapt it after inspecting the actual JSON keys, placeholder names in the XML, and corrective action field names. Do NOT run it blindly.

### 6. Detailed adaptation checklist
After inspecting the files:
- Map every `{{placeholder}}` in the XML to the correct JSON key/value.
- For corrective actions: identify the placeholder names (e.g., `{{corrective_action_1}}`) and map them to the three items from `corrective_actions.json` in order.
- Ensure the risk tier replacement adds the Korean severity note (e.g., `"High 즉시조치"`).
- Ensure ALL occurrences of the date are converted from `YYYY-MM-DD` to `YYYY.MM.DD`.
- Ensure ALL occurrences of the risk tier get the severity note appended.
- Verify no `{{...}}` placeholders remain.
- Remove all `lineSegArray` and `lineSeg` elements from any modified XML.

### 7. Run and verify
```bash
python3 /tmp/hwpx_work/fill_template.py
```

Then verify:
```bash
# Check it's a valid ZIP
unzip -t /root/safety_audit_brief_final.hwpx

# Check no remaining placeholders
mkdir /tmp/verify
cd /tmp/verify
unzip /root/safety_audit_brief_final.hwpx
grep -r '{{' . || echo 'No placeholders remaining - GOOD'

# Check date format converted
grep -r 'YYYY-MM-DD pattern or actual date with hyphens' . || echo 'No hyphenated dates - GOOD'

# Check severity note present
grep -r '즉시조치\|계획보완\|모니터링' . && echo 'Severity note found - GOOD'

# Check no lineSegArray/lineSeg remain in modified files
grep -r 'lineSegArray\|lineSeg' . || echo 'No layout cache - GOOD'
```

### 8. Run the verifier/tests if available
```bash
cd <task_dir>
# Look for test files
ls test* pytest* verif*
# Run them
python3 -m pytest test_output.py -v 2>&1
```

Fix any failures and re-run until the verifier passes.

### Critical reminders
- **Split placeholders**: HWPX XML commonly splits `{{key}}` across multiple `<hp:t>` tags. You MUST merge them before replacement.
- **Layout cache**: Remove `<hp:lineSegArray>...</hp:lineSegArray>` and any `<hp:lineSeg .../>` elements from paragraphs you modify.
- **Risk tier + severity note**: The note goes immediately after the risk tier text with a space, e.g., `"High 즉시조치"`. Apply this everywhere the risk tier appears, not just in placeholders.
- **Date format**: Convert EVERY occurrence of the specific date from `YYYY-MM-DD` to `YYYY.MM.DD`.
- **Preserve structure**: Keep section titles, row labels, and overall XML structure intact.
- **Valid HWPX**: The output must be a proper ZIP file with the same internal structure.

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