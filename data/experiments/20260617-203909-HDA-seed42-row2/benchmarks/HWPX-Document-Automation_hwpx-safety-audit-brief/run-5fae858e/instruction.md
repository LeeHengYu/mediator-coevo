# Task Instruction

You must produce `/root/safety_audit_brief_final.hwpx` by filling in the HWPX template with data from two JSON files. Follow these steps exactly:

## 1 – Inspect the workspace
```bash
ls /root/
```
Identify the template (`safety_audit_template.hwpx`) and the two JSON data files (`audit_overview.json`, `corrective_actions.json`).

## 2 – Read the JSON data files
```bash
cat /root/audit_overview.json
cat /root/corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The `risk_tier` (or equivalent) value and its case (High / Medium / Low).
- The `inspection_date` (or equivalent) in YYYY-MM-DD format.
- The three corrective-action entries and their order.

## 3 – Explore the HWPX template structure
HWPX is a ZIP archive containing XML files.
```bash
cp /root/safety_audit_template.hwpx /tmp/template.zip
cd /tmp && mkdir template_extract && cd template_extract
unzip /tmp/template.zip
find . -type f
```
Identify the main content XML (likely `Contents/section0.xml` or similar). There may also be a `content.hpf` or header XML.

## 4 – Examine every XML file for placeholders and content
For each XML file found, `cat` it and look for:
- `{{...}}` placeholder patterns — note the exact placeholder names.
- Existing text that contains the inspection date or risk tier.
- `<hp:lineSegArray>` elements (layout cache that must be removed from edited paragraphs).
- Section titles and row labels that must be preserved.

## 5 – Write a Python script to perform the transformation
Create `/tmp/fill_template.py` with the following logic:

```python
import zipfile, json, os, re, shutil
from lxml import etree

# Paths
template_path = '/root/safety_audit_template.hwpx'
output_path = '/root/safety_audit_brief_final.hwpx'
overview_path = '/root/audit_overview.json'
actions_path = '/root/corrective_actions.json'

# Load JSON data
with open(overview_path) as f:
    overview = json.load(f)
with open(actions_path) as f:
    actions = json.load(f)

# Build severity mapping
severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}

# Determine risk tier and severity note
risk_tier = overview.get('risk_tier') or overview.get('riskTier') or overview.get('risk_level', '')
severity_note = severity_map.get(risk_tier, '')

# Format date: YYYY-MM-DD -> YYYY.MM.DD
raw_date = overview.get('inspection_date') or overview.get('inspectionDate') or overview.get('date', '')
formatted_date = raw_date.replace('-', '.')

# Build replacement dict from overview (map every key to its value)
# We'll do placeholder replacement AND global text replacement

# Extract template
extract_dir = '/tmp/hwpx_work'
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir)
with zipfile.ZipFile(template_path, 'r') as zf:
    zf.extractall(extract_dir)

# Collect all XML files
xml_files = []
for root, dirs, files in os.walk(extract_dir):
    for fn in files:
        if fn.endswith('.xml') or fn.endswith('.hpf'):
            xml_files.append(os.path.join(root, fn))

# Build placeholder replacement map
# We need to discover exact placeholder names from the XML first
# Read all XML content to find {{...}} patterns
all_placeholders = set()
for xf in xml_files:
    with open(xf, 'r', encoding='utf-8') as f:
        content = f.read()
    all_placeholders.update(re.findall(r'\{\{[^}]+\}\}', content))

print('Found placeholders:', all_placeholders)

# Build a flexible replacement map
# Flatten overview keys (handle nested dicts if needed)
def flatten(d, prefix=''):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(flatten(v, prefix))
        else:
            items[k] = str(v)
    return items

overview_flat = flatten(overview)

# Also prepare corrective actions list
if isinstance(actions, dict):
    actions_list = actions.get('corrective_actions') or actions.get('actions') or list(actions.values())[0]
else:
    actions_list = actions

# Build replacement dict
replacements = {}
for ph in all_placeholders:
    key = ph.strip('{}').strip()
    # Try direct match
    if key in overview_flat:
        val = overview_flat[key]
    else:
        # Try case-insensitive / underscore-insensitive match
        norm = key.lower().replace(' ', '_').replace('-', '_')
        matched = False
        for ok, ov in overview_flat.items():
            if ok.lower().replace(' ', '_').replace('-', '_') == norm:
                replacements[ph] = ov
                matched = True
                break
        if not matched:
            # Check if it's a corrective action placeholder
            # e.g., {{corrective_action_1}}, {{action_1}}, etc.
            m = re.search(r'(\d+)', key)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(actions_list):
                    act = actions_list[idx]
                    if isinstance(act, dict):
                        # Join all values or pick description
                        val = act.get('description') or act.get('action') or act.get('content') or ' / '.join(str(v) for v in act.values())
                        replacements[ph] = str(val)
                    else:
                        replacements[ph] = str(act)
            continue
        continue
    replacements[ph] = val

print('Replacement map:', replacements)

# Now process each XML file
for xf in xml_files:
    with open(xf, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Apply placeholder replacements
    for ph, val in replacements.items():
        content = content.replace(ph, val)
    
    # Handle any remaining placeholders for corrective actions
    # (they might need special field-by-field replacement)
    # We'll handle this after seeing what placeholders remain
    
    # Global: replace risk tier with "RiskTier (SeverityNote)"
    # But be careful not to double-replace if already in placeholder
    # First do placeholder replacement, then do global risk tier update
    if risk_tier and risk_tier in content:
        # Replace bare risk tier occurrences with "RiskTier (SeverityNote)"
        # But avoid replacing if already followed by the severity note
        pattern = re.escape(risk_tier) + r'(?!\s*\(' + re.escape(severity_note) + r'\))'
        content = re.sub(pattern, f'{risk_tier} ({severity_note})', content)
    
    # Global: reformat all dates from YYYY-MM-DD to YYYY.MM.DD
    content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1.\2.\3', content)
    
    # Remove lineSegArray elements from paragraphs we modified
    # Since we can't easily track which paragraphs changed, remove ALL lineSegArray
    # elements (they are layout cache and will be regenerated)
    if content != original:
        content = re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL)
    
    with open(xf, 'w', encoding='utf-8') as f:
        f.write(content)

# Verify no remaining placeholders
for xf in xml_files:
    with open(xf, 'r', encoding='utf-8') as f:
        content = f.read()
    remaining = re.findall(r'\{\{[^}]+\}\}', content)
    if remaining:
        print(f'WARNING: Remaining placeholders in {xf}: {remaining}')

# Repackage as HWPX
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(extract_dir):
        for fn in files:
            filepath = os.path.join(root, fn)
            arcname = os.path.relpath(filepath, extract_dir)
            zf.write(filepath, arcname)

print(f'Output written to {output_path}')
```

**IMPORTANT**: This script is a starting framework. After step 4, you will know the exact placeholder names and JSON field names. You MUST adapt the replacement logic to match the actual placeholder names to the actual JSON keys. Do NOT run the script blindly — adjust it based on what you discover.

## 6 – Run the script
```bash
python3 /tmp/fill_template.py
```

## 7 – Verify the output

### 7a – Check it's a valid ZIP/HWPX
```bash
unzip -t /root/safety_audit_brief_final.hwpx
```

### 7b – Check for remaining placeholders
```bash
mkdir -p /tmp/verify && cd /tmp/verify && unzip -o /root/safety_audit_brief_final.hwpx
grep -r '{{' . || echo 'No remaining placeholders - GOOD'
```

### 7c – Verify risk tier formatting
```bash
grep -r '즉시조치\|계획보완\|모니터링' /tmp/verify/ || echo 'WARNING: No severity note found'
```

### 7d – Verify date format (should be YYYY.MM.DD, not YYYY-MM-DD)
```bash
grep -rP '\d{4}-\d{2}-\d{2}' /tmp/verify/ && echo 'WARNING: Old date format still present' || echo 'Date format OK'
```

### 7e – Verify lineSegArray removal
```bash
grep -r 'lineSegArray' /tmp/verify/ && echo 'WARNING: lineSegArray still present' || echo 'Layout cache cleaned'
```

## 8 – Run the verifier
```bash
cd /root && python -m pytest test_output.py -v
```

If any test fails, read the error carefully, inspect the relevant XML content in the output HWPX, and fix the specific issue. Common issues:
- Placeholder name mismatch (re-examine the template XML and JSON keys)
- Risk tier not formatted as `High (즉시조치)` exactly (check spacing and parentheses)
- Corrective actions not in the right order or missing fields
- Some placeholders split across XML tags (need to handle text spanning multiple `<hp:t>` elements)

## 9 – Handle split placeholders (if needed)
If placeholders like `{{field_name}}` are split across multiple `<hp:t>` XML elements (e.g., `<hp:t>{{field</hp:t><hp:t>_name}}</hp:t>`), you need to:
1. Parse the XML properly with lxml
2. Concatenate text from consecutive `<hp:t>` elements within the same `<hp:run>` or `<hp:p>`
3. Perform replacement on the concatenated text
4. Write the result back to the first `<hp:t>` element and clear the rest

This is a known issue with HWPX templates. Handle it if grep reveals partial `{{` or `}}` patterns.

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