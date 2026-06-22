# Task Instruction

Execute the following steps in order to produce `/root/safety_audit_brief_final.hwpx`.

## 1. Explore the workspace
```bash
cd /root
ls -la
find . -maxdepth 2 -type f | head -60
```
Identify the location of `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

## 2. Inspect the JSON data files
```bash
cat audit_overview.json
cat corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The inspection date (will be in `YYYY-MM-DD` format; must become `YYYY.MM.DD`).
- The risk tier value (e.g., `High`, `Medium`, or `Low`).
- The corrective action items and their order.

## 3. Inspect the HWPX template
HWPX is a ZIP archive. Unzip it to examine its XML contents:
```bash
mkdir -p /tmp/hwpx_work
cp safety_audit_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -o template.hwpx -d template_contents
find template_contents -type f
```
Read every XML file inside, especially the main content XML (often `Contents/section0.xml` or similar):
```bash
for f in $(find template_contents -name '*.xml'); do echo "=== $f ==="; cat "$f"; echo; done
```
Identify:
- All `{{...}}` placeholder tokens and their exact spelling.
- The structure of the summary section and audit table.
- The three corrective-action lines.
- Every occurrence of the risk tier placeholder.
- Every occurrence of the date placeholder.
- Any `<hp:lineSegArray>` or `<hp:lineSeg>` elements (layout cache) that appear inside paragraphs.

## 4. Build a Python script to do the replacement
Write `/tmp/hwpx_work/build.py` that:

```python
import json, os, re, shutil, zipfile
from pathlib import Path
from lxml import etree

# Paths
TEMPLATE = '/tmp/hwpx_work/template.hwpx'
OUTPUT = '/root/safety_audit_brief_final.hwpx'
OVERVIEW = None  # will be set after finding the file
CORRECTIVE = None

# Find JSON files (they may be in /root or a subdirectory)
for candidate_dir in ['/root', '/root/HWPX-Document-Automation', '/root/HWPX-Document-Automation/hwpx-safety-audit-brief']:
    p = Path(candidate_dir)
    if (p / 'audit_overview.json').exists():
        OVERVIEW = p / 'audit_overview.json'
        CORRECTIVE = p / 'corrective_actions.json'
        break

assert OVERVIEW and OVERVIEW.exists(), 'Cannot find audit_overview.json'

overview = json.loads(OVERVIEW.read_text(encoding='utf-8'))
corrective = json.loads(CORRECTIVE.read_text(encoding='utf-8'))

# Determine risk tier and severity note
SEVERITY_MAP = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}
risk_tier = overview.get('risk_tier') or overview.get('riskTier') or overview.get('risk_level', '')
# Try common key patterns
for k, v in overview.items():
    if 'risk' in k.lower():
        risk_tier = v
        break
severity_note = SEVERITY_MAP.get(risk_tier, '')
risk_with_severity = f"{risk_tier} ({severity_note})"

# Reformat date
raw_date = ''
for k, v in overview.items():
    if 'date' in k.lower() and isinstance(v, str) and re.match(r'\d{4}-\d{2}-\d{2}', v):
        raw_date = v
        break
formatted_date = raw_date.replace('-', '.') if raw_date else ''

print(f'Risk tier: {risk_tier}')
print(f'Risk with severity: {risk_with_severity}')
print(f'Date raw: {raw_date} -> formatted: {formatted_date}')
print(f'Overview keys: {list(overview.keys())}')
print(f'Corrective actions: {json.dumps(corrective, ensure_ascii=False, indent=2)}')

# --- Phase 1: Text replacement ---
# Unzip
work_dir = Path('/tmp/hwpx_work/output_contents')
if work_dir.exists():
    shutil.rmtree(work_dir)
work_dir.mkdir(parents=True)

with zipfile.ZipFile(TEMPLATE, 'r') as zin:
    zin.extractall(work_dir)

# Build replacement map from overview (placeholder -> value)
# We'll discover exact placeholder names from the XML content
xml_files = list(work_dir.rglob('*.xml'))

# First pass: discover all {{...}} placeholders
all_placeholders = set()
for xf in xml_files:
    content = xf.read_text(encoding='utf-8')
    all_placeholders.update(re.findall(r'\{\{[^}]+\}\}', content))

print(f'Found placeholders: {all_placeholders}')

# Build replacement dict
replacements = {}
for ph in all_placeholders:
    key = ph.strip('{}').strip()
    # Try direct match in overview
    if key in overview:
        val = str(overview[key])
        # If this is the date field, reformat
        if re.match(r'\d{4}-\d{2}-\d{2}', val):
            val = val.replace('-', '.')
        # If this is the risk tier field, append severity
        if val in SEVERITY_MAP:
            val = f"{val} ({SEVERITY_MAP[val]})"
        replacements[ph] = val
    else:
        # Try case-insensitive / underscore variants
        key_lower = key.lower().replace(' ', '_')
        for ok, ov in overview.items():
            if ok.lower().replace(' ', '_') == key_lower:
                val = str(ov)
                if re.match(r'\d{4}-\d{2}-\d{2}', val):
                    val = val.replace('-', '.')
                if val in SEVERITY_MAP:
                    val = f"{val} ({SEVERITY_MAP[val]})"
                replacements[ph] = val
                break

# Handle corrective actions - they may be {{corrective_action_1}} etc or in a list
# We need to figure out the pattern from placeholders
corrective_phs = sorted([ph for ph in all_placeholders if 'corrective' in ph.lower() or 'action' in ph.lower()])
print(f'Corrective placeholders: {corrective_phs}')

# corrective_actions.json might be a list of objects or a dict
if isinstance(corrective, list):
    actions = corrective
else:
    actions = corrective.get('actions') or corrective.get('corrective_actions') or [corrective]

# Map corrective action placeholders to values
# Sort them to match order
for i, ph in enumerate(sorted(corrective_phs)):
    if i < len(actions):
        action = actions[i]
        if isinstance(action, dict):
            # Join all values or use a specific field
            # Check what fields exist
            val_parts = []
            for ak, av in action.items():
                val_parts.append(str(av))
            replacements[ph] = ' / '.join(val_parts)
        else:
            replacements[ph] = str(action)

print(f'Replacements: {json.dumps(replacements, ensure_ascii=False, indent=2)}')

# IMPORTANT: Before doing bulk replacement, also handle any standalone risk tier
# and date occurrences that are NOT inside {{...}} but are literal values
# (the task says "Update every occurrence of the risk tier" and "Rewrite the
# inspection date ... everywhere it appears")

# Apply replacements to all XML files
modified_files = set()
for xf in xml_files:
    content = xf.read_text(encoding='utf-8')
    original = content
    for ph, val in replacements.items():
        content = content.replace(ph, val)
    # Also replace any remaining raw date with formatted date
    if raw_date and formatted_date:
        content = content.replace(raw_date, formatted_date)
    # Also replace any standalone risk tier without severity note
    # (but be careful not to double-replace)
    if risk_tier and severity_note:
        # Replace standalone risk tier that isn't already followed by severity
        # Use regex to avoid replacing inside already-correct strings
        pattern = re.escape(risk_tier) + r'(?!\s*\(' + re.escape(severity_note) + r'\))'
        replacement_str = risk_with_severity
        content = re.sub(pattern, replacement_str, content)
    if content != original:
        modified_files.add(str(xf))
        xf.write_text(content, encoding='utf-8')

print(f'Modified files: {modified_files}')

# --- Phase 2: Remove layout cache (lineSegArray) from modified paragraphs ---
# Parse each modified XML and remove <hp:lineSegArray> or <lineSegArray> elements
for xf_path in [Path(p) for p in modified_files]:
    content = xf_path.read_bytes()
    try:
        tree = etree.fromstring(content)
    except Exception:
        # Try parsing as full document
        tree = etree.parse(str(xf_path)).getroot()
    
    nsmap = tree.nsmap
    # Find all lineSegArray elements regardless of namespace
    removed = 0
    for elem in tree.iter():
        local = etree.QName(elem.tag).localname if isinstance(elem.tag, str) else ''
        if local == 'lineSegArray':
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)
                removed += 1
    
    if removed > 0:
        # Write back
        # Preserve XML declaration if present
        decl = b''
        first_line = content.split(b'\n')[0]
        if first_line.startswith(b'<?xml'):
            decl = first_line + b'\n'
        body = etree.tostring(tree, xml_declaration=False, encoding='unicode')
        xf_path.write_text(decl.decode('utf-8', errors='replace') + body, encoding='utf-8')
        print(f'Removed {removed} lineSegArray elements from {xf_path}')

# --- Phase 3: Verify no remaining placeholders ---
for xf in xml_files:
    content = xf.read_text(encoding='utf-8')
    remaining = re.findall(r'\{\{[^}]+\}\}', content)
    if remaining:
        print(f'WARNING: Remaining placeholders in {xf}: {remaining}')

# --- Phase 4: Repackage as HWPX (ZIP) ---
# HWPX must be a valid ZIP. Preserve the original structure.
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root_dir, dirs, files in os.walk(work_dir):
        for fname in files:
            full_path = os.path.join(root_dir, fname)
            arcname = os.path.relpath(full_path, work_dir)
            zout.write(full_path, arcname)

print(f'Output written to {OUTPUT}')
print(f'Output size: {os.path.getsize(OUTPUT)} bytes')

# Quick validation
with zipfile.ZipFile(OUTPUT, 'r') as z:
    print(f'ZIP entries: {z.namelist()}')
    # Check no placeholders in any XML
    for name in z.namelist():
        if name.endswith('.xml'):
            data = z.read(name).decode('utf-8', errors='replace')
            phs = re.findall(r'\{\{[^}]+\}\}', data)
            if phs:
                print(f'ERROR: Placeholders remain in {name}: {phs}')
```

IMPORTANT: The script above is a starting framework. Before writing it, you MUST first inspect the actual JSON files and XML content to understand the exact placeholder names, field names, and structure. Adapt the script based on what you find. The key points are:

- Map every `{{...}}` placeholder to the correct value from the JSON files.
- For corrective actions, fill them in the SAME ORDER as they appear in `corrective_actions.json`.
- Replace the date format from `YYYY-MM-DD` to `YYYY.MM.DD` everywhere.
- Append severity note after risk tier: e.g., `High (즉시조치)` — use parentheses format.
- Remove `lineSegArray` elements from any paragraph whose text was modified.
- Ensure no `{{...}}` placeholders remain.
- Repackage as a valid ZIP/HWPX.

## 5. Run the script
```bash
cd /tmp/hwpx_work
python3 build.py
```
Review the output carefully. If there are warnings about remaining placeholders, debug and fix.

## 6. Validate the output
```bash
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
for name in z.namelist():
    if name.endswith('.xml'):
        data = z.read(name).decode('utf-8')
        phs = re.findall(r'\{\{[^}]+\}\}', data)
        if phs:
            print(f'FAIL: {name} has placeholders: {phs}')
        if 'lineSegArray' in data:
            print(f'WARN: {name} still has lineSegArray')
print('Validation complete')
"
```

## 7. Run the verifier if available
```bash
cd /root
if [ -f test_output.py ]; then python3 -m pytest test_output.py -v; fi
```
If tests fail, read the error messages, fix the script, and re-run until all tests pass.

## Critical Reminders
- The severity note format MUST use parentheses: `High (즉시조치)`, `Medium (계획보완)`, `Low (모니터링)`.
- Date format MUST be `YYYY.MM.DD` (dots, not hyphens) everywhere.
- ALL `{{...}}` placeholders must be replaced — none may remain.
- `lineSegArray` elements must be removed from modified paragraphs.
- The output file MUST be at exactly `/root/safety_audit_brief_final.hwpx`.
- The output MUST be a valid ZIP file (HWPX format).
- Keep existing section titles and row labels unchanged.

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