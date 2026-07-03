# Task Instruction

Execute the following steps in order to produce `/root/safety_audit_brief_final.hwpx`.

## 1. Inspect the workspace

```bash
cd /root
ls -la
find . -maxdepth 2 -type f | head -60
```

Identify the template file `safety_audit_template.hwpx`, the two JSON data files (`audit_overview.json`, `corrective_actions.json`), and the test file (`test_output.py` or `test_outputs.py`).

## 2. Read the test file first

```bash
cat /root/test_output*.py
```

Understand every assertion the verifier makes. Pay special attention to:
- The exact format of the risk tier + severity note (the test expects `'High (즉시조치)'` with parentheses, not just `'High 즉시조치'`).
- Date format checks (`YYYY.MM.DD`).
- Whether it checks for absence of `{{` placeholders.
- Which XML files inside the HWPX it reads (section0.xml, section1.xml, etc.).
- Any ordering or content checks for corrective actions.

## 3. Read the data files

```bash
cat /root/audit_overview.json
cat /root/corrective_actions.json
```

Note all field names and values.

## 4. Explore the HWPX template

HWPX is a ZIP archive. Extract it to inspect its structure:

```bash
mkdir -p /root/hwpx_work
cp /root/safety_audit_template.hwpx /root/hwpx_work/template.zip
cd /root/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```

Read each section XML file (typically under `Contents/`):

```bash
for f in $(find template_contents -name 'section*.xml'); do echo "=== $f ==="; cat "$f"; echo; done
```

Also check for any other XML files that might contain text or metadata.

## 5. Write a Python script to perform all substitutions

Create `/root/build_hwpx.py` with the following logic:

```python
import json, zipfile, os, re, shutil, copy
from lxml import etree

# --- Load data ---
with open('/root/audit_overview.json', 'r', encoding='utf-8') as f:
    overview = json.load(f)
with open('/root/corrective_actions.json', 'r', encoding='utf-8') as f:
    actions = json.load(f)

# --- Severity mapping ---
severity_map = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}

# --- Determine risk tier and severity note ---
risk_tier = overview.get('risk_tier') or overview.get('risk_level') or overview.get('riskTier', '')
# Search all keys for risk-related field if not found
if not risk_tier:
    for k, v in overview.items():
        if 'risk' in k.lower() and isinstance(v, str) and v in severity_map:
            risk_tier = v
            break
severity_note = severity_map.get(risk_tier, '')
risk_with_note = f'{risk_tier} ({severity_note})'

# --- Determine inspection date and reformat ---
date_raw = overview.get('inspection_date') or overview.get('inspectionDate') or overview.get('date', '')
if not date_raw:
    for k, v in overview.items():
        if 'date' in k.lower() and isinstance(v, str) and re.match(r'\d{4}-\d{2}-\d{2}', v):
            date_raw = v
            break
date_reformatted = date_raw.replace('-', '.')

# --- Extract template ---
work_dir = '/root/hwpx_build'
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)
os.makedirs(work_dir)
with zipfile.ZipFile('/root/safety_audit_template.hwpx', 'r') as zf:
    zf.extractall(work_dir)

# --- Build placeholder map from overview ---
# We'll replace all {{key}} patterns with corresponding values.
# Also handle risk tier specially.
placeholder_map = {}
for k, v in overview.items():
    placeholder_map[k] = str(v)

# For corrective actions, we need positional replacement
# They appear as {{corrective_action_1}}, {{corrective_action_2}}, etc. or similar
# We'll discover the actual placeholder names from the XML.

def process_xml_file(filepath):
    """Process a single XML file: replace placeholders, fix dates, add severity notes."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    
    # Find all placeholders
    placeholders_found = re.findall(r'\{\{([^}]+)\}\}', content)
    print(f'  Placeholders in {os.path.basename(filepath)}: {placeholders_found}')
    
    # Replace overview placeholders
    for k, v in placeholder_map.items():
        placeholder = '{{' + k + '}}'
        if placeholder in content:
            # Special handling for date fields
            if 'date' in k.lower():
                v_replaced = v.replace('-', '.')
                content = content.replace(placeholder, v_replaced)
            # Special handling for risk tier
            elif 'risk' in k.lower():
                content = content.replace(placeholder, risk_with_note)
            else:
                content = content.replace(placeholder, v)
    
    # Replace corrective action placeholders
    # Try various naming patterns
    if isinstance(actions, list):
        for i, action in enumerate(actions):
            action_text = ''
            if isinstance(action, dict):
                # Try common keys
                for key in ['description', 'action', 'text', 'content', 'corrective_action', 'measure']:
                    if key in action:
                        action_text = str(action[key])
                        break
                if not action_text:
                    # Use first string value
                    for v in action.values():
                        if isinstance(v, str) and len(v) > 5:
                            action_text = v
                            break
            else:
                action_text = str(action)
            
            # Try multiple placeholder patterns
            for pattern in [
                '{{corrective_action_' + str(i+1) + '}}',
                '{{action_' + str(i+1) + '}}',
                '{{corrective_' + str(i+1) + '}}',
                '{{item_' + str(i+1) + '}}',
            ]:
                content = content.replace(pattern, action_text)
    
    # Replace any remaining {{...}} with values from actions if they are dicts
    remaining = re.findall(r'\{\{([^}]+)\}\}', content)
    if remaining and isinstance(actions, list):
        for ph in remaining:
            # Check if it matches an action field pattern
            for i, action in enumerate(actions):
                if isinstance(action, dict):
                    for k, v in action.items():
                        if k in ph or ph in k:
                            content = content.replace('{{' + ph + '}}', str(v))
    
    # Now handle non-placeholder occurrences:
    # 1. Replace ALL occurrences of the raw risk tier with risk_with_note
    #    But be careful not to double-replace
    if risk_tier and risk_tier in content and risk_with_note not in content:
        content = content.replace(risk_tier, risk_with_note)
    
    # 2. Replace ALL date occurrences from YYYY-MM-DD to YYYY.MM.DD
    if date_raw:
        content = content.replace(date_raw, date_reformatted)
    # Also catch any other YYYY-MM-DD patterns that match
    content = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1.\2.\3', content)
    
    # 3. Remove stale layout cache elements (hp:linesegarray, hp:lineSegArray, etc.)
    # These cause overlapping characters when text length changes
    # Parse as XML and remove them
    try:
        tree = etree.fromstring(content.encode('utf-8'))
        nsmap = tree.nsmap
        # Find and remove lineSegArray / linesegarray elements
        for tag_local in ['linesegarray', 'lineSegArray', 'LineSeg', 'lineseg']:
            for ns_prefix, ns_uri in nsmap.items():
                if ns_uri:
                    for elem in tree.iter('{' + ns_uri + '}' + tag_local):
                        parent = elem.getparent()
                        if parent is not None:
                            parent.remove(elem)
            # Also try without namespace
            for elem in tree.iter(tag_local):
                parent = elem.getparent()
                if parent is not None:
                    parent.remove(elem)
        
        # Re-serialize
        content = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', pretty_print=False).decode('utf-8')
    except Exception as e:
        print(f'  XML parse warning for {filepath}: {e}')
        # Fallback: regex removal of linesegarray elements
        content = re.sub(r'<[^>]*[Ll]ine[Ss]eg[^>]*>.*?</[^>]*[Ll]ine[Ss]eg[^>]*>', '', content, flags=re.DOTALL)
        content = re.sub(r'<[^>]*[Ll]ine[Ss]eg[^/]*/>', '', content)
    
    # Verify no remaining placeholders
    remaining_final = re.findall(r'\{\{([^}]+)\}\}', content)
    if remaining_final:
        print(f'  WARNING: remaining placeholders in {filepath}: {remaining_final}')
    
    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'  Modified: {os.path.basename(filepath)}')
    else:
        print(f'  Unchanged: {os.path.basename(filepath)}')

# --- Process all XML files ---
for root_dir, dirs, files in os.walk(work_dir):
    for fname in files:
        if fname.endswith('.xml'):
            fpath = os.path.join(root_dir, fname)
            process_xml_file(fpath)

# --- Repackage as HWPX ---
output_path = '/root/safety_audit_brief_final.hwpx'
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root_dir, dirs, files in os.walk(work_dir):
        for fname in files:
            fpath = os.path.join(root_dir, fname)
            arcname = os.path.relpath(fpath, work_dir)
            zf.write(fpath, arcname)

print(f'\nOutput written to {output_path}')

# --- Verify ---
with zipfile.ZipFile(output_path, 'r') as zf:
    for name in zf.namelist():
        if name.endswith('.xml') and 'section' in name.lower():
            data = zf.read(name).decode('utf-8')
            if '{{' in data:
                print(f'FAIL: placeholders remain in {name}')
            if risk_with_note in data:
                print(f'OK: risk+severity found in {name}: {risk_with_note}')
            if date_reformatted in data:
                print(f'OK: reformatted date found in {name}: {date_reformatted}')
            if date_raw in data:
                print(f'WARN: raw date still in {name}: {date_raw}')
```

**IMPORTANT**: Before running this script, you MUST first inspect the actual JSON keys and XML placeholder names. Adjust the script's key lookups to match the real data. The script above is a starting framework—adapt it after inspection.

## 6. Run the script

```bash
cd /root
python3 build_hwpx.py
```

Review the output carefully. Check for:
- All placeholders replaced
- Risk tier appears as `'High (즉시조치)'` (or whatever the actual tier is, with parenthesized severity)
- Dates in `YYYY.MM.DD` format
- No `{{...}}` remaining

## 7. Validate with the test suite

```bash
cd /root
python3 -m pytest test_output*.py -v
```

If any test fails, read the error message carefully, inspect the relevant section XML from the output HWPX, and fix the script accordingly. Common issues to watch for:

- **Risk tier format**: Must be `'Value (Severity)'` with parentheses and a space before the opening paren.
- **Date format**: ALL occurrences must be `YYYY.MM.DD`, not just placeholder ones.
- **Corrective actions**: Must appear in the same order as in the JSON.
- **Remaining placeholders**: Any `{{...}}` text means a placeholder name didn't match.
- **Layout cache**: `linesegarray` or similar elements must be removed from modified paragraphs.

## 8. Iterate if needed

If tests fail, re-examine the section XMLs inside the output HWPX:

```bash
mkdir -p /root/verify_output
cd /root/verify_output
unzip -o /root/safety_audit_brief_final.hwpx
for f in $(find . -name 'section*.xml'); do echo "=== $f ==="; cat "$f"; echo; done
```

Compare against what the test expects and fix the build script. Then re-run steps 6-7.

## Critical Reminders

- The severity note format MUST use parentheses: `Risk (Severity)` e.g. `High (즉시조치)`. This was the exact failure in the previous run.
- Read the test file BEFORE writing the build script so you know the exact assertions.
- Inspect actual JSON keys and XML placeholders BEFORE hardcoding any names.
- The HWPX must be a valid ZIP with the same internal structure as the template.
- Do NOT leave any `{{...}}` placeholders in the output.

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