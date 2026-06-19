# Task Instruction

Execute the following steps in a single Python script to produce `/root/safety_audit_brief_final.hwpx`.

### Step 0 – Inspect source files
```
cat /root/audit_overview.json
cat /root/corrective_actions.json
```
Then unzip the template and inspect the XML content file:
```
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip -o /root/safety_audit_template.hwpx -d template
find template -type f -name '*.xml' | head -20
```
Read the main content XML (likely `template/Contents/section0.xml`) and print it so you can see every `{{…}}` placeholder, the namespace declarations, the `<hp:lineSegArray>` elements, and the overall structure.

### Step 1 – Write and run the Python processing script

Create `/tmp/hwpx_work/build.py` with the logic below. Adjust the content-XML path if inspection shows a different location.

```python
import json, os, re, shutil, zipfile, glob

# --- paths ---
TEMPLATE_DIR = '/tmp/hwpx_work/template'
OUT_PATH = '/root/safety_audit_brief_final.hwpx'
OVERVIEW = json.load(open('/root/audit_overview.json', encoding='utf-8'))
ACTIONS = json.load(open('/root/corrective_actions.json', encoding='utf-8'))

# --- locate content XML ---
xml_candidates = glob.glob(os.path.join(TEMPLATE_DIR, 'Contents', 'section*.xml'))
assert xml_candidates, 'No section XML found'
XML_PATH = xml_candidates[0]

with open(XML_PATH, 'r', encoding='utf-8') as f:
    xml = f.read()

# --- severity mapping ---
SEVERITY = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}

# --- build replacement map from overview ---
# Flatten overview JSON into {{key}} -> value.  Inspect the JSON keys first;
# they may be nested.  Build the map dynamically.
def flatten(obj, prefix=''):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = k  # use the raw key as placeholder name
            if isinstance(v, (dict, list)):
                items.update(flatten(v, new_key))
            else:
                items[new_key] = str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(flatten(v, f'{prefix}_{i}'))
    return items

overview_map = flatten(OVERVIEW)
print('Overview placeholders:', overview_map)

# --- handle corrective actions (ordered) ---
# They are a list; placeholders are likely {{action_1}}, {{action_2}}, {{action_3}} or similar.
# We will discover the exact placeholder names from the XML.
print('Corrective actions:', ACTIONS)

# --- consolidate split placeholders ---
# HWPX may split {{placeholder}} across multiple <hp:t> tags.
# Strategy: within each <hp:p>…</hp:p>, concatenate all <hp:t> text,
# do replacements on the concatenated string, then put it back into a single <hp:t>.
# But safer: do string-level consolidation of adjacent </hp:t><hp:t> fragments
# that together form a {{…}} token.

# First, remove all mid-placeholder tag splits:
# e.g. {{place</hp:t></hp:run>...<hp:run>...<hp:t>holder}} -> {{placeholder}}
# Approach: strip tags inside {{ }} by iterating.

def consolidate_placeholders(text):
    """Remove XML tags that break up {{...}} tokens."""
    # Repeatedly find {{ ... }} spans that contain tags and remove the tags
    pattern = re.compile(r'(\{\{)(.*?)(\}\})', re.DOTALL)
    def clean_match(m):
        inner = re.sub(r'<[^>]+>', '', m.group(2))
        return '{{' + inner.strip() + '}}'
    prev = None
    while prev != text:
        prev = text
        text = pattern.sub(clean_match, text)
    return text

xml = consolidate_placeholders(xml)

# --- discover all remaining placeholders ---
all_ph = re.findall(r'\{\{([^}]+)\}\}', xml)
print('Placeholders found after consolidation:', all_ph)

# --- build full replacement map ---
replace_map = {}
for k, v in overview_map.items():
    replace_map[k] = v

# For corrective actions: match placeholder names to the ordered list
# Sort action-related placeholders by their natural order in the XML
action_phs = [ph for ph in all_ph if ph not in overview_map]
# Also try matching by index
for ph in all_ph:
    for ak in overview_map:
        if ak in ph:
            # already covered
            pass

# If actions is a list of dicts or strings, handle accordingly
if isinstance(ACTIONS, list):
    # Find action placeholders in document order
    action_placeholders = []
    for ph in all_ph:
        if ph not in replace_map:
            action_placeholders.append(ph)
    # Remove duplicates preserving order
    seen = set()
    ordered_action_ph = []
    for p in action_placeholders:
        if p not in seen:
            seen.add(p)
            ordered_action_ph.append(p)
    print('Action placeholders (ordered):', ordered_action_ph)
    
    # Map each action placeholder to the corresponding action
    for i, ph in enumerate(ordered_action_ph):
        if i < len(ACTIONS):
            act = ACTIONS[i]
            if isinstance(act, dict):
                # Join all values or use a specific key
                replace_map[ph] = ' / '.join(str(v) for v in act.values())
            else:
                replace_map[ph] = str(act)

# --- handle risk_tier + severity note ---
# The risk tier key in overview might be 'risk_tier', 'risk_level', etc.
risk_key = None
risk_value = None
for k, v in overview_map.items():
    if 'risk' in k.lower():
        risk_key = k
        risk_value = v
        break

if risk_value and risk_value in SEVERITY:
    severity_note = SEVERITY[risk_value]
    # The test expects: 'High (즉시조치)' — with parentheses
    combined = f'{risk_value} ({severity_note})'
    # Replace in the map so the placeholder gets the combined value
    if risk_key:
        replace_map[risk_key] = combined
    # Also need to replace ANY already-substituted bare risk_value later
    # We'll do a post-replacement pass

print('Full replacement map:', replace_map)

# --- perform replacements ---
for ph, val in replace_map.items():
    xml = xml.replace('{{' + ph + '}}', val)

# --- date rewriting: YYYY-MM-DD -> YYYY.MM.DD everywhere ---
xml = re.sub(r'(\d{4})-(\d{2})-(\d{2})', r'\1.\2.\3', xml)

# --- ensure ALL occurrences of bare risk tier are updated with severity note ---
if risk_value and risk_value in SEVERITY:
    severity_note = SEVERITY[risk_value]
    combined = f'{risk_value} ({severity_note})'
    # Find bare risk_value inside <hp:t> tags that don't already have the note
    # Replace only inside text content, not in tag attributes
    def add_severity(m):
        text = m.group(1)
        # Replace bare risk_value not already followed by the note
        text = re.sub(
            re.escape(risk_value) + r'(?!\s*\(' + re.escape(severity_note) + r'\))',
            combined,
            text
        )
        return '<hp:t>' + text + '</hp:t>'
    xml = re.sub(r'<hp:t>(.*?)</hp:t>', add_severity, xml, flags=re.DOTALL)

# --- verify no remaining placeholders ---
remaining = re.findall(r'\{\{[^}]+\}\}', xml)
if remaining:
    print('WARNING: remaining placeholders:', remaining)
    # Force-clear them
    for r in remaining:
        xml = xml.replace(r, '')

# --- remove layout cache from modified paragraphs ---
# Remove ALL <hp:lineSegArray>...</hp:lineSegArray> to be safe
xml = re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', xml, flags=re.DOTALL)
# Also handle self-closing
xml = re.sub(r'<hp:lineSegArray[^/]*/>', '', xml)

# --- write back ---
with open(XML_PATH, 'w', encoding='utf-8') as f:
    f.write(xml)

print('XML written successfully.')

# --- repackage as .hwpx (ZIP) ---
# Must zip from inside the template dir so paths are relative
if os.path.exists(OUT_PATH):
    os.remove(OUT_PATH)

with zipfile.ZipFile(OUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(TEMPLATE_DIR):
        for fn in files:
            abs_path = os.path.join(root, fn)
            arc_name = os.path.relpath(abs_path, TEMPLATE_DIR)
            # mimetype should be stored, not deflated
            if fn == 'mimetype':
                zf.write(abs_path, arc_name, compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(abs_path, arc_name)

print(f'Output written to {OUT_PATH}')

# --- final verification ---
# Re-read and check
with zipfile.ZipFile(OUT_PATH, 'r') as zf:
    for name in zf.namelist():
        if name.endswith('.xml') and 'section' in name:
            content = zf.read(name).decode('utf-8')
            assert '{{' not in content, f'Leftover placeholder in {name}'
            if risk_value:
                combined = f'{risk_value} ({SEVERITY[risk_value]})'
                assert combined in content, f'Missing combined risk+severity: {combined}'
            # Check date format
            if re.search(r'\d{4}-\d{2}-\d{2}', content):
                print('WARNING: YYYY-MM-DD date still present')
            print(f'Verification of {name}: OK')
            # Print a snippet around risk tier for debugging
            idx = content.find(risk_value) if risk_value else -1
            if idx >= 0:
                print(f'  Risk context: ...{content[max(0,idx-30):idx+60]}...')

print('All checks passed.')
```

### Step 2 – Run the script
```bash
cd /tmp/hwpx_work && python3 build.py
```

### Step 3 – Troubleshoot and iterate
- If any placeholder names don't match, re-inspect the XML and JSON, adjust the mapping, and re-run.
- If the corrective actions JSON has nested structure (e.g., each action is a dict with fields like `area`, `finding`, `action`), inspect the placeholder names in the XML and map each field to its corresponding placeholder.
- If the verification assertions fail, print the relevant XML region, diagnose, fix, and re-run.
- Make sure the final file at `/root/safety_audit_brief_final.hwpx` passes all assertions.

### Critical Requirements (from prior feedback)
1. **Severity note format**: Use parentheses — `High (즉시조치)` not `High 즉시조치`. The test asserts the parenthesized form.
2. **Date format**: Every `YYYY-MM-DD` must become `YYYY.MM.DD`.
3. **Risk tier everywhere**: Every occurrence of the bare risk tier in `<hp:t>` text must be updated to include the severity note.
4. **Layout cache**: Remove all `<hp:lineSegArray>` elements.
5. **No leftover `{{…}}`** placeholders.
6. **Valid HWPX ZIP** with correct relative paths and mimetype stored uncompressed.
7. **Corrective actions in order**: The three actions must appear in the same order as in `corrective_actions.json`.
8. **Preserve section titles and row labels**: Only replace placeholder values, don't alter structural text.

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