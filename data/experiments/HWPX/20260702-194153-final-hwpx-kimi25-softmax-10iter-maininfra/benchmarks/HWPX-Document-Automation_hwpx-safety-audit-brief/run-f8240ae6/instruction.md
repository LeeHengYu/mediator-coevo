# Task Instruction

Execute the following steps in order to produce `/root/safety_audit_brief_final.hwpx`.

## 1. Inspect the workspace

```bash
ls /root/
find /root/ -name '*.json' -o -name '*.hwpx' | head -30
```

Identify the exact paths of `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

## 2. Inspect the JSON data files

```bash
cat <path_to>/audit_overview.json
cat <path_to>/corrective_actions.json
```

Note every field value. Pay special attention to:
- The inspection date (in `YYYY-MM-DD` format — you will rewrite it to `YYYY.MM.DD` everywhere).
- The risk tier value (e.g. `High`, `Medium`, or `Low`).
- All overview fields and audit-table values.
- The three corrective-action entries and their order.

## 3. Unpack the HWPX template

HWPX is a ZIP archive. Unpack it to inspect and edit its XML contents:

```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip <path_to>/safety_audit_template.hwpx -d template_unpacked
find template_unpacked -type f | sort
```

## 4. Inspect the section XML files

Look at every section XML (typically `Contents/section0.xml`, possibly more sections):

```bash
cat template_unpacked/Contents/section0.xml
```

(Repeat for section1.xml etc. if they exist.)

Identify:
- All `{{...}}` placeholder tokens and their exact spelling.
- The XML namespace for `hp:` elements (likely `http://www.hancom.co.kr/hwpml/2010/HWPML` or similar).
- The structure of `<hp:p>` paragraphs, `<hp:run>`, `<hp:t>` text elements.
- The presence of `<hp:lineSegArray>` (or `<hp:linesegarray>` in the namespace) elements inside `<hp:p>` tags.

## 5. Write and run the Python transformation script

Create `/tmp/hwpx_work/transform.py` with the following logic:

```python
import json
import os
import re
import shutil
import zipfile
from lxml import etree

# --- Paths (adjust after inspection) ---
TEMPLATE_DIR = '/tmp/hwpx_work/template_unpacked'
OVERVIEW_JSON = '<path_to>/audit_overview.json'
ACTIONS_JSON = '<path_to>/corrective_actions.json'
OUTPUT_HWPX = '/root/safety_audit_brief_final.hwpx'

# --- Load JSON data ---
with open(OVERVIEW_JSON) as f:
    overview = json.load(f)
with open(ACTIONS_JSON) as f:
    actions = json.load(f)  # list of 3 items, preserve order

# --- Build replacement map ---
# Map each {{placeholder}} to its replacement value.
# You MUST inspect the actual placeholders in the XML and map them here.
# Example (adjust after inspection):
# replacements = {
#     '{{facility_name}}': overview['facility_name'],
#     '{{inspector}}': overview['inspector'],
#     ...
# }
# For the inspection date: rewrite from YYYY-MM-DD to YYYY.MM.DD
# For risk tier: append severity note using mapping High->즉시조치, Medium->계획보완, Low->모니터링

SEVERITY_MAP = {'High': '즉시조치', 'Medium': '계획보완', 'Low': '모니터링'}

# Build replacements dict from actual placeholder names found in XML.
# ... (fill in after inspecting the XML)

# --- Process each section XML ---
NS = {}  # will be populated from XML

def process_section(xml_path):
    tree = etree.parse(xml_path)
    root = tree.getroot()
    
    # Collect namespaces
    nsmap = {}
    for prefix, uri in root.nsmap.items():
        if prefix:
            nsmap[prefix] = uri
    
    # Find all text elements
    modified_paragraphs = set()
    
    for t_elem in root.iter():
        if t_elem.text and '{{' in t_elem.text:
            original = t_elem.text
            new_text = original
            for placeholder, value in replacements.items():
                if placeholder in new_text:
                    new_text = new_text.replace(placeholder, value)
            if new_text != original:
                t_elem.text = new_text
                # Track the parent <hp:p> for lineSegArray removal
                p = t_elem
                while p is not None:
                    tag_local = etree.QName(p.tag).localname if isinstance(p.tag, str) else ''
                    if tag_local == 'p':
                        modified_paragraphs.add(p)
                        break
                    p = p.getparent()
        # Also check .tail
        if t_elem.tail and '{{' in t_elem.tail:
            original = t_elem.tail
            new_text = original
            for placeholder, value in replacements.items():
                if placeholder in new_text:
                    new_text = new_text.replace(placeholder, value)
            if new_text != original:
                t_elem.tail = new_text
                p = t_elem
                while p is not None:
                    tag_local = etree.QName(p.tag).localname if isinstance(p.tag, str) else ''
                    if tag_local == 'p':
                        modified_paragraphs.add(p)
                        break
                    p = p.getparent()
    
    # Also do a second pass for risk tier and date that may not be in {{}} form
    # (in case they were already partially filled or appear as literal values)
    # This handles the 'every occurrence' requirement.
    
    # Remove lineSegArray from ALL modified paragraphs
    for p_elem in modified_paragraphs:
        for child in list(p_elem):
            local = etree.QName(child.tag).localname if isinstance(child.tag, str) else ''
            if local.lower() == 'linesegarray':
                p_elem.remove(child)
    
    # Verify no {{...}} remain
    xml_str = etree.tostring(root, encoding='unicode')
    leftover = re.findall(r'\{\{[^}]+\}\}', xml_str)
    if leftover:
        print(f'WARNING: leftover placeholders in {xml_path}: {leftover}')
    
    # Write back
    tree.write(xml_path, xml_declaration=True, encoding='UTF-8')

# Process all section files
for fname in sorted(os.listdir(os.path.join(TEMPLATE_DIR, 'Contents'))):
    if fname.startswith('section') and fname.endswith('.xml'):
        process_section(os.path.join(TEMPLATE_DIR, 'Contents', fname))

# --- Repack as HWPX (ZIP) ---
# HWPX must be repacked as a ZIP preserving the directory structure.
with zipfile.ZipFile(OUTPUT_HWPX, 'w', zipfile.ZIP_DEFLATED) as zf:
    for dirpath, dirnames, filenames in os.walk(TEMPLATE_DIR):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            arcname = os.path.relpath(full, TEMPLATE_DIR)
            zf.write(full, arcname)

print('Done:', OUTPUT_HWPX)
```

**CRITICAL**: The above is a skeleton. After inspecting the actual XML placeholders and JSON keys in steps 2 and 4, fill in the `replacements` dictionary precisely. The placeholder names must match exactly what's in the XML.

### Key requirements to implement in the script:

1. **Overview fields**: Replace each `{{...}}` placeholder in the summary/overview section with the corresponding value from `audit_overview.json`.

2. **Audit table values**: Replace each `{{...}}` placeholder in the audit table value cells.

3. **Corrective actions**: Fill the three corrective-action lines in the exact order from `corrective_actions.json`.

4. **Risk tier**: Replace EVERY occurrence of the risk tier placeholder. After each risk tier value, append the severity note with a space, e.g., if risk is `High`, the text becomes `High 즉시조치`.

5. **Date reformat**: The inspection date from JSON is `YYYY-MM-DD`. Every occurrence in the output must be `YYYY.MM.DD`. Replace the placeholder with the reformatted date, AND do a global search-replace across all text nodes to convert any `YYYY-MM-DD` to `YYYY.MM.DD`.

6. **No leftover placeholders**: Assert no `{{...}}` text remains anywhere.

7. **Remove `hp:lineSegArray`** from EVERY `<hp:p>` paragraph that had ANY text modification. This is the most critical requirement from the feedback. The localname comparison should be case-insensitive (`linesegarray`, `lineSegArray`, `LINESEGARRAY` — check for all). Walk up from modified text nodes to find the enclosing `<hp:p>` and remove any child whose local name is `linesegarray` (case-insensitive).

8. **Preserve section titles and row labels** — only replace `{{...}}` placeholders; do not alter other text.

## 6. Run the script

```bash
cd /tmp/hwpx_work
python3 transform.py
```

## 7. Validate the output

```bash
# Check it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx'); z.testzip(); print('Valid ZIP'); z.printdir()"

# Check no leftover placeholders
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
for name in z.namelist():
    if name.endswith('.xml'):
        data = z.read(name).decode('utf-8')
        import re
        found = re.findall(r'\{\{[^}]+\}\}', data)
        if found:
            print(f'FAIL: {name} has placeholders: {found}')
        # Check no YYYY-MM-DD dates remain (should be YYYY.MM.DD)
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', data)
        if dates:
            print(f'WARNING: {name} still has dash-dates: {dates}')
print('Validation complete')
"

# Check no lineSegArray in modified paragraphs
python3 -c "
import zipfile
from lxml import etree
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx')
for name in z.namelist():
    if 'section' in name and name.endswith('.xml'):
        root = etree.fromstring(z.read(name))
        for p in root.iter():
            if etree.QName(p.tag).localname == 'p':
                texts = ''.join(p.itertext())
                # Check for lineSegArray children
                for child in p:
                    if etree.QName(child.tag).localname.lower() == 'linesegarray':
                        # Only flag if paragraph has data content (not empty)
                        if texts.strip():
                            print(f'POTENTIAL ISSUE: lineSegArray in paragraph with text: {texts[:80]}')
print('lineSegArray check complete')
"
```

## 8. Run the verifier if available

```bash
cd /root
# Look for test files
find . -name 'test_*.py' -o -name 'pytest.ini' -o -name 'Makefile' | head -10
# Run tests
python3 -m pytest tests/ -v 2>&1 | tail -40
```

If tests fail, read the error carefully, fix the script, and re-run. Pay special attention to:
- Exact string matching (the verifier may check for exact formatted strings)
- lineSegArray removal (the #1 failure mode from previous runs)
- Date format (must be YYYY.MM.DD everywhere, no YYYY-MM-DD)
- Risk tier + severity note format (check if space-separated or needs specific formatting)

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