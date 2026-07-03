# Task Instruction

Produce the file `/root/clinic_intake_ready.hwpx` by completing the clinic-intake template with patient data.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```
Identify the template (`clinic_intake_template.hwpx`) and data file (`patient_intake.json`). Read the JSON:
```bash
cat /root/patient_intake.json
```
(or wherever it is located)

### 2. Understand the HWPX structure
A `.hwpx` file is a ZIP archive. Unzip the template into a working directory:
```bash
mkdir -p /tmp/hwpx_work
cp /root/clinic_intake_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
Identify the XML files that contain document body text. Typically these are under `Contents/` (e.g., `Contents/section0.xml`). Read each XML file and look for `{{...}}` placeholders:
```bash
grep -rn '{{' template_contents/
```
Also look for `<hp:linesegarray` or `<hp:lineSegArray` elements (note namespace):
```bash
grep -rn 'linesegarray\|lineSegArray\|LineSeg' template_contents/
```
Record the exact namespace prefix and element names used.

### 3. Write a Python script to do the transformation
Create `/tmp/hwpx_work/fill_template.py` with the following logic:

```python
import json, os, re, shutil, zipfile
from lxml import etree

# --- Paths ---
TEMPLATE = '/tmp/hwpx_work/template.zip'
WORK_DIR = '/tmp/hwpx_work/template_contents'
OUTPUT = '/root/clinic_intake_ready.hwpx'
DATA_FILE = '<path_to_patient_intake.json>'  # fill in actual path

# --- Load patient data ---
with open(DATA_FILE) as f:
    data = json.load(f)

# --- Prepare replacement values ---
# Phone normalization: strip everything non-digit, then format 000-0000-0000
raw_phone = data.get('callback_phone', '') or data.get('phone', '') or ''
digits = re.sub(r'\D', '', raw_phone)
if len(digits) == 11:
    formatted_phone = f'{digits[:3]}-{digits[3:7]}-{digits[7:]}'
elif len(digits) == 10:
    formatted_phone = f'{digits[:3]}-{digits[3:6]}-{digits[6:]}'
else:
    formatted_phone = digits  # fallback

# Age calculation: Korean full-year age as of visit date
from datetime import date
birth_str = data.get('birth_date', '') or data.get('date_of_birth', '') or ''
visit_str = data.get('visit_date', '') or ''
birth_date = date.fromisoformat(birth_str)
visit_date = date.fromisoformat(visit_str)
age = visit_date.year - birth_date.year - ((visit_date.month, visit_date.day) < (birth_date.month, birth_date.day))

# Build the replacement map: keys are placeholder names (without braces), values are replacement strings
# IMPORTANT: inspect the JSON keys and the placeholders carefully to build this map.
# The birth_date value should be followed by the age note: "1990-11-02 (35세)" etc.
# Map every {{KEY}} found in the XML to the correct value from the JSON.
# For the birth date placeholder, append the age note.

replacements = {}  # will be filled after inspecting actual placeholders and JSON keys
# Example pattern:
# replacements['patient_name'] = data['patient_name']
# replacements['birth_date'] = f"{birth_str} ({age}세)"
# replacements['callback_phone'] = formatted_phone
# ... etc for every key

# --- Process XML files ---
for root_dir, dirs, files in os.walk(WORK_DIR):
    for fname in files:
        fpath = os.path.join(root_dir, fname)
        if not fname.endswith('.xml'):
            continue
        with open(fpath, 'rb') as f:
            raw = f.read()
        text = raw.decode('utf-8')
        if '{{' not in text and 'linesegarray' not in text.lower():
            continue

        # Parse XML
        tree = etree.fromstring(raw)
        nsmap = tree.nsmap
        # Determine hp namespace URI
        hp_ns = None
        for prefix, uri in nsmap.items():
            if prefix == 'hp' or 'hancom' in uri.lower() or 'hwp' in uri.lower():
                hp_ns = uri
                break
        if hp_ns is None:
            # Try to find from element tags
            for elem in tree.iter():
                if 'linesegarray' in elem.tag.lower() or 'lineSegArray' in elem.tag:
                    hp_ns = elem.tag.split('}')[0].lstrip('{')
                    break

        # Find all text nodes, do placeholder replacement, track modified <hp:p> elements
        modified_paragraphs = set()
        for elem in tree.iter():
            if elem.text and '{{' in elem.text:
                original = elem.text
                for key, val in replacements.items():
                    elem.text = elem.text.replace('{{' + key + '}}', val)
                if elem.text != original:
                    # Walk up to find the paragraph ancestor
                    # We need parent map
                    pass  # handled below
            if elem.tail and '{{' in elem.tail:
                original = elem.tail
                for key, val in replacements.items():
                    elem.tail = elem.tail.replace('{{' + key + '}}', val)

        # Build parent map for linesegarray removal
        parent_map = {c: p for p in tree.iter() for c in p}

        # Re-scan to find modified paragraphs (paragraphs containing replaced text)
        # Actually, simpler approach: after replacement, remove ALL linesegarray
        # elements from ANY paragraph that was modified. Since we replaced text,
        # the safest approach is to remove linesegarray from ALL paragraphs
        # (the viewer will regenerate them). This is safe and avoids missing any.
        # Remove all linesegarray elements regardless.
        to_remove = []
        for elem in tree.iter():
            local = etree.QName(elem).localname.lower()
            if local == 'linesegarray':
                to_remove.append(elem)
        for elem in to_remove:
            parent = parent_map.get(elem)
            if parent is not None:
                parent.remove(elem)

        # Write back
        with open(fpath, 'wb') as f:
            f.write(etree.tostring(tree, xml_declaration=True, encoding='UTF-8'))

# --- Verify no placeholders remain ---
for root_dir, dirs, files in os.walk(WORK_DIR):
    for fname in files:
        if fname.endswith('.xml'):
            fpath = os.path.join(root_dir, fname)
            with open(fpath) as f:
                content = f.read()
            if '{{' in content:
                print(f'WARNING: leftover placeholder in {fpath}')
                # Find and print them
                for m in re.finditer(r'\{\{[^}]*\}\}', content):
                    print(f'  {m.group()}')

# --- Repackage as HWPX (ZIP) ---
# HWPX must preserve the original ZIP structure. Repackage:
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)
with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root_dir, dirs, files in os.walk(WORK_DIR):
        for fname in files:
            fpath = os.path.join(root_dir, fname)
            arcname = os.path.relpath(fpath, WORK_DIR)
            zout.write(fpath, arcname)

print('Done. Output:', OUTPUT)
```

### 4. Critical details to get right

**Placeholder mapping**: After inspecting the actual JSON keys and XML placeholders, build the `replacements` dict precisely. Every `{{SOMETHING}}` in the XML must have a corresponding entry. The patient name placeholder may appear multiple times (including a confirmation line) — the global replace handles this.

**Birth date + age note**: The replacement for the birth-date placeholder must be `"<date> (<age>세)"` — e.g., `"1990-11-02 (35세)"`. The age must be Korean full-year age (Western age, not Korean counting age). For someone born 1990-11-02 with a visit date in 2025 before November 2, the age would be 34; if on or after November 2, it's 35. **Check the visit date carefully.** The previous feedback says the test expects 35, so verify the visit date is on or after 2025-11-02, OR the test uses a different year. Calculate precisely.

**Phone normalization**: Strip all non-digits, then format as `NNN-NNNN-NNNN` (Korean mobile format, 11 digits).

**linesegarray removal**: The previous failure was caused by not removing these elements. The safest approach: remove ALL `<hp:linesegarray>` (or whatever the exact tag is) from ALL paragraphs in ALL XML files, not just modified ones. The viewer regenerates them. Use the actual namespace URI found in the document, not a hardcoded string. Also check for `<hp:lineSegArray>` (case variation) — use case-insensitive local name matching.

**IMPORTANT namespace handling**: When searching for elements, use `tree.iter()` and check `etree.QName(elem).localname` case-insensitively. Do NOT rely on `tree.findall()` with a hardcoded namespace prefix — the prefix might differ.

**Placeholder splitting across XML elements**: HWPX (like OOXML) may split a single placeholder like `{{patient_name}}` across multiple `<hp:t>` runs. If `grep -rn '{{' template_contents/` shows placeholders intact within single elements, simple string replacement works. If they're split, you need to concatenate adjacent text runs first. **Check this before writing the final script.**

### 5. Execute and verify
```bash
python3 /tmp/hwpx_work/fill_template.py
```
Then verify:
```bash
# Check it's a valid ZIP
unzip -t /root/clinic_intake_ready.hwpx

# Check no placeholders remain
unzip -p /root/clinic_intake_ready.hwpx | grep -c '{{'
# Should be 0

# Check age note is present
unzip -p /root/clinic_intake_ready.hwpx | grep -o '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\} ([0-9]*세)'

# Check no linesegarray in modified content
unzip -p /root/clinic_intake_ready.hwpx | grep -ci 'linesegarray'
# Should be 0 (or at least 0 in paragraphs with replaced text)

# Check phone format
unzip -p /root/clinic_intake_ready.hwpx | grep -oE '[0-9]{3}-[0-9]{4}-[0-9]{4}'
```

### 6. Run the verifier
```bash
cd /root && python -m pytest tests/ -v
```
If any test fails, read the error, fix the script, and re-run.

### Key warnings from past failures
- The `hp:linesegarray` removal MUST work. Double-check by grepping the output. If any remain in modified paragraphs, the test fails.
- The age note format must be exactly `(<N>세)` with a space before the parenthesis, placed right after the birth date value.
- Every single `{{...}}` must be replaced — check for typos in the replacement keys.
- Preserve all existing Korean text labels and the handwritten-signature note verbatim.

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