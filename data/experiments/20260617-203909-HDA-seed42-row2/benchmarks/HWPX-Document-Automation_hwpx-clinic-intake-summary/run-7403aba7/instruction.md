# Task Instruction

Execute the following steps to produce `/root/clinic_intake_ready.hwpx` from the template and patient data.

### 1. Inspect the inputs
```bash
cd /root
ls -la
cat patient_intake.json
```
Then inspect the HWPX template structure:
```bash
python3 -c "
import zipfile, sys
with zipfile.ZipFile('clinic_intake_template.hwpx','r') as z:
    for info in z.infolist():
        print(f'{info.compress_type:2d} {info.file_size:>8d} {info.filename}')
"
```
Identify which XML files inside the HWPX contain `{{` placeholders:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('clinic_intake_template.hwpx','r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8','replace')
            if '{{' in data:
                print(f'--- {name} ---')
                # print lines with placeholders
                for i, line in enumerate(data.splitlines()):
                    if '{{' in line:
                        print(f'  L{i}: {line[:300]}')
        except: pass
"
```
Also print the full content of each section XML that has placeholders so you can see the exact tag structure, namespace declarations, and how placeholders are split across `<hp:t>` elements.

### 2. Write and run the generation script
Create `/root/generate.py` with the following logic:

```python
import json, zipfile, os, re, copy
import xml.etree.ElementTree as ET
from datetime import date

# ── Load patient data ──
with open('patient_intake.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# ── Derive computed values ──
# Korean full-year age: visit_year - birth_year, minus 1 if birthday not yet passed
visit_date = date.fromisoformat(data['visit_date'])
birth_date = date.fromisoformat(data['date_of_birth'])
age = visit_date.year - birth_date.year - (
    (visit_date.month, visit_date.day) < (birth_date.month, birth_date.day)
)
age_note = f'({age}세)'

# Normalize phone: strip non-digits, format as 000-0000-0000
raw_phone = re.sub(r'\D', '', data.get('callback_phone', data.get('phone', '')))
if len(raw_phone) == 11:
    norm_phone = f'{raw_phone[:3]}-{raw_phone[3:7]}-{raw_phone[7:]}'
elif len(raw_phone) == 10:
    norm_phone = f'{raw_phone[:3]}-{raw_phone[3:6]}-{raw_phone[6:]}'
else:
    norm_phone = raw_phone  # fallback

# ── Build replacement map ──
# Start with all JSON keys mapped as {{key}}
replacements = {}
for k, v in data.items():
    replacements['{{' + k + '}}'] = str(v)

# Override/add computed values
replacements['{{age_note}}'] = age_note
if 'callback_phone' in data:
    replacements['{{callback_phone}}'] = norm_phone
if 'phone' in data:
    replacements['{{phone}}'] = norm_phone

# ── Register HWPX namespaces ──
# We'll discover them from parsing; register common ones upfront
NAMESPACES = {
    'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hc': 'http://www.hancom.co.kr/hwpml/2011/core',
    'hh': 'http://www.hancom.co.kr/hwpml/2011/head',
    'hs': 'http://www.hancom.co.kr/hwpml/2011/section',
    'config': 'urn:oasis:names:tc:opendocument:xmlns:config:1.0',
    'odf': 'urn:oasis:names:tc:opendocument:xmlns:container',
    'text': 'urn:oasis:names:tc:opendocument:xmlns:text:1.0',
}

def register_all_namespaces(xml_bytes):
    """Parse XML to discover and register all namespace prefixes."""
    import io
    for event, elem in ET.iterparse(io.BytesIO(xml_bytes), events=['start-ns']):
        prefix, uri = elem
        if prefix:
            try:
                ET.register_namespace(prefix, uri)
            except:
                pass
        NAMESPACES[prefix] = uri

def process_xml(xml_bytes):
    """Replace placeholders in an HWPX XML, handling split <hp:t> tags."""
    register_all_namespaces(xml_bytes)
    tree = ET.ElementTree(ET.fromstring(xml_bytes))
    root = tree.getroot()

    # Find the hp namespace URI
    hp_ns = None
    for prefix, uri in NAMESPACES.items():
        if prefix == 'hp' or 'paragraph' in uri:
            hp_ns = uri
            break

    if hp_ns is None:
        # Try alternate: just do string-level replacement
        text = xml_bytes.decode('utf-8')
        for placeholder, value in replacements.items():
            text = text.replace(placeholder, value)
        return text.encode('utf-8'), '{{' in text

    hp_t_tag = f'{{{hp_ns}}}t'
    hp_p_tag = f'{{{hp_ns}}}p'
    hp_run_tag = f'{{{hp_ns}}}run'
    hp_lineseg_tag = f'{{{hp_ns}}}lineSegArray'

    modified_paragraphs = set()

    # Process each paragraph
    for p_elem in root.iter(hp_p_tag):
        # Collect all <hp:t> elements in this paragraph (across runs)
        t_elems = list(p_elem.iter(hp_t_tag))
        if not t_elems:
            continue

        # Concatenate all text
        full_text = ''.join((t.text or '') for t in t_elems)

        if '{{' not in full_text:
            continue

        # Perform replacements
        new_text = full_text
        for placeholder, value in replacements.items():
            new_text = new_text.replace(placeholder, value)

        # Handle age_note insertion after date_of_birth
        # If the birth date appears and age_note is not already there
        dob = data.get('date_of_birth', '')
        if dob and dob in new_text and age_note not in new_text:
            new_text = new_text.replace(dob, dob + ' ' + age_note)

        if new_text != full_text:
            # Put all text into the first <hp:t>, clear the rest
            t_elems[0].text = new_text
            for t in t_elems[1:]:
                t.text = ''
            modified_paragraphs.add(p_elem)

    # Remove lineSegArray from modified paragraphs to invalidate layout cache
    for p_elem in modified_paragraphs:
        for lineseg in list(p_elem.iter(hp_lineseg_tag)):
            parent = None
            # Find parent of lineSegArray
            for parent_candidate in p_elem.iter():
                if lineseg in list(parent_candidate):
                    parent = parent_candidate
                    break
            if parent is not None:
                parent.remove(lineseg)
            else:
                # Try removing from p_elem directly
                try:
                    p_elem.remove(lineseg)
                except:
                    pass

    # Also do a string-level pass for any placeholders outside <hp:p> tags
    import io
    out = io.BytesIO()
    tree.write(out, xml_declaration=True, encoding='utf-8')
    result = out.getvalue()

    # Final string-level replacement for any stragglers
    text_result = result.decode('utf-8')
    for placeholder, value in replacements.items():
        text_result = text_result.replace(placeholder, value)

    has_remaining = '{{' in text_result and '}}' in text_result
    return text_result.encode('utf-8'), has_remaining

# ── Repackage HWPX ──
template_path = 'clinic_intake_template.hwpx'
output_path = '/root/clinic_intake_ready.hwpx'

with zipfile.ZipFile(template_path, 'r') as zin:
    # Get ordered list; mimetype must be first
    names = zin.namelist()
    if 'mimetype' in names:
        names.remove('mimetype')
        names.insert(0, 'mimetype')

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
        for name in names:
            item_data = zin.read(name)
            compress = zipfile.ZIP_DEFLATED

            if name == 'mimetype':
                compress = zipfile.ZIP_STORED

            # Process XML files that might have placeholders
            if name.endswith('.xml') or name.endswith('.rels'):
                try:
                    text = item_data.decode('utf-8')
                    if '{{' in text:
                        item_data, has_remaining = process_xml(item_data)
                        if has_remaining:
                            print(f'WARNING: remaining placeholders in {name}')
                except Exception as e:
                    print(f'Error processing {name}: {e}')

            zout.writestr(
                zipfile.ZipInfo(name),
                item_data,
                compress_type=compress
            )

print('Output written to', output_path)

# ── Validate ──
with zipfile.ZipFile(output_path, 'r') as z:
    for name in z.namelist():
        try:
            content = z.read(name).decode('utf-8', 'replace')
            matches = re.findall(r'\{\{[^}]+\}\}', content)
            if matches:
                print(f'REMAINING PLACEHOLDERS in {name}: {matches}')
        except:
            pass
    print('Mimetype first?', z.namelist()[0] == 'mimetype' if 'mimetype' in z.namelist() else 'no mimetype')
    print('Mimetype info:', z.getinfo('mimetype').compress_type if 'mimetype' in z.namelist() else 'N/A')
print('Done.')
```

Run it:
```bash
cd /root && python3 generate.py
```

### 3. Handle edge cases
If the script reports remaining placeholders, inspect the specific XML to understand the placeholder key names (they may differ from the JSON keys). Adjust the replacement map accordingly and re-run.

Also check that:
- The `date_of_birth` line has the age note `(<N>세)` appended after the birth date.
- The phone number is in `000-0000-0000` format.
- Korean labels and the handwritten-signature note are preserved.
- No `{{...}}` text remains.

### 4. Run the verifier
```bash
cd /root && python3 -m pytest test_output.py -v
```

If any tests fail, read the error messages carefully, inspect the relevant XML sections in the output HWPX, fix the generation script, and re-run until all tests pass.

### Important notes
- When inspecting the template, pay close attention to the exact placeholder names used (e.g., `{{patient_name}}` vs `{{name}}`, `{{callback_phone}}` vs `{{phone}}`). Map them precisely from the JSON keys.
- The age note must appear after the birth date value, not as a separate placeholder replacement. If there IS a `{{age_note}}` placeholder, replace it; if not, append the age note after the birth date string.
- The phone normalization applies to whichever field represents the callback phone number.
- After writing the output, verify the ZIP is valid and the mimetype entry is first and uncompressed.

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