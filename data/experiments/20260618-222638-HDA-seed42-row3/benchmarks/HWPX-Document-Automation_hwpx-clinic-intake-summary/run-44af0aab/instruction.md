# Task Instruction

Produce the file `/root/clinic_intake_ready.hwpx` by completing the clinic-intake template with patient data.

## Step-by-step plan

### 1. Inspect the inputs
```bash
cat /root/patient_intake.json
```
Understand every field and its value.

```bash
cd /root && python3 -c "
import zipfile, sys
zf = zipfile.ZipFile('clinic_intake_template.hwpx')
for n in zf.namelist():
    print(n)
"
```
List every entry in the HWPX zip.

### 2. Dump the raw XML of the section file
```bash
python3 -c "
import zipfile
zf = zipfile.ZipFile('clinic_intake_template.hwpx')
for name in zf.namelist():
    if 'section' in name.lower() and name.endswith('.xml'):
        print('===', name, '===')
        print(zf.read(name).decode('utf-8'))
"
```
Read it carefully. Identify:
- Every `{{placeholder}}` and where they appear.
- Whether placeholders are split across multiple `<hp:t>` or `<hp:run>` elements.
- The namespace declarations.
- Any `<hp:linesegarray>` or `<hp:lineSegArray>` elements (layout cache).

### 3. Write a Python script that does the replacement robustly

Create `/root/build.py` with this logic:

```python
import zipfile, json, re, copy, os
from lxml import etree

# --- Load patient data ---
with open('patient_intake.json') as f:
    data = json.load(f)

# --- Compute derived values ---
# Korean full-year age: age as of visit date
from datetime import date
birth = date.fromisoformat(data['birth_date'])          # e.g. '1990-11-02'
visit = date.fromisoformat(data['visit_date'])          # e.g. '2025-11-15'
age = visit.year - birth.year - ((visit.month, visit.day) < (birth.month, birth.day))

# Normalize phone: strip everything non-digit, then format 000-0000-0000
digits = re.sub(r'\D', '', data['callback_phone'])
phone_formatted = f"{digits[:3]}-{digits[3:7]}-{digits[7:11]}"

# Build replacement map  (placeholder text -> replacement text)
# Inspect the JSON keys and the XML placeholders to build this map.
# Common pattern: {{key_name}}
# Also handle the age note: birth_date placeholder becomes "<birth_date> (<age>세)"
replacements = {}
for key, val in data.items():
    placeholder = '{{' + key + '}}'
    if key == 'birth_date':
        replacements[placeholder] = f"{val} ({age}세)"
    elif key == 'callback_phone':
        replacements[placeholder] = phone_formatted
    else:
        replacements[placeholder] = str(val)

print('Replacement map:')
for k, v in replacements.items():
    print(f'  {k} -> {v}')

# --- Process the HWPX zip ---
INPUT = 'clinic_intake_template.hwpx'
OUTPUT = 'clinic_intake_ready.hwpx'

zin = zipfile.ZipFile(INPUT, 'r')
zout = zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED)

for item in zin.namelist():
    raw = zin.read(item)
    if item.endswith('.xml'):
        text = raw.decode('utf-8')
        # Check if this XML has any placeholders
        if '{{' in text:
            # Parse as XML
            root = etree.fromstring(raw)
            nsmap = root.nsmap
            # Detect the hp namespace URI
            hp_ns = None
            for prefix, uri in nsmap.items():
                if prefix == 'hp' or 'hancom' in uri.lower() or 'hwp' in uri.lower():
                    hp_ns = uri
                    break
            if hp_ns is None:
                # fallback: try to find from tag
                for prefix, uri in nsmap.items():
                    if 'hcfp' in uri or 'para' in uri or 'run' in uri:
                        hp_ns = uri
                        break

            # Strategy: for each <hp:run> (or equivalent), merge all <hp:t>
            # children text, do replacements, then put the result in a single <hp:t>.
            # This handles split placeholders.

            # Find all run-like elements that contain <hp:t> children
            # We need to iterate all elements and find those whose children include 't' elements
            # Use the namespace from the 't' elements themselves
            
            # First, collect all 't' elements to learn the namespace
            all_t = root.iter()
            t_ns = None
            for el in root.iter():
                tag = el.tag
                if '}' in tag:
                    local = tag.split('}')[1]
                    ns = tag.split('}')[0][1:]
                    if local == 't':
                        t_ns = ns
                        break
            
            if t_ns is None:
                # No <hp:t> elements; do plain text replacement
                text_out = text
                for ph, repl in replacements.items():
                    text_out = text_out.replace(ph, repl)
                raw = text_out.encode('utf-8')
            else:
                t_tag = '{' + t_ns + '}t'
                run_tag = '{' + t_ns + '}run'
                lineseg_tag1 = '{' + t_ns + '}linesegarray'
                lineseg_tag2 = '{' + t_ns + '}lineSegArray'
                
                # Merge split placeholders within each <run> element
                for run_el in list(root.iter(run_tag)):
                    t_children = [ch for ch in run_el if ch.tag == t_tag]
                    if len(t_children) > 1:
                        # Merge all text into the first <hp:t>, remove the rest
                        merged = ''.join((t.text or '') for t in t_children)
                        t_children[0].text = merged
                        for extra_t in t_children[1:]:
                            run_el.remove(extra_t)
                
                # Also check for placeholders split across DIFFERENT <run> elements
                # within the same parent (paragraph). Merge adjacent runs if their
                # combined text forms a placeholder.
                # Approach: for each parent that has <run> children, concatenate
                # all <hp:t> text, check for placeholders, and if found, do a
                # smarter merge.
                
                # Collect all elements that have run children
                parents_of_runs = set()
                for run_el in root.iter(run_tag):
                    parent = run_el.getparent()
                    if parent is not None:
                        parents_of_runs.add(id(parent))
                
                # For cross-run merging, gather text per parent
                for run_el in root.iter(run_tag):
                    parent = run_el.getparent()
                    if parent is None:
                        continue
                    runs_in_parent = [ch for ch in parent if ch.tag == run_tag]
                    # Build combined text
                    combined = ''
                    for r in runs_in_parent:
                        for t in r:
                            if t.tag == t_tag:
                                combined += (t.text or '')
                    if '{{' in combined and '}}' in combined:
                        # Check if any single run already has a complete placeholder
                        # If not, we need to merge runs
                        needs_merge = False
                        for r in runs_in_parent:
                            run_text = ''.join((t.text or '') for t in r if t.tag == t_tag)
                            if '{{' in run_text and '}}' not in run_text:
                                needs_merge = True
                                break
                            if '}}' in run_text and '{{' not in run_text:
                                needs_merge = True
                                break
                        if needs_merge:
                            # Merge all runs' text into the first run's first <hp:t>
                            first_run = runs_in_parent[0]
                            first_t = None
                            for ch in first_run:
                                if ch.tag == t_tag:
                                    first_t = ch
                                    break
                            if first_t is None:
                                first_t = etree.SubElement(first_run, t_tag)
                            first_t.text = combined
                            # Remove extra <hp:t> from first run
                            for ch in list(first_run):
                                if ch.tag == t_tag and ch is not first_t:
                                    first_run.remove(ch)
                            # Remove other runs
                            for r in runs_in_parent[1:]:
                                parent.remove(r)
                    # Only process each parent once
                    parents_of_runs.discard(id(parent))
                
                # Now do text replacements on all <hp:t> elements
                for t_el in root.iter(t_tag):
                    if t_el.text and '{{' in t_el.text:
                        for ph, repl in replacements.items():
                            t_el.text = t_el.text.replace(ph, repl)
                
                # Remove layout cache elements (linesegarray / lineSegArray)
                for tag_to_remove in [lineseg_tag1, lineseg_tag2]:
                    for el in list(root.iter(tag_to_remove)):
                        el.getparent().remove(el)
                
                # Also try case-insensitive removal by local name
                for el in list(root.iter()):
                    local = el.tag.split('}')[1] if '}' in el.tag else el.tag
                    if local.lower() == 'linesegarray':
                        if el.getparent() is not None:
                            el.getparent().remove(el)
                
                raw = etree.tostring(root, xml_declaration=True, encoding='UTF-8')
    
    zout.writestr(item, raw)

zin.close()
zout.close()
print('Done. Output:', OUTPUT)
```

### 4. Run the build script
```bash
cd /root && python3 build.py
```

### 5. Validate the output

#### 5a. Check it's a valid zip
```bash
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for n in zf.namelist():
    print(n)
print('Valid zip: OK')
"
```

#### 5b. Check no placeholders remain
```bash
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in zf.namelist():
    data = zf.read(name)
    try:
        text = data.decode('utf-8')
    except:
        continue
    if '{{' in text:
        print(f'FAIL: placeholder found in {name}')
        # Show context
        idx = text.find('{{')
        print(text[max(0,idx-50):idx+80])
    else:
        if name.endswith('.xml'):
            print(f'{name}: clean')
print('Placeholder check done.')
"
```

#### 5c. Check the specific birth date + age string
```bash
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in zf.namelist():
    if 'section' in name.lower():
        text = zf.read(name).decode('utf-8')
        # Check for the expected birth date string
        # (Compute expected age dynamically)
        from datetime import date
        import json
        with open('patient_intake.json') as f:
            d = json.load(f)
        birth = date.fromisoformat(d['birth_date'])
        visit = date.fromisoformat(d['visit_date'])
        age = visit.year - birth.year - ((visit.month, visit.day) < (birth.month, birth.day))
        expected = f\"{d['birth_date']} ({age}세)\"
        if expected in text:
            print(f'Birth date + age OK: {expected}')
        else:
            print(f'FAIL: expected \"{expected}\" not found in {name}')
            # Show all hp:t content for debugging
            import re
            for m in re.finditer(r'>([^<]*(?:birth|1990|세)[^<]*)<', text):
                print(f'  found: {m.group(1)}')
"
```

#### 5d. Check phone normalization
```bash
python3 -c "
import zipfile, json, re
with open('patient_intake.json') as f:
    d = json.load(f)
digits = re.sub(r'\\D', '', d['callback_phone'])
expected_phone = f'{digits[:3]}-{digits[3:7]}-{digits[7:11]}'
zf = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in zf.namelist():
    if 'section' in name.lower():
        text = zf.read(name).decode('utf-8')
        if expected_phone in text:
            print(f'Phone OK: {expected_phone}')
        else:
            print(f'FAIL: expected phone {expected_phone} not in {name}')
"
```

#### 5e. Check no linesegarray elements remain in modified files
```bash
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in zf.namelist():
    if 'section' in name.lower():
        text = zf.read(name).decode('utf-8').lower()
        if 'linesegarray' in text:
            print(f'FAIL: linesegarray still in {name}')
        else:
            print(f'{name}: no linesegarray - OK')
"
```

#### 5f. Check Korean labels and signature note preserved
```bash
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/clinic_intake_ready.hwpx')
for name in zf.namelist():
    if 'section' in name.lower():
        text = zf.read(name).decode('utf-8')
        if '수기 서명' in text:
            print('Signature note preserved: OK')
        else:
            print('WARN: signature note may be missing')
"
```

### 6. Run the verifier test if available
```bash
cd /root && ls test_output.py 2>/dev/null && python3 -m pytest test_output.py -v
```

### 7. Debug and fix
If any check fails:
- Re-read the raw XML from the template to understand the actual tag structure.
- Adjust the `build.py` script accordingly.
- Re-run and re-validate.

**Critical attention points from prior failure:**
- Placeholders may be split across multiple `<hp:t>` elements within a single `<hp:run>`, OR across multiple `<hp:run>` elements. The script must handle BOTH cases.
- The age calculation must use Korean full-year age (Western age): `visit_year - birth_year - 1` if birthday hasn't occurred yet in the visit year, otherwise `visit_year - birth_year`.
- The birth date line must read exactly `1990-11-02 (35세)` (or whatever the correct computed values are) with a space before the parenthesis.
- After replacement, no `{{...}}` text may remain anywhere in any file of the zip.
- All `linesegarray`/`lineSegArray` elements must be removed from paragraphs whose text was modified (the script removes them all, which is safe).
- The output must be a valid HWPX (zip) package with the same entry structure as the input.

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