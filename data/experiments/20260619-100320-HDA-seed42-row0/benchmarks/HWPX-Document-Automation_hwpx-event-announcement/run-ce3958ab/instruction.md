# Task Instruction

Execute the following steps to produce `/root/event_announcement_ready.hwpx`.

## 1 – Inspect inputs
```bash
cd /root
ls -la event_announcement_template.hwpx event_data.json
cat event_data.json
```
Understand every key in the JSON; these are the replacement values for `{{key}}` placeholders.

## 2 – Extract the HWPX archive
```bash
mkdir -p /root/hwpx_work
cd /root/hwpx_work
unzip -o /root/event_announcement_template.hwpx -d extracted
find extracted -type f | sort
```
Note the directory layout. Identify the `mimetype` file and all XML files (especially under `Contents/`).

## 3 – Examine XML content files for placeholders
```bash
grep -rn '{{' extracted/
```
Also look at one or two XML files to understand namespace prefixes and tag structure:
```bash
cat extracted/Contents/section0.xml | head -200
```

## 4 – Write and run the Python replacement script

Create `/root/hwpx_work/process.py` with the following logic:

```python
import json, os, re, shutil, zipfile
import xml.etree.ElementTree as ET

# ---- paths ----
DATA = '/root/event_data.json'
EXTRACTED = '/root/hwpx_work/extracted'
OUTPUT = '/root/event_announcement_ready.hwpx'

with open(DATA, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ---- namespace handling ----
# Parse one XML to discover namespaces, then register them all so
# ET.write() preserves prefixes.
NS = {}
for evt, elem in ET.iterparse(os.path.join(EXTRACTED, 'Contents', 'section0.xml'), events=['start-ns']):
    prefix, uri = elem
    if prefix not in NS:
        NS[prefix] = uri
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

HP_NS = NS.get('hp', NS.get('', ''))
hp = '{' + HP_NS + '}' if HP_NS else ''

# ---- process every XML file under Contents/ ----
xml_files = []
for root_dir, dirs, files in os.walk(os.path.join(EXTRACTED, 'Contents')):
    for fn in files:
        if fn.endswith('.xml'):
            xml_files.append(os.path.join(root_dir, fn))

for xml_path in xml_files:
    # Re-register namespaces for each file (they may differ)
    file_ns = {}
    for evt, elem in ET.iterparse(xml_path, events=['start-ns']):
        prefix, uri = elem
        if prefix not in file_ns:
            file_ns[prefix] = uri
    for p, u in file_ns.items():
        ET.register_namespace(p, u)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    hp_uri = file_ns.get('hp', HP_NS)
    hp_tag = '{' + hp_uri + '}' if hp_uri else ''

    t_tag = hp_tag + 't'
    p_tag = hp_tag + 'p'
    run_tag = hp_tag + 'run'
    lsa_tag = hp_tag + 'lineSegArray'

    modified = False

    # Find all <hp:p> paragraphs
    for para in root.iter(p_tag):
        # Collect all <hp:t> elements inside this paragraph (across all runs)
        t_elems = list(para.iter(t_tag))
        if not t_elems:
            continue

        # Concatenate all text to check for placeholders
        full_text = ''.join((t.text or '') for t in t_elems)
        if '{{' not in full_text:
            continue

        # ---- Consolidate text within each <hp:run> ----
        for run_elem in para.iter(run_tag):
            run_t_elems = list(run_elem.iter(t_tag))
            if len(run_t_elems) <= 1:
                continue
            # Merge all text into the first <hp:t>, clear the rest
            merged = ''.join((t.text or '') for t in run_t_elems)
            run_t_elems[0].text = merged
            for t in run_t_elems[1:]:
                t.text = ''

        # Now do replacement across the whole paragraph's <hp:t> elements
        # Re-collect after consolidation
        t_elems = list(para.iter(t_tag))
        full_text = ''.join((t.text or '') for t in t_elems)

        if '{{' not in full_text:
            continue

        # If a placeholder spans across multiple <hp:t> tags (across runs),
        # merge everything into the first tag
        if re.search(r'\{\{[^}]*$', (t_elems[0].text or '')) or any(
            '}}' in (t.text or '') and '{{' not in (t.text or '') for t in t_elems[1:]
        ):
            t_elems[0].text = full_text
            for t in t_elems[1:]:
                t.text = ''
            full_text = t_elems[0].text

        # Perform placeholder replacement
        new_text = full_text
        for key, value in data.items():
            new_text = new_text.replace('{{' + key + '}}', str(value))

        # Also handle any remaining {{...}} with a fallback (should not happen)
        # but catch nested/unknown keys
        remaining = re.findall(r'\{\{(.+?)\}\}', new_text)
        if remaining:
            print(f'WARNING: unresolved placeholders: {remaining}')

        if new_text != full_text:
            t_elems[0].text = new_text
            for t in t_elems[1:]:
                t.text = ''
            modified = True

            # Remove <hp:lineSegArray> from this paragraph
            for lsa in list(para.iter(lsa_tag)):
                # Find the parent of lsa and remove it
                for parent in para.iter():
                    if lsa in list(parent):
                        parent.remove(lsa)
                        break

    # Also do a brute-force removal: remove ALL lineSegArray from paragraphs
    # that contain modified text (belt-and-suspenders)
    # Actually, per the safety-audit-brief feedback, remove lineSegArray from
    # ALL modified paragraphs reliably using a parent map.
    if modified:
        parent_map = {c: p for p in root.iter() for c in p}
        for lsa in list(root.iter(lsa_tag)):
            parent = parent_map.get(lsa)
            if parent is not None:
                # Check if this lsa is inside a paragraph that was modified
                # For safety, remove from any paragraph that has replacement text
                parent.remove(lsa)

        tree.write(xml_path, encoding='utf-8', xml_declaration=True)

# ---- Re-package as HWPX ----
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

with zipfile.ZipFile(OUTPUT, 'w') as zf:
    # mimetype must be first and stored uncompressed
    mimetype_path = os.path.join(EXTRACTED, 'mimetype')
    if os.path.exists(mimetype_path):
        zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)

    for dirpath, dirnames, filenames in os.walk(EXTRACTED):
        for fn in sorted(filenames):
            full = os.path.join(dirpath, fn)
            arcname = os.path.relpath(full, EXTRACTED)
            if arcname == 'mimetype':
                continue
            zf.write(full, arcname, compress_type=zipfile.ZIP_DEFLATED)

print('Output written to', OUTPUT)
```

Run it:
```bash
python3 /root/hwpx_work/process.py
```

## 5 – Validate the output

### 5a – Verify it is a valid ZIP
```bash
unzip -t /root/event_announcement_ready.hwpx
```

### 5b – Check no placeholders remain
```bash
mkdir -p /root/hwpx_verify
unzip -o /root/event_announcement_ready.hwpx -d /root/hwpx_verify
grep -rn '{{' /root/hwpx_verify/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'
```

### 5c – Verify lineSegArray removed from modified paragraphs
```bash
# Check that no lineSegArray appears in paragraphs containing replacement values
python3 -c "
import xml.etree.ElementTree as ET, os, sys
for root_dir, dirs, files in os.walk('/root/hwpx_verify/Contents'):
    for fn in files:
        if not fn.endswith('.xml'): continue
        path = os.path.join(root_dir, fn)
        ns = {}
        for evt, elem in ET.iterparse(path, events=['start-ns']):
            ns[elem[0]] = elem[1]
        tree = ET.parse(path)
        hp = '{' + ns.get('hp','') + '}'
        for p in tree.getroot().iter(hp+'p'):
            texts = ''.join((t.text or '') for t in p.iter(hp+'t'))
            has_lsa = any(True for _ in p.iter(hp+'lineSegArray'))
            if has_lsa and texts.strip():
                print(f'WARNING: lineSegArray still present in paragraph with text: {texts[:60]}')
print('lineSegArray check complete')
"
```

### 5d – Verify Korean labels preserved
```bash
grep -c '행사' /root/hwpx_verify/Contents/section0.xml && echo 'Korean labels present' || echo 'WARNING: Korean labels may be missing'
```

### 5e – Verify JSON values appear in output
```bash
python3 -c "
import json
with open('/root/event_data.json') as f: data = json.load(f)
with open('/root/hwpx_verify/Contents/section0.xml') as f: content = f.read()
missing = [k for k,v in data.items() if str(v) not in content]
if missing: print('FAIL: missing values for keys:', missing)
else: print('PASS: all JSON values found in output')
"
```

If any validation step fails, diagnose the issue, fix the script, and re-run. Do not consider the task complete until all five checks pass and `/root/event_announcement_ready.hwpx` exists as a valid HWPX package with no remaining placeholders.

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