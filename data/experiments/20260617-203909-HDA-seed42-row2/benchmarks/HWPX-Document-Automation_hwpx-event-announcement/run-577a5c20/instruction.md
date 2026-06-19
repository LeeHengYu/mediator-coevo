# Task Instruction

Complete the following task to prepare an HWPX event announcement document.

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, and save the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step Plan

### 1. Inspect the workspace
```bash
ls /root/
```
Identify `event_announcement_template.hwpx` and `event_data.json`.

### 2. Read the JSON data
```bash
cat /root/event_data.json
```
Note every key-value pair. These keys correspond to `{{key}}` placeholders in the template.

### 3. Explore the HWPX structure
HWPX files are ZIP archives. List the contents:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/event_announcement_template.hwpx'); print('\n'.join(z.namelist()))"
```
Identify all XML section files (e.g., `Contents/section0.xml`) and any other content files that might contain placeholders.

### 4. Extract and inspect section XML files
Extract the HWPX to a temp directory and search for all `{{` occurrences:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/event_announcement_template.hwpx')
z.extractall('/tmp/hwpx_work/extracted')
"
grep -r '{{' /tmp/hwpx_work/extracted/ || echo 'No raw placeholders found in extracted text'
```

Also dump the raw XML of each section file to inspect whether placeholders are split across multiple XML text nodes (e.g., `{{` in one `<hp:t>` and `key}}` in the next).

### 5. Write the Python automation script
Create `/tmp/hwpx_work/process.py` with the following logic:

```python
import zipfile
import json
import os
import re
import shutil
from lxml import etree

# Paths
TEMPLATE = '/root/event_announcement_template.hwpx'
OUTPUT = '/root/event_announcement_ready.hwpx'
DATA_FILE = '/root/event_data.json'
EXTRACT_DIR = '/tmp/hwpx_work/extracted'
OUTPUT_DIR = '/tmp/hwpx_work/output'

# Load JSON data
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract template
if os.path.exists(EXTRACT_DIR):
    shutil.rmtree(EXTRACT_DIR)
os.makedirs(EXTRACT_DIR)
with zipfile.ZipFile(TEMPLATE, 'r') as zin:
    zin.extractall(EXTRACT_DIR)
    member_list = zin.namelist()

# Process each XML file that could contain placeholders
for root_dir, dirs, files in os.walk(EXTRACT_DIR):
    for fname in files:
        if not fname.endswith('.xml'):
            continue
        fpath = os.path.join(root_dir, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            content = f.read()
        if '{{' not in content and '}}' not in content:
            continue

        # Parse XML
        tree = etree.fromstring(content.encode('utf-8'))
        nsmap = tree.nsmap
        # Build namespace prefix map for xpath
        ns = {}
        for prefix, uri in nsmap.items():
            if prefix is not None:
                ns[prefix] = uri
        # Also handle common HWPX namespaces
        if 'hp' not in ns:
            # Try to find it
            for prefix, uri in nsmap.items():
                if uri and 'hancom' in uri.lower() and 'hwp' in uri.lower():
                    ns['hp'] = uri
                    break

        # Strategy: For each paragraph (<hp:p>), collect all <hp:t> text,
        # merge them, do replacements, then redistribute.
        # This handles placeholders split across multiple <hp:t> elements.
        
        # Find all paragraph elements
        # Try multiple namespace patterns
        paragraphs = tree.iter()
        p_tags = []
        for elem in tree.iter():
            tag = etree.QName(elem.tag).localname if '}' in elem.tag else elem.tag
            if tag == 'p':
                p_tags.append(elem)

        for p_elem in p_tags:
            # Collect all <hp:t> (text run) elements in this paragraph
            t_elems = []
            for child in p_elem.iter():
                local = etree.QName(child.tag).localname if '}' in child.tag else child.tag
                if local == 't' and child.text is not None:
                    t_elems.append(child)
            
            if not t_elems:
                continue
            
            # Merge all text
            merged = ''.join((t.text or '') for t in t_elems)
            
            if '{{' not in merged:
                continue
            
            # Replace all placeholders
            def replacer(m):
                key = m.group(1).strip()
                return str(data.get(key, m.group(0)))
            
            replaced = re.sub(r'\{\{\s*(.+?)\s*\}\}', replacer, merged)
            
            # Put all text in the first <hp:t>, clear the rest
            t_elems[0].text = replaced
            for t in t_elems[1:]:
                t.text = ''
            
            # Remove stale layout cache elements (lineSegArray, linesegarray)
            # These cause overlapping character rendering if not cleared
            to_remove = []
            for child in p_elem.iter():
                local = etree.QName(child.tag).localname if '}' in child.tag else child.tag
                if local.lower() in ('lineSegArray', 'linesegarray', 'linesegarr'):
                    to_remove.append(child)
            for elem_to_rm in to_remove:
                parent = elem_to_rm.getparent()
                if parent is not None:
                    parent.remove(elem_to_rm)

        # Also do a broader search for lineSegArray at any level and remove
        for elem in list(tree.iter()):
            local = etree.QName(elem.tag).localname if '}' in elem.tag else elem.tag
            if 'lineseg' in local.lower():
                parent = elem.getparent()
                if parent is not None:
                    parent.remove(elem)

        # Serialize back
        result = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=False)
        with open(fpath, 'wb') as f:
            f.write(result)

# Rebuild the HWPX ZIP with mimetype as first uncompressed entry
if os.path.exists(OUTPUT):
    os.remove(OUTPUT)

with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
    # Write mimetype first, uncompressed
    mimetype_path = os.path.join(EXTRACT_DIR, 'mimetype')
    if os.path.exists(mimetype_path):
        zout.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    # Write all other files in original order
    for member in member_list:
        if member == 'mimetype':
            continue
        full_path = os.path.join(EXTRACT_DIR, member)
        if os.path.isfile(full_path):
            zout.write(full_path, member, compress_type=zipfile.ZIP_DEFLATED)
        elif os.path.isdir(full_path):
            # Directory entries - skip or add as needed
            pass

print('HWPX written to', OUTPUT)
```

Run this script:
```bash
python3 /tmp/hwpx_work/process.py
```

### 6. Validate the output

#### 6a. Check no placeholders remain
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/event_announcement_ready.hwpx')
for name in z.namelist():
    if name.endswith('.xml'):
        content = z.read(name).decode('utf-8', errors='replace')
        if '{{' in content or '}}' in content:
            print(f'FAIL: Placeholder found in {name}')
            # Show context
            import re
            for m in re.finditer(r'.{0,40}\{\{.+?\}\}.{0,40}', content):
                print(f'  {m.group()}')
print('Placeholder check complete')
"
```

#### 6b. Verify it's a valid ZIP
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/event_announcement_ready.hwpx')
print('Valid ZIP. Members:', len(z.namelist()))
print('First entry:', z.namelist()[0])
print('mimetype content:', z.read('mimetype') if 'mimetype' in z.namelist() else 'NO MIMETYPE')
"
```

#### 6c. Verify no lineSegArray elements remain in modified sections
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/event_announcement_ready.hwpx')
for name in z.namelist():
    if name.endswith('.xml'):
        content = z.read(name).decode('utf-8', errors='replace')
        if 'lineseg' in content.lower() or 'lineSeg' in content:
            print(f'WARNING: lineSegArray found in {name}')
        else:
            print(f'OK: {name}')
"
```

#### 6d. Spot-check that JSON values appear in the output
```bash
python3 -c "
import zipfile, json
z = zipfile.ZipFile('/root/event_announcement_ready.hwpx')
with open('/root/event_data.json') as f:
    data = json.load(f)
all_text = ''
for name in z.namelist():
    if name.endswith('.xml'):
        all_text += z.read(name).decode('utf-8', errors='replace')
for key, val in data.items():
    val_str = str(val)
    if val_str in all_text:
        print(f'OK: {key} = {val_str}')
    else:
        print(f'MISSING: {key} = {val_str}')
"
```

### 7. Run the verifier if available
```bash
cd /root && ls test_output.py 2>/dev/null && python3 -m pytest test_output.py -v
```

### Critical Notes
- **Split placeholders**: Placeholders like `{{event_name}}` may be split across multiple `<hp:t>` XML elements (e.g., `{{ev` and `ent_name}}`). The script handles this by merging all text in a paragraph before replacement.
- **Layout cache**: Remove all `lineSegArray` elements from modified paragraphs to prevent overlapping character rendering.
- **ZIP mimetype**: The `mimetype` file MUST be the first entry in the ZIP and stored uncompressed (`ZIP_STORED`). This is required for valid HWPX/ODF-style packages.
- **Korean text preservation**: Only replace `{{...}}` patterns. Do not modify any Korean labels or static text.

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