# Task Instruction

Complete the following task step by step:

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json` and save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read and display `event_data.json` to understand all available keys and values.
- List the contents of the HWPX file (it's a ZIP): `python3 -c "import zipfile; z=zipfile.ZipFile('event_announcement_template.hwpx'); print('\n'.join(z.namelist()))"`
- Extract and display all XML files from the HWPX, especially files under `Contents/` (like `section0.xml`, `content.hpf`, etc.) to find where `{{...}}` placeholders appear.

### 2. Write and run a Python script
Create a Python script (`/root/process_hwpx.py`) that does the following:

```python
import zipfile
import json
import re
import os
import shutil
from lxml import etree

# Paths
TEMPLATE = 'event_announcement_template.hwpx'
DATA_FILE = 'event_data.json'
OUTPUT = '/root/event_announcement_ready.hwpx'
TEMP_DIR = '/tmp/hwpx_work'

# Load JSON data
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract HWPX
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)
os.makedirs(TEMP_DIR)

with zipfile.ZipFile(TEMPLATE, 'r') as zin:
    entry_list = zin.namelist()
    zin.extractall(TEMP_DIR)

# Process each XML file
for entry in entry_list:
    filepath = os.path.join(TEMP_DIR, entry)
    if not os.path.isfile(filepath):
        continue
    if not (entry.endswith('.xml') or entry.endswith('.hpf')):
        continue
    
    with open(filepath, 'rb') as f:
        raw = f.read()
    
    # Skip files without any placeholder fragments
    if b'{{' not in raw and b'}}' not in raw:
        continue
    
    tree = etree.fromstring(raw)
    nsmap = tree.nsmap
    
    # Find all hp:p or equivalent paragraph elements
    # We need to handle namespaces carefully
    # Get the hp namespace
    hp_ns = None
    for prefix, uri in nsmap.items():
        if prefix == 'hp' or 'hwpml' in uri.lower() or 'hancom' in uri.lower():
            hp_ns = uri
            break
    if not hp_ns:
        # Try to find from element tags
        for elem in tree.iter():
            tag = elem.tag
            if '}p' in tag:
                hp_ns = tag.split('}')[0].lstrip('{')
                break
    
    if not hp_ns:
        # Just do string-level replacement as fallback
        text = raw.decode('utf-8')
        for key, value in data.items():
            text = text.replace('{{' + key + '}}', str(value))
        if '{{' in text:
            print(f'WARNING: Remaining placeholders in {entry}')
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        continue
    
    ns = {'hp': hp_ns}
    
    # Find all paragraph elements
    modified_paras = set()
    
    for p_elem in tree.iter('{%s}p' % hp_ns):
        # Collect all <hp:t> text elements within this paragraph's runs
        t_elements = list(p_elem.iter('{%s}t' % hp_ns))
        if not t_elements:
            continue
        
        # Join all text content to check for placeholders
        full_text = ''.join((t.text or '') for t in t_elements)
        
        if '{{' not in full_text:
            continue
        
        # Perform replacement on the joined text
        replaced_text = full_text
        for key, value in data.items():
            replaced_text = replaced_text.replace('{{' + key + '}}', str(value))
        
        if replaced_text == full_text:
            continue
        
        # Redistribute text: put all text in first <hp:t>, clear the rest
        t_elements[0].text = replaced_text
        for t in t_elements[1:]:
            t.text = ''
        
        modified_paras.add(p_elem)
    
    # Remove layout cache elements (lineSegArray, linesegarray) from modified paragraphs
    for p_elem in modified_paras:
        for child in list(p_elem):
            local_tag = etree.QName(child.tag).localname.lower()
            if 'lineseg' in local_tag:
                p_elem.remove(child)
    
    # Serialize back
    output_bytes = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
    # Preserve original XML declaration style if needed
    with open(filepath, 'wb') as f:
        f.write(output_bytes)

# Rebuild ZIP with mimetype first and stored
with zipfile.ZipFile(OUTPUT, 'w') as zout:
    # Write mimetype first, uncompressed
    if 'mimetype' in entry_list:
        mime_path = os.path.join(TEMP_DIR, 'mimetype')
        zout.write(mime_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    # Write all other entries
    for entry in entry_list:
        if entry == 'mimetype':
            continue
        filepath = os.path.join(TEMP_DIR, entry)
        if os.path.isfile(filepath):
            zout.write(filepath, entry, compress_type=zipfile.ZIP_DEFLATED)
        elif os.path.isdir(filepath):
            # Directory entries
            zout.writestr(entry, '')

print('Output written to', OUTPUT)

# Validation: check no placeholders remain
with zipfile.ZipFile(OUTPUT, 'r') as z:
    for name in z.namelist():
        content = z.read(name)
        try:
            text = content.decode('utf-8')
            matches = re.findall(r'\{\{.*?\}\}', text)
            if matches:
                print(f'ERROR: Remaining placeholders in {name}: {matches}')
        except:
            pass

print('Validation complete.')
```

### 3. Run the script
```bash
cd /root  # or wherever the template files are located
python3 /root/process_hwpx.py
```

### 4. Validate the output
- Verify no `{{...}}` placeholders remain: `python3 -c "import zipfile,re; z=zipfile.ZipFile('/root/event_announcement_ready.hwpx'); [print(f'PLACEHOLDER in {n}: {re.findall(r\"{{.*?}}\", z.read(n).decode(\"utf-8\"))}') for n in z.namelist() if z.read(n).startswith(b'<') and re.findall(r'\{\{.*?\}\}', z.read(n).decode('utf-8'))]"`
- Verify the HWPX is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/event_announcement_ready.hwpx'); print('Valid ZIP, entries:', len(z.namelist()))"`
- Spot-check that Korean labels are preserved by printing a section of the XML content.
- Verify that `<hp:lineSegArray>` or similar layout cache elements have been removed from modified paragraphs.

### Important Notes
- **Fragmented text nodes**: Placeholders may be split across multiple `<hp:t>` XML elements (e.g., `<hp:t>{{</hp:t><hp:t>event_name</hp:t><hp:t>}}</hp:t>`). You MUST join text from all `<hp:t>` elements in a paragraph before doing replacement, then redistribute.
- **Layout cache**: Remove `<hp:lineSegArray>` (and any case variant like `<hp:linesegarray>`) from every paragraph whose text was modified. This prevents overlapping character rendering.
- **mimetype entry**: Must be the FIRST entry in the ZIP and must use ZIP_STORED (no compression).
- **Preserve structure**: Do not add or remove files from the HWPX package. Only modify text content in XML files.
- If the template or data files are not in `/root/`, first find them with `find / -name 'event_announcement_template.hwpx' 2>/dev/null` and `find / -name 'event_data.json' 2>/dev/null`, then adjust paths accordingly.
- If `lxml` is not available, install it with `pip install lxml` or use `xml.etree.ElementTree` instead (but be careful with namespace handling).

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