# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in `inventory_report_template.hwpx` with values from `inventory_data.json`, then save the result to `/root/inventory_report_ready.hwpx`.

## Step-by-step plan

### 1. Inspect the input files
- Read `inventory_data.json` and print its contents to understand the key-value mapping.
- List the contents of `inventory_report_template.hwpx` (it's a ZIP archive): `unzip -l inventory_report_template.hwpx`.
- Identify which XML files inside the HWPX contain `{{` placeholders: extract the archive to a temp directory and grep for `{{`.

### 2. Write and run a Python script that does the following:

```python
import json, os, re, shutil, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

# Paths
TEMPLATE = '/root/inventory_report_template.hwpx'  # adjust if different
DATA_FILE = '/root/inventory_data.json'              # adjust if different
OUTPUT = '/root/inventory_report_ready.hwpx'
TMP_DIR = '/tmp/hwpx_work'
TMP_OUT = '/tmp/hwpx_out'

# Clean up
for d in [TMP_DIR, TMP_OUT]:
    if os.path.exists(d):
        shutil.rmtree(d)

# Extract
with zipfile.ZipFile(TEMPLATE, 'r') as z:
    z.extractall(TMP_DIR)

# Load data
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# If data is nested, flatten it into a simple key->value dict.
# Handle both flat {"key": "value"} and nested structures.
def flatten(obj, prefix=''):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (dict, list)):
                items.update(flatten(v, k))
            else:
                items[k] = str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                items.update(flatten(v, prefix))
            else:
                items[f'{prefix}_{i}'] = str(v)
    return items

flat_data = flatten(data)
print('Flat data keys:', list(flat_data.keys()))

# Find XML files with placeholders
HP_NS = 'http://www.hancom.co.kr/hwpml/2011/paragraph'
for ns_prefix, ns_uri in [('hp', HP_NS)]:
    ET.register_namespace(ns_prefix, ns_uri)

# We'll process ALL xml files to be safe
xml_files = []
for root_dir, dirs, files in os.walk(TMP_DIR):
    for fname in files:
        if fname.endswith('.xml'):
            fpath = os.path.join(root_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            if '{{' in content:
                xml_files.append(fpath)
                print(f'Found placeholders in: {fpath}')

# Process each XML file
for xml_path in xml_files:
    with open(xml_path, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # Detect all namespace URIs used so we can register them to avoid ns0/ns1 prefixes
    for match in re.finditer(r'xmlns:(\w+)="([^"]+)"', xml_content):
        ET.register_namespace(match.group(1), match.group(2))
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Detect the hp namespace dynamically
    hp_ns = None
    for match in re.finditer(r'xmlns:(\w+)="([^"]+)"', xml_content):
        if 'paragraph' in match.group(2) or 'hwpml' in match.group(2):
            hp_ns = match.group(2)
            break
    if not hp_ns:
        # Fallback: try common namespaces
        hp_ns = HP_NS
    
    ns = {'hp': hp_ns}
    
    # Strategy: For each <hp:p> paragraph, consolidate all <hp:t> text,
    # perform replacements, put result in first <hp:t>, clear the rest.
    # Then remove <hp:lineSegArray> from modified paragraphs.
    
    modified_paragraphs = []
    
    for p_elem in root.iter(f'{{{hp_ns}}}p'):
        # Collect all <hp:t> elements in order
        t_elems = list(p_elem.iter(f'{{{hp_ns}}}t'))
        if not t_elems:
            continue
        
        # Concatenate all text
        full_text = ''.join((t.text or '') for t in t_elems)
        
        if '{{' not in full_text:
            continue
        
        # Perform replacements
        new_text = full_text
        for key, value in flat_data.items():
            placeholder = '{{' + key + '}}'
            new_text = new_text.replace(placeholder, value)
        
        # Also try with data dict directly (non-flattened) for nested keys
        # that might use dot notation or direct keys
        if isinstance(data, dict):
            for key, value in data.items():
                if not isinstance(value, (dict, list)):
                    placeholder = '{{' + key + '}}'
                    new_text = new_text.replace(placeholder, str(value))
        
        if new_text == full_text:
            continue  # No changes made
        
        # Put all text in first <hp:t>, clear the rest
        t_elems[0].text = new_text
        for t in t_elems[1:]:
            t.text = ''
        
        modified_paragraphs.append(p_elem)
    
    # Remove <hp:lineSegArray> from modified paragraphs using XML tree
    for p_elem in modified_paragraphs:
        for lsa in list(p_elem.iter(f'{{{hp_ns}}}lineSegArray')):
            # Find parent and remove
            for parent in p_elem.iter():
                if lsa in list(parent):
                    parent.remove(lsa)
                    break
    
    # Also do a second pass: remove lineSegArray using regex on the serialized XML
    # to catch any that the tree-based approach missed
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    
    with open(xml_path, 'r', encoding='utf-8') as f:
        final_xml = f.read()
    
    # Regex removal of any remaining lineSegArray in case tree missed some
    # (This is a belt-and-suspenders approach based on prior feedback)
    final_xml = re.sub(r'<[^>]*:lineSegArray[^>]*>.*?</[^>]*:lineSegArray>', '', final_xml, flags=re.DOTALL)
    final_xml = re.sub(r'<lineSegArray[^>]*>.*?</lineSegArray>', '', final_xml, flags=re.DOTALL)
    
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(final_xml)
    
    print(f'Processed: {xml_path}')

# Verify no placeholders remain
for root_dir, dirs, files in os.walk(TMP_DIR):
    for fname in files:
        fpath = os.path.join(root_dir, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            matches = re.findall(r'\{\{[^}]+\}\}', content)
            if matches:
                print(f'WARNING: Remaining placeholders in {fpath}: {matches}')
        except:
            pass

# Re-zip as HWPX
# CRITICAL: mimetype file must be first entry, stored (not deflated)
with zipfile.ZipFile(OUTPUT, 'w') as zout:
    mimetype_path = os.path.join(TMP_DIR, 'mimetype')
    if os.path.exists(mimetype_path):
        zout.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    for root_dir, dirs, files in os.walk(TMP_DIR):
        for fname in files:
            fpath = os.path.join(root_dir, fname)
            arcname = os.path.relpath(fpath, TMP_DIR)
            if arcname == 'mimetype':
                continue  # Already added
            zout.write(fpath, arcname, compress_type=zipfile.ZIP_DEFLATED)

print(f'Output written to {OUTPUT}')
```

### 3. Adjust the script as needed
- First, inspect the actual files to find the correct paths (the template and JSON might be in `/root/` or a subdirectory).
- Inspect the JSON structure to understand if keys are flat or nested, and adjust the placeholder matching accordingly.
- If the JSON has nested structures (e.g., arrays of items for a table), handle row-level placeholders appropriately.

### 4. Validation checks after running the script
- Verify no `{{...}}` placeholders remain: `unzip -p /root/inventory_report_ready.hwpx | grep -o '{{[^}]*}}'` (should return nothing).
- Verify the output is a valid ZIP: `unzip -t /root/inventory_report_ready.hwpx`.
- Verify `mimetype` is the first entry: `unzip -l /root/inventory_report_ready.hwpx | head`.
- Verify no `lineSegArray` elements exist in modified paragraphs by grepping the XML content.
- Spot-check that Korean labels and empty paragraphs are preserved by comparing structure.

### 5. Critical details from prior feedback
- **Split placeholders**: `{{placeholder}}` text is often split across multiple `<hp:t>` tags in HWPX XML. You MUST consolidate text within a paragraph before doing replacements.
- **Layout cache removal**: Remove `<hp:lineSegArray>` elements from ANY paragraph whose text was modified. Use BOTH XML tree removal AND regex removal as a belt-and-suspenders approach. The prior failure on a similar task was caused by lineSegArray elements surviving regex-only removal.
- **mimetype file**: Must be the first entry in the ZIP and stored with `ZIP_STORED` (no compression).
- **Namespace handling**: Register all XML namespaces found in the file before parsing/writing to avoid `ns0:` prefix corruption.
- **Empty paragraphs**: Do not remove or modify paragraphs that have no placeholders.

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