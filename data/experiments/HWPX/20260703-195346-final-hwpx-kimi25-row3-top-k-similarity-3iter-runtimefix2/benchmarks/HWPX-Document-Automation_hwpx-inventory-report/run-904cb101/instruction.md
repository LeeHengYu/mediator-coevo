# Task Instruction

Complete the inventory status report by filling the HWPX template with JSON data.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
ls /root/inventory_report_template.hwpx 2>/dev/null || find / -name 'inventory_report_template.hwpx' 2>/dev/null
ls /root/inventory_data.json 2>/dev/null || find / -name 'inventory_data.json' 2>/dev/null
```

### 2. Read the JSON data
```bash
cat /root/inventory_data.json
```
Note all key-value pairs. These will be used to replace `{{key}}` placeholders.

### 3. Extract the HWPX archive
HWPX files are ZIP archives. Extract to a working directory:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip -o /root/inventory_report_template.hwpx
```
List all extracted files:
```bash
find /tmp/hwpx_work -type f
```

### 4. Identify XML files containing placeholders
Search for `{{` in all extracted files:
```bash
grep -rl '{{' /tmp/hwpx_work/
```
Read each file that contains placeholders to understand the XML structure and namespace declarations.

### 5. Write a Python script to perform the replacements
Create `/tmp/fill_template.py` with the following logic:

```python
import json
import os
import re
import zipfile
from lxml import etree

# Load JSON data
with open('/root/inventory_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Build a flat replacement map from the JSON.
# If the JSON is nested, flatten it so that keys like "report_date" or
# "items[0].name" map to their values. Inspect the actual JSON structure
# and the placeholder names in the XML to determine the correct mapping.
# For now, assume a flat dict or adapt as needed after inspection.

def flatten_json(obj, prefix=''):
    """Flatten nested JSON into dot-separated keys."""
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v)
    return items

replacements = flatten_json(data)
# Also keep top-level keys without prefix
if isinstance(data, dict):
    for k, v in data.items():
        if not isinstance(v, (dict, list)):
            replacements[k] = str(v)

print("Replacement map:")
for k, v in sorted(replacements.items()):
    print(f"  {k} -> {v}")

# Find all XML files with placeholders
work_dir = '/tmp/hwpx_work'
placeholder_files = []
for root, dirs, files in os.walk(work_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        if fname.endswith('.xml'):
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            if '{{' in content:
                placeholder_files.append(fpath)

print(f"\nFiles with placeholders: {placeholder_files}")

# Process each file
for fpath in placeholder_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        raw_xml = f.read()

    # Register all namespaces to prevent ns0: prefix corruption
    # Parse once to discover namespaces
    ns_map = {}
    for event, elem in etree.iterparse(fpath, events=['start-ns']):
        prefix, uri = elem
        if prefix not in ns_map:
            ns_map[prefix] = uri
    for prefix, uri in ns_map.items():
        if prefix:
            etree.register_namespace(prefix, uri)
        else:
            etree.register_namespace('', uri)

    tree = etree.parse(fpath)
    root_elem = tree.getroot()

    # Find the hp namespace URI
    hp_ns = None
    for prefix, uri in ns_map.items():
        if prefix == 'hp' or 'hwpml' in uri.lower() or 'hancom' in uri.lower():
            hp_ns = uri
            break
    # Also try to find it from the root element's nsmap
    if hp_ns is None and hasattr(root_elem, 'nsmap'):
        for prefix, uri in root_elem.nsmap.items():
            if prefix == 'hp':
                hp_ns = uri
                break

    modified_paragraphs = set()

    # Walk all elements looking for text with {{ }}
    for elem in root_elem.iter():
        for attr in ['text', 'tail']:
            val = getattr(elem, attr)
            if val and '{{' in val:
                # Replace all {{key}} patterns
                new_val = val
                for match in re.findall(r'\{\{([^}]+)\}\}', val):
                    key = match.strip()
                    if key in replacements:
                        new_val = new_val.replace('{{' + match + '}}', replacements[key])
                    else:
                        print(f"  WARNING: No replacement found for key '{key}'")
                setattr(elem, attr, new_val)
                # Track the paragraph ancestor for lineSegArray removal
                modified_paragraphs.add(id(elem))
                # Walk up to find paragraph element
                parent_map_needed = True

    # Build parent map for lineSegArray removal
    parent_map = {}
    for parent in root_elem.iter():
        for child in parent:
            parent_map[id(child)] = parent

    # For each modified element, walk up to find the enclosing <hp:p> or <p> paragraph
    # and remove any <hp:lineSegArray> children
    def find_paragraph_ancestor(elem_id, parent_map, all_elems):
        """Walk up from elem to find paragraph."""
        current_id = elem_id
        visited = set()
        while current_id in parent_map and current_id not in visited:
            visited.add(current_id)
            parent = parent_map[current_id]
            tag = parent.tag
            local_tag = tag.split('}')[-1] if '}' in tag else tag
            if local_tag == 'p':
                return parent
            current_id = id(parent)
        return None

    # Build id -> element map
    id_to_elem = {id(e): e for e in root_elem.iter()}

    # Rebuild parent_map with element references
    parent_map_elems = {}
    for parent in root_elem.iter():
        for child in parent:
            parent_map_elems[id(child)] = parent

    paragraphs_to_clean = set()
    for mod_id in modified_paragraphs:
        # The modified element itself or walk up
        elem = id_to_elem.get(mod_id)
        if elem is not None:
            tag = elem.tag
            local_tag = tag.split('}')[-1] if '}' in tag else tag
            if local_tag == 'p':
                paragraphs_to_clean.add(id(elem))
            else:
                # Walk up
                current = mod_id
                while current in parent_map_elems:
                    p = parent_map_elems[current]
                    ptag = p.tag.split('}')[-1] if '}' in p.tag else p.tag
                    if ptag == 'p':
                        paragraphs_to_clean.add(id(p))
                        break
                    current = id(p)

    # Remove lineSegArray from modified paragraphs
    removed_count = 0
    for para_id in paragraphs_to_clean:
        para = id_to_elem.get(para_id)
        if para is not None:
            for child in list(para):
                child_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
                if child_tag == 'lineSegArray':
                    para.remove(child)
                    removed_count += 1

    print(f"  {fpath}: removed {removed_count} lineSegArray elements from {len(paragraphs_to_clean)} paragraphs")

    # Write back
    tree.write(fpath, xml_declaration=True, encoding='UTF-8')

# Verify no remaining placeholders
print("\nVerification - remaining placeholders:")
for root_d, dirs, files in os.walk(work_dir):
    for fname in files:
        fpath = os.path.join(root_d, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            matches = re.findall(r'\{\{[^}]+\}\}', content)
            if matches:
                print(f"  REMAINING in {fpath}: {matches}")
        except:
            pass

# Repackage as HWPX
output_path = '/root/inventory_report_ready.hwpx'
if os.path.exists(output_path):
    os.remove(output_path)

# mimetype must be first entry, stored (not compressed)
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    # Add mimetype first, uncompressed
    mimetype_path = os.path.join(work_dir, 'mimetype')
    if os.path.exists(mimetype_path):
        zf.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)

    # Add all other files
    for root_d, dirs, files in os.walk(work_dir):
        for fname in files:
            fpath = os.path.join(root_d, fname)
            arcname = os.path.relpath(fpath, work_dir)
            if arcname == 'mimetype':
                continue
            zf.write(fpath, arcname)

print(f"\nOutput written to {output_path}")
print(f"Output size: {os.path.getsize(output_path)} bytes")
```

### 6. Run the script
```bash
python3 /tmp/fill_template.py
```

**IMPORTANT**: After running, carefully check the output for:
- Any `WARNING: No replacement found` messages — if these appear, inspect the JSON structure and placeholder names, then adjust the replacement mapping logic accordingly.
- Any `REMAINING` placeholder warnings — these must be zero.
- If the JSON has nested structure (e.g., list of items), you may need to adapt the flattening logic or use a different key naming convention that matches the `{{...}}` placeholders in the XML.

### 7. Adapt if needed
If the placeholder keys don't match the flattened JSON keys:
1. Print all placeholder patterns found in the XML: `grep -oP '\{\{[^}]+\}\}' /tmp/hwpx_work/Contents/*.xml`
2. Print all JSON keys from the flattened map
3. Create a manual mapping between them
4. Update the script and re-run

### 8. Final verification
```bash
# Verify it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx'); print('Valid ZIP, entries:', len(z.namelist())); z.close()"

# Verify no placeholders remain
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/inventory_report_ready.hwpx')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8')
        matches = re.findall(r'\{\{[^}]+\}\}', content)
        if matches:
            print(f'FAIL: {name} has placeholders: {matches}')
    except: pass
print('Verification complete')
z.close()"
```

### 9. Run the verifier if available
```bash
cd /root && find . -name 'test_output*' -o -name 'test_*.py' | head -5
# If tests exist:
cd /root && python3 -m pytest tests/ -v 2>&1 | tail -30
```

## Critical Notes
- **Namespace registration**: Before parsing XML with lxml, register ALL namespaces found in the document to prevent `ns0:` prefix corruption. This is essential.
- **lineSegArray removal**: For every paragraph where text was modified, remove `<hp:lineSegArray>` (or equivalent) child elements. This forces the HWP viewer to recalculate text layout, preventing overlapping characters.
- **mimetype handling**: The `mimetype` file must be the FIRST entry in the ZIP and must be STORED (not DEFLATED).
- **Empty paragraphs**: Do not remove any paragraphs, even empty ones. The document structure must be preserved.
- **Korean text**: Do not modify any Korean labels or static note lines that don't contain `{{...}}` placeholders.

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