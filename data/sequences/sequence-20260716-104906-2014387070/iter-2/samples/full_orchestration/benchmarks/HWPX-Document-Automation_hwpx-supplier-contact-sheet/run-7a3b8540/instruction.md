# Task Instruction

Complete the following task step by step:

## Goal
Update the HWPX supplier contact sheet template by replacing all `{{...}}` placeholders with values from a JSON file, then save the result as a valid `.hwpx` package.

## Steps

### 1. Inspect the workspace
```bash
ls -la /root/
find /root/ -name 'supplier_contact*' -type f
```
Identify the template file `supplier_contact_template.hwpx` and `supplier_contact.json`.

### 2. Read the JSON data
```bash
cat supplier_contact.json
```
Note all key-value pairs. These are the replacement values for `{{key}}` placeholders.

### 3. Examine the HWPX structure
HWPX files are ZIP archives containing XML files. Unzip and inspect:
```bash
mkdir -p /tmp/hwpx_work
cp supplier_contact_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip -o template.zip -d template_contents
find template_contents -type f
```

### 4. Find all placeholders
Search for `{{` across all XML files to identify every placeholder and which files contain them:
```bash
grep -rn '{{' template_contents/
```

### 5. Write a Python script to perform the replacement
Create a Python script that:
- Loads the JSON file
- Iterates over every file in the extracted HWPX directory
- For XML files, parses them and replaces all `{{key}}` placeholders with corresponding JSON values in text nodes
- For each paragraph (`<hp:p>`) that contains a modified text run, removes any `<hp:linesegarray>` child element (layout cache) to prevent overlapping characters when the document is opened
- Preserves all Korean field labels and static note lines (only the placeholder text inside `{{...}}` is replaced, not surrounding text)
- Writes the modified XML back
- Repackages everything into a valid ZIP with `.hwpx` extension at `/root/supplier_contact_ready.hwpx`

Here is the recommended approach:

```python
import json, os, re, zipfile
from lxml import etree

# Load JSON
with open('supplier_contact.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Build replacement map: flatten nested JSON if needed
def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten(v, new_key, sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)

flat_data = flatten(data)
# Also keep non-flattened keys for simple lookups
for k, v in data.items():
    if not isinstance(v, dict):
        flat_data[k] = str(v)

print("Replacement map:", flat_data)

# Placeholder pattern
pattern = re.compile(r'\{\{([^}]+)\}\}')

def replace_placeholders(text):
    def replacer(m):
        key = m.group(1).strip()
        if key in flat_data:
            return flat_data[key]
        # Try case-insensitive or partial match
        for k, v in flat_data.items():
            if k.lower() == key.lower():
                return v
        print(f"WARNING: No replacement found for placeholder: {{{{{key}}}}}")
        return m.group(0)  # leave as-is if not found (should not happen)
    return pattern.sub(replacer, text)

# Process extracted files
extracted_dir = '/tmp/hwpx_work/template_contents'
modified_files = set()

for root, dirs, files in os.walk(extracted_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        if fname.endswith('.xml') or fname.endswith('.rels'):
            with open(fpath, 'rb') as f:
                raw = f.read()
            try:
                tree = etree.fromstring(raw)
            except:
                # Not valid XML, try text replacement
                text = raw.decode('utf-8', errors='replace')
                if '{{' in text:
                    new_text = replace_placeholders(text)
                    with open(fpath, 'wb') as f:
                        f.write(new_text.encode('utf-8'))
                    modified_files.add(fpath)
                continue
            
            nsmap = tree.nsmap
            # Find all text elements and replace placeholders
            changed_paragraphs = set()
            
            for elem in tree.iter():
                if elem.text and '{{' in elem.text:
                    elem.text = replace_placeholders(elem.text)
                    changed_paragraphs.add(elem)
                if elem.tail and '{{' in elem.tail:
                    elem.tail = replace_placeholders(elem.tail)
                    changed_paragraphs.add(elem)
            
            if changed_paragraphs:
                modified_files.add(fpath)
                # Remove linesegarray from parent paragraphs of changed elements
                # Walk up to find <hp:p> or <p> ancestors
                for changed_elem in changed_paragraphs:
                    # Find the paragraph ancestor
                    para = changed_elem
                    while para is not None:
                        tag = etree.QName(para.tag).localname if '}' in para.tag else para.tag
                        if tag == 'p':
                            # Remove linesegarray children
                            for child in list(para):
                                child_tag = etree.QName(child.tag).localname if '}' in child.tag else child.tag
                                if child_tag == 'linesegarray':
                                    para.remove(child)
                                    print(f"Removed linesegarray from paragraph in {fname}")
                            break
                        para = para.getparent()
                
                # Write back
                with open(fpath, 'wb') as f:
                    f.write(etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True))

print(f"Modified files: {modified_files}")
```

Run this script.

### 6. Verify no placeholders remain
```bash
grep -rn '{{' /tmp/hwpx_work/template_contents/
```
If any `{{...}}` remain, investigate the JSON keys and fix the replacement map (e.g., handle nested keys, array indices, or different naming conventions). Re-run until zero placeholders remain.

### 7. Repackage as HWPX
The HWPX ZIP must preserve the original directory structure. Use Python to create the ZIP:

```python
import zipfile, os

output_path = '/root/supplier_contact_ready.hwpx'
extracted_dir = '/tmp/hwpx_work/template_contents'

with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(extracted_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            arcname = os.path.relpath(fpath, extracted_dir)
            zf.write(fpath, arcname)

print(f"Created {output_path}")
```

### 8. Validate the output
```bash
# Verify it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/supplier_contact_ready.hwpx'); print(z.namelist()); z.close()"

# Verify no placeholders in the final package
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/supplier_contact_ready.hwpx')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8', errors='replace')
        matches = re.findall(r'\{\{[^}]+\}\}', content)
        if matches:
            print(f'FAIL: {name} still has placeholders: {matches}')
    except:
        pass
print('Validation complete')
z.close()
"
```

If any placeholders remain, go back and fix. The file `/root/supplier_contact_ready.hwpx` must exist, be a valid ZIP/HWPX, contain no `{{...}}` placeholders, preserve Korean labels, and have layout-cache elements removed from modified paragraphs.

### 9. Run the verifier if available
```bash
cd /root && find . -name 'test_output*' -o -name 'verify*' | head -5
# If a test file exists, run it:
python3 -m pytest test_output.py -v 2>&1 || true
```

## Critical Notes
- **Flatten nested JSON**: The JSON may have nested objects. Build a flat key map using dot notation (e.g., `address.city`) AND try simple key names.
- **Layout cache**: Always remove `<hp:linesegarray>` (or `<linesegarray>` in the hp namespace) from any `<hp:p>` paragraph whose text was modified. This is essential for the document to open cleanly.
- **Preserve structure**: Do not modify files that don't contain placeholders. Keep the ZIP structure identical to the original.
- **Korean text**: Only replace `{{...}}` tokens. Do not alter surrounding Korean text.
- **Static note line**: Leave any note/comment lines in the document unchanged.

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