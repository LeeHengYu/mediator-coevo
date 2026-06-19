# Task Instruction

Automate the HWPX supplier contact sheet by replacing all `{{...}}` placeholders with values from the JSON file and saving the result as a valid `.hwpx` package.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
cat /root/supplier_contact.json
```
Understand the JSON keys and values.

### 2. Inspect the HWPX template
```bash
cd /root
python3 -c "
import zipfile, sys
with zipfile.ZipFile('supplier_contact_template.hwpx','r') as z:
    for info in z.infolist():
        print(info.filename, info.compress_type, info.file_size)
"
```
Then extract and print every XML section file (typically `Contents/section0.xml`, possibly more) to see the placeholders:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('supplier_contact_template.hwpx','r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            print('=== ' + name + ' ===')
            print(z.read(name).decode('utf-8'))
"
```
Also print any non-XML files that might contain placeholders (e.g., `mimetype`, `META-INF/*`).

### 3. Identify all placeholders
List every `{{...}}` token found across all files inside the zip. Confirm each one has a matching key in the JSON. If a JSON key maps to a value containing Korean text, note it for preservation.

### 4. Write the automation script
Create `/root/build.py` with this logic:

```python
import zipfile, json, re, os, shutil
from lxml import etree

SOURCE = '/root/supplier_contact_template.hwpx'
OUTPUT = '/root/supplier_contact_ready.hwpx'
JSON_FILE = '/root/supplier_contact.json'

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Flatten nested JSON if needed (check structure first)
# If data is flat dict, use directly; if nested, flatten with dot notation or
# match the placeholder names to the JSON paths.

def replace_placeholders(text, mapping):
    """Replace all {{key}} with mapping[key]."""
    def replacer(m):
        key = m.group(1).strip()
        if key in mapping:
            return str(mapping[key])
        # Try without spaces
        for k, v in mapping.items():
            if k.strip() == key:
                return str(v)
        return m.group(0)  # leave if no match (should not happen)
    return re.sub(r'\{\{\s*([^}]+?)\s*\}\}', replacer, text)

# Build a flat replacement map from JSON
# Adjust this section after inspecting the actual JSON structure
def flatten(obj, prefix=''):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (dict, list)):
                items.update(flatten(v, new_key))
            else:
                items[new_key] = v
                items[k] = v  # also store without prefix
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(flatten(v, f"{prefix}[{i}]"))
    return items

mapping = flatten(data)

# Read template, process, write output
with zipfile.ZipFile(SOURCE, 'r') as zin:
    # Get mimetype content if present
    namelist = zin.namelist()
    
    with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in namelist:
            raw = zin.read(item)
            
            # mimetype must be first and uncompressed (ODF convention)
            if item == 'mimetype':
                zout.writestr(item, raw, compress_type=zipfile.ZIP_STORED)
                continue
            
            # Process XML files that may contain placeholders
            if item.endswith('.xml'):
                text = raw.decode('utf-8')
                if '{{' in text:
                    # Parse XML, do text replacement, remove lineSegArray from modified paragraphs
                    # Use namespace-aware parsing
                    root = etree.fromstring(raw)
                    nsmap = {
                        'hp': root.nsmap.get('hp', None),
                        'ha': root.nsmap.get('ha', None),
                        'hc': root.nsmap.get('hc', None),
                    }
                    # Remove None entries
                    nsmap = {k: v for k, v in nsmap.items() if v is not None}
                    
                    modified_paragraphs = set()
                    
                    # Walk all text nodes and replace placeholders
                    for elem in root.iter():
                        if elem.text and '{{' in elem.text:
                            elem.text = replace_placeholders(elem.text, mapping)
                            # Track the parent <hp:p> paragraph
                            modified_paragraphs.add(id(elem))
                        if elem.tail and '{{' in elem.tail:
                            elem.tail = replace_placeholders(elem.tail, mapping)
                            modified_paragraphs.add(id(elem))
                    
                    # Remove lineSegArray from any paragraph that contains modified text
                    # Walk paragraphs and check if any descendant was modified
                    for p in root.iter():
                        if p.tag.endswith('}p') or p.tag == 'p':
                            # Check if any descendant text was modified
                            dominated = False
                            for desc in p.iter():
                                if id(desc) in modified_paragraphs:
                                    dominated = True
                                    break
                            if not dominated:
                                # Also check if paragraph text itself has been replaced
                                # by checking if placeholder pattern was in original
                                continue
                            # Remove lineSegArray children
                            for child in list(p):
                                if child.tag.endswith('}lineSegArray') or child.tag == 'lineSegArray':
                                    p.remove(child)
                    
                    # Also do a broader approach: find ALL paragraphs, check if their
                    # serialized text differs from original, remove lineSegArray
                    # Simpler: just remove lineSegArray from any <hp:p> that has text
                    # matching a replaced value. But safest: for every <hp:p>, check
                    # if its full text content has no {{ }} - if original did have {{ }}
                    # then remove lineSegArray.
                    
                    output_bytes = etree.tostring(root, xml_declaration=True, encoding='UTF-8')
                    
                    # Final safety: verify no {{ }} remain
                    output_text = output_bytes.decode('utf-8')
                    remaining = re.findall(r'\{\{[^}]+\}\}', output_text)
                    if remaining:
                        print(f'WARNING: unreplaced placeholders in {item}: {remaining}')
                    
                    zout.writestr(item, output_bytes)
                else:
                    zout.writestr(item, raw)
            else:
                # Non-XML files: check for placeholders in text-like files
                try:
                    text = raw.decode('utf-8')
                    if '{{' in text:
                        text = replace_placeholders(text, mapping)
                        zout.writestr(item, text.encode('utf-8'))
                    else:
                        zout.writestr(item, raw)
                except UnicodeDecodeError:
                    zout.writestr(item, raw)

print('Output written to', OUTPUT)

# Verify
with zipfile.ZipFile(OUTPUT, 'r') as z:
    for name in z.namelist():
        content = z.read(name)
        try:
            text = content.decode('utf-8')
            found = re.findall(r'\{\{[^}]+\}\}', text)
            if found:
                print(f'ERROR: {name} still has placeholders: {found}')
        except:
            pass
    print('Verification complete. File count:', len(z.namelist()))
```

### 5. Adjust the script after inspection
After step 2-3, you will know:
- The exact JSON structure (flat or nested)
- The exact placeholder names in the XML
- Which namespaces are used

Adjust the `flatten` function and `mapping` construction so every `{{placeholder}}` maps correctly to a JSON value. **Pay special attention to**:
- Placeholders that might reference nested JSON paths (e.g., `{{company.name}}`)
- Array-indexed placeholders (e.g., `{{contacts[0].name}}`)
- Korean text values - ensure they are preserved exactly

### 6. Handle lineSegArray removal properly
The previous successful run confirmed that removing `<hp:lineSegArray>` (or whatever the local tag name is) from modified `<hp:p>` paragraphs prevents layout-cache corruption. The script above implements this. After inspecting the actual namespace URIs, adjust the tag matching accordingly.

A more robust approach for lineSegArray removal: after all text replacements, re-serialize the XML to string, then re-parse and for each paragraph element, check if the serialized paragraph text (all descendant text concatenated) differs from the original. If it does, remove lineSegArray. Alternatively, since we know which elements had `{{` in them, track the ancestor `<hp:p>` elements directly.

### 7. Run the script
```bash
cd /root && python3 build.py
```

### 8. Validate the output
```bash
# Check it's a valid zip
python3 -c "
import zipfile
with zipfile.ZipFile('/root/supplier_contact_ready.hwpx','r') as z:
    print('Files:', z.namelist())
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            if '{{' in content:
                print('FAIL: placeholder remains in', name)
            else:
                print('OK:', name)
    print('Valid zip: OK')
"
```

### 9. Run the verifier
```bash
cd /root && python3 -m pytest test_output.py -v
```
If any test fails, read the error carefully, inspect what the test expects, fix the script, and re-run.

### Key constraints to remember:
- **No `{{...}}` placeholders may remain** in any file in the output package
- **Korean field labels** in the document must be preserved (don't replace label text, only placeholder values)
- **Static note lines** must be unchanged
- **`mimetype` file** must be stored first and uncompressed in the zip
- **lineSegArray removal** from modified paragraphs to prevent layout corruption
- The output must be at exactly `/root/supplier_contact_ready.hwpx`

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