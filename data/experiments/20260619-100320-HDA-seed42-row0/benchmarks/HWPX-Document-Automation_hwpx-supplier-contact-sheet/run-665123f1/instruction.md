# Task Instruction

Execute the following steps to complete the HWPX supplier contact sheet task:

1. **Inspect the workspace**: List files in the task directory to locate `supplier_contact_template.hwpx` and `supplier_contact.json`. Read `supplier_contact.json` to understand all key-value pairs.

2. **Write and run a Python script** that does the following:

```python
import json, os, shutil, zipfile, re, glob
from pathlib import Path

# Paths
template_path = None
json_path = None

# Find the files
for root, dirs, files in os.walk('/root'):
    for f in files:
        if f == 'supplier_contact_template.hwpx':
            template_path = os.path.join(root, f)
        if f == 'supplier_contact.json':
            json_path = os.path.join(root, f)

assert template_path, 'Template not found'
assert json_path, 'JSON not found'

print(f'Template: {template_path}')
print(f'JSON: {json_path}')

# Load JSON values
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f'JSON keys: {list(data.keys()) if isinstance(data, dict) else type(data)}')
print(f'JSON data: {json.dumps(data, ensure_ascii=False, indent=2)}')

# Extract HWPX (it's a ZIP)
extract_dir = '/tmp/hwpx_extract'
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir)

with zipfile.ZipFile(template_path, 'r') as zf:
    zf.extractall(extract_dir)

print('Extracted files:')
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        fp = os.path.join(root, f)
        print(f'  {fp}')

# Find all XML files in Contents/ that may contain placeholders
xml_files = []
for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.endswith('.xml'):
            xml_files.append(os.path.join(root, f))

print(f'XML files: {xml_files}')

# Build replacement map: flatten nested JSON if needed
def flatten_json(obj, prefix=''):
    items = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
            if isinstance(v, (dict, list)):
                items.update(flatten_json(v, new_key))
            else:
                items[new_key] = str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items.update(flatten_json(v, f'{prefix}[{i}]'))
    return items

# First try simple dict mapping
replacements = {}
if isinstance(data, dict):
    for k, v in data.items():
        if isinstance(v, (str, int, float, bool)):
            replacements[k] = str(v)
    # Also flatten for nested
    flat = flatten_json(data)
    replacements.update(flat)

print(f'Replacement keys: {list(replacements.keys())}')

# Process each XML file
import xml.etree.ElementTree as ET

# First, let's look at the raw XML to understand placeholder format
for xf in xml_files:
    with open(xf, 'r', encoding='utf-8') as f:
        content = f.read()
    if '{{' in content or 'hp:t' in content:
        print(f'\n=== {xf} (first 5000 chars) ===')
        print(content[:5000])
        # Find all placeholders
        placeholders = re.findall(r'\{\{[^}]*\}\}', content)
        print(f'Placeholders found directly: {placeholders}')

# Now do the actual replacement with split-tag awareness
for xf in xml_files:
    with open(xf, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '{{' not in content and '}}' not in content:
        continue
    
    print(f'\nProcessing: {xf}')
    
    # Strategy: First try direct replacement on the raw XML string
    # for any complete {{key}} patterns
    original_content = content
    
    for key, value in replacements.items():
        placeholder = '{{' + key + '}}'
        if placeholder in content:
            content = content.replace(placeholder, value)
            print(f'  Replaced: {placeholder} -> {value}')
    
    # Check if any {{...}} remain - they might be split across tags
    remaining = re.findall(r'\{\{[^}]*\}\}', content)
    if remaining:
        print(f'  Remaining complete placeholders: {remaining}')
    
    # Handle split placeholders: strip all XML tags, find placeholders,
    # then work on the tag level
    # More robust: consolidate <hp:t> text within each <hp:run>
    # We'll use namespace-aware parsing
    
    # Register namespaces to preserve them
    # Parse the namespace declarations from the file
    ns_matches = re.findall(r'xmlns:(\w+)="([^"]+)"', content)
    for prefix, uri in ns_matches:
        ET.register_namespace(prefix, uri)
    # Also default namespace
    default_ns = re.findall(r'xmlns="([^"]+)"', content)
    for uri in default_ns:
        ET.register_namespace('', uri)
    
    tree = ET.parse(xf if content == original_content else None)
    # Actually, let's write modified content first, then parse
    with open(xf, 'w', encoding='utf-8') as f:
        f.write(content)
    
    tree = ET.fromstring(content)
    
    # Find namespace for hp
    hp_ns = None
    for prefix, uri in ns_matches:
        if prefix == 'hp':
            hp_ns = uri
            break
    
    if hp_ns:
        # Consolidate split placeholders within runs
        # Find all run elements
        for run in tree.iter(f'{{{hp_ns}}}run'):
            t_elements = list(run.iter(f'{{{hp_ns}}}t'))
            if len(t_elements) > 1:
                combined = ''.join((t.text or '') for t in t_elements)
                if '{{' in combined:
                    # Replace in combined text
                    for key, value in replacements.items():
                        placeholder = '{{' + key + '}}'
                        combined = combined.replace(placeholder, value)
                    # Put all text in first t, clear rest
                    t_elements[0].text = combined
                    for t in t_elements[1:]:
                        t.text = ''
        
        # Remove lineSegArray from paragraphs that were modified
        # Actually, remove from ALL paragraphs to be safe
        for p in tree.iter(f'{{{hp_ns}}}p'):
            for lsa in list(p.iter(f'{{{hp_ns}}}lineSegArray')):
                # Find parent and remove
                for parent in p.iter():
                    if lsa in list(parent):
                        parent.remove(lsa)
                        break
        
        # Write back
        final_xml = ET.tostring(tree, encoding='unicode', xml_declaration=True)
        # Restore original XML declaration if needed
        if content.startswith('<?xml'):
            decl_match = re.match(r'<\?xml[^?]*\?>', content)
            final_decl = re.match(r'<\?xml[^?]*\?>', final_xml)
            if decl_match and final_decl:
                final_xml = decl_match.group() + final_xml[len(final_decl.group()):]
        
        with open(xf, 'w', encoding='utf-8') as f:
            f.write(final_xml)
        
        # Verify no placeholders remain
        with open(xf, 'r', encoding='utf-8') as f:
            verify = f.read()
        remaining_final = re.findall(r'\{\{[^}]*\}\}', verify)
        if remaining_final:
            print(f'  WARNING: Still remaining: {remaining_final}')
        else:
            print(f'  All placeholders replaced successfully')
    else:
        # No hp namespace, just do string replacement
        with open(xf, 'w', encoding='utf-8') as f:
            f.write(content)

# Re-package as HWPX
output_path = '/root/supplier_contact_ready.hwpx'
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            fp = os.path.join(root, f)
            arcname = os.path.relpath(fp, extract_dir)
            # mimetype should be stored, not deflated
            if f == 'mimetype':
                zf.write(fp, arcname, compress_type=zipfile.ZIP_STORED)
            else:
                zf.write(fp, arcname)

print(f'\nOutput written to: {output_path}')
print(f'Output size: {os.path.getsize(output_path)} bytes')

# Final verification: unzip and check for any remaining placeholders
verify_dir = '/tmp/hwpx_verify'
if os.path.exists(verify_dir):
    shutil.rmtree(verify_dir)
with zipfile.ZipFile(output_path, 'r') as zf:
    zf.extractall(verify_dir)

all_clean = True
for root, dirs, files in os.walk(verify_dir):
    for f in files:
        if f.endswith('.xml'):
            fp = os.path.join(root, f)
            with open(fp, 'r', encoding='utf-8') as fh:
                txt = fh.read()
            found = re.findall(r'\{\{[^}]*\}\}', txt)
            if found:
                print(f'VERIFICATION FAIL: {fp} still has: {found}')
                all_clean = False

if all_clean:
    print('VERIFICATION PASSED: No placeholders remain')
else:
    print('VERIFICATION FAILED: Some placeholders still present')
```

3. **Important**: The script above is a comprehensive template. Before running it as-is, first do a quick inspection run:
   - Find the template and JSON files
   - Read the JSON to understand its structure (it may be flat or nested)
   - Peek at the XML content in the HWPX to see the exact placeholder names and XML structure
   - Then adapt the replacement logic accordingly

4. **Split the work into phases**:
   - **Phase 1**: Locate files, read JSON, extract HWPX, inspect XML content and identify all placeholders and their exact format.
   - **Phase 2**: Write the processing script with the correct replacement mapping based on what you found in Phase 1. Key considerations:
     - Placeholders may be split across multiple `<hp:t>` tags — consolidate text within each `<hp:run>` before replacing.
     - Remove `<hp:lineSegArray>` elements from any modified `<hp:p>` paragraphs (or all paragraphs to be safe).
     - Register all XML namespaces before parsing to preserve them in output.
     - Preserve Korean field labels and static note lines.
   - **Phase 3**: Package the result as a ZIP (HWPX), with `mimetype` stored uncompressed at the archive root.
   - **Phase 4**: Verify the output contains zero `{{...}}` placeholders and is a valid ZIP.

5. **Output**: The final file must be saved to `/root/supplier_contact_ready.hwpx`.

6. **Validation**: After creating the output, unzip it and grep all XML files for `{{` to confirm no placeholders remain. Print the text content of the main section XML to visually confirm Korean labels are preserved and values are filled in correctly.

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