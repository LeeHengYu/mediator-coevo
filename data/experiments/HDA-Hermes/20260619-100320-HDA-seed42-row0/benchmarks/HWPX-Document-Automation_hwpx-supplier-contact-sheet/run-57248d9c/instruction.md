# Task Instruction

Complete the following task to update an HWPX supplier contact sheet template with JSON data.

## Goal
Replace all `{{...}}` placeholders in `supplier_contact_template.hwpx` with values from `supplier_contact.json`, and save the result to `/root/supplier_contact_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `supplier_contact.json` to understand the keys and values available.
- List the contents of `supplier_contact_template.hwpx` (it's a ZIP archive) to understand its structure.
- Extract and inspect the main content XML file(s) (typically `Contents/section0.xml` or similar) to see the placeholders and XML structure.
- Note the namespace used (typically `hp` namespace like `http://www.hancom.co.kr/hwpml/2011/paragraph`).

### 2. Write and run a Python script that does the following:

```python
import json, os, re, shutil, zipfile
import xml.etree.ElementTree as ET

# Load JSON data
with open('supplier_contact.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# If data is a dict, use it directly. If it has nested structure, flatten as needed.

template_path = 'supplier_contact_template.hwpx'
output_path = '/root/supplier_contact_ready.hwpx'
extract_dir = '/tmp/hwpx_extracted'

# Clean extraction directory
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

# Extract the HWPX ZIP
with zipfile.ZipFile(template_path, 'r') as z:
    z.extractall(extract_dir)
    namelist = z.namelist()

# Find all XML files in Contents/ directory
xml_files = [n for n in namelist if n.startswith('Contents/') and n.endswith('.xml')]

for xml_file in xml_files:
    filepath = os.path.join(extract_dir, xml_file)
    with open(filepath, 'r', encoding='utf-8') as f:
        xml_content = f.read()
    
    # STEP A: Consolidate split placeholder text within <hp:run> elements.
    # Placeholders like {{key}} often get split across multiple <hp:t> tags.
    # Strategy: Within each <hp:run>, merge all <hp:t> text into the first <hp:t>
    # and remove subsequent <hp:t> tags.
    
    # Use regex to find each <hp:run>...</hp:run> block and consolidate <hp:t> tags
    def consolidate_run(match):
        run_xml = match.group(0)
        # Find all <hp:t> content
        t_pattern = re.compile(r'<hp:t[^>]*>(.*?)</hp:t>', re.DOTALL)
        t_matches = list(t_pattern.finditer(run_xml))
        if len(t_matches) <= 1:
            return run_xml
        # Merge all text into first <hp:t>, remove others
        merged_text = ''.join(m.group(1) for m in t_matches)
        # Replace first <hp:t>...</hp:t> with merged text
        first = t_matches[0]
        new_t = run_xml[first.start()-0:first.start()] # not needed
        # Rebuild: keep everything before first <hp:t>, replace first with merged, remove rest
        result = run_xml[:t_matches[0].start()] + '<hp:t>' + merged_text + '</hp:t>' + run_xml[t_matches[0].end():]
        # Now remove remaining <hp:t>...</hp:t>
        for m in reversed(t_matches[1:]):
            # Adjust positions since we modified the string
            pass
        # Simpler approach: do it in two passes
        return result
    
    # Actually, a cleaner approach:
    def consolidate_run_v2(match):
        run_xml = match.group(0)
        t_pattern = re.compile(r'<hp:t[^>]*>(.*?)</hp:t>', re.DOTALL)
        t_matches = list(t_pattern.finditer(run_xml))
        if len(t_matches) <= 1:
            return run_xml
        merged_text = ''.join(m.group(1) for m in t_matches)
        # Remove all <hp:t>...</hp:t> tags
        cleaned = t_pattern.sub('', run_xml, count=len(t_matches))
        # Insert merged <hp:t> before </hp:run>
        cleaned = cleaned.replace('</hp:run>', '<hp:t>' + merged_text + '</hp:t></hp:run>')
        return cleaned
    
    xml_content = re.sub(r'<hp:run\b[^>]*>.*?</hp:run>', consolidate_run_v2, xml_content, flags=re.DOTALL)
    
    # STEP B: Replace all {{key}} placeholders with JSON values
    modified = False
    for key, value in data.items():
        placeholder = '{{' + key + '}}'
        if placeholder in xml_content:
            xml_content = xml_content.replace(placeholder, str(value))
            modified = True
    
    # Also handle any remaining placeholders that might use nested keys
    # Check if any {{...}} remain
    remaining = re.findall(r'\{\{([^}]+)\}\}', xml_content)
    if remaining:
        print(f'WARNING: Remaining placeholders in {xml_file}: {remaining}')
        # Try to resolve from nested data
        for key in remaining:
            parts = key.split('.')
            val = data
            try:
                for p in parts:
                    if isinstance(val, dict):
                        val = val[p]
                    elif isinstance(val, list):
                        val = val[int(p)]
                xml_content = xml_content.replace('{{' + key + '}}', str(val))
                modified = True
            except (KeyError, IndexError, ValueError):
                print(f'Could not resolve placeholder: {key}')
    
    # STEP C: Remove <hp:lineSegArray> elements from modified paragraphs
    # Use ElementTree for reliable removal
    # But first, let's try a robust regex that handles the namespace
    # Actually, per feedback, regex can miss variations. Let's use ET.
    
    # Parse with ElementTree, find and remove all lineSegArray in paragraphs that were modified
    # Since we can't easily track which paragraphs were modified at the ET level,
    # and the requirement says "Any paragraph whose text you modify must not retain stale layout-cache elements",
    # the safest approach is to remove ALL lineSegArray elements (this worked in successful runs).
    
    # Save the text-replaced XML first, then parse with ET to remove lineSegArray
    # Register namespaces to preserve them
    # First, extract namespace declarations from the XML
    ns_pattern = re.compile(r'xmlns:(\w+)="([^"]+)"')
    for prefix, uri in ns_pattern.findall(xml_content):
        ET.register_namespace(prefix, uri)
    # Also register default namespace if present
    default_ns = re.search(r'xmlns="([^"]+)"', xml_content)
    if default_ns:
        ET.register_namespace('', default_ns.group(1))
    
    tree = ET.fromstring(xml_content)
    
    # Find all lineSegArray elements (with namespace)
    # Try multiple namespace possibilities
    namespaces_to_try = [
        'http://www.hancom.co.kr/hwpml/2011/paragraph',
        'http://www.hancom.co.kr/hwpml/2016/paragraph',
    ]
    
    removed = False
    for ns_uri in namespaces_to_try:
        tag = '{%s}lineSegArray' % ns_uri
        for elem in tree.iter():
            children_to_remove = [child for child in elem if child.tag == tag]
            for child in children_to_remove:
                elem.remove(child)
                removed = True
    
    # Also try without specific namespace - just find any element ending with lineSegArray
    for elem in tree.iter():
        children_to_remove = [child for child in elem if child.tag.endswith('lineSegArray')]
        for child in children_to_remove:
            elem.remove(child)
            removed = True
    
    if removed:
        print(f'Removed lineSegArray elements from {xml_file}')
    
    # Write back
    xml_output = ET.tostring(tree, encoding='unicode', xml_declaration=False)
    # Preserve XML declaration if original had one
    if xml_content.startswith('<?xml'):
        decl_end = xml_content.index('?>') + 2
        xml_output = xml_content[:decl_end] + '\n' + xml_output
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(xml_output)

# STEP D: Re-zip into HWPX with mimetype stored first
with zipfile.ZipFile(output_path, 'w') as zout:
    # Add mimetype first, stored (not deflated)
    mimetype_path = os.path.join(extract_dir, 'mimetype')
    if os.path.exists(mimetype_path):
        zout.write(mimetype_path, 'mimetype', compress_type=zipfile.ZIP_STORED)
    
    # Add all other files with deflation
    for root, dirs, files in os.walk(extract_dir):
        for fname in sorted(files):
            full_path = os.path.join(root, fname)
            arcname = os.path.relpath(full_path, extract_dir)
            if arcname == 'mimetype':
                continue
            zout.write(full_path, arcname, compress_type=zipfile.ZIP_DEFLATED)

print('Output written to', output_path)
```

**IMPORTANT**: The above is a reference script. Before writing the final script:
1. First inspect `supplier_contact.json` to understand the data structure.
2. First extract and inspect the XML content of the HWPX to understand the actual namespace URIs, placeholder format, and structure.
3. Adapt the script based on what you find.

### 3. Verification
After running the script:
- Verify `/root/supplier_contact_ready.hwpx` exists.
- Extract it and check the content XML to confirm:
  - No `{{...}}` placeholders remain anywhere in any XML file.
  - Korean field labels are preserved.
  - Static note lines are unchanged.
  - No `lineSegArray` elements exist in paragraphs that had text modifications.
- Verify it's a valid ZIP file.
- If any `{{...}}` placeholders remain, debug by checking the JSON keys vs placeholder names and fix.

### Key Technical Notes (from cross-task feedback)
- HWPX placeholders are frequently split across multiple `<hp:t>` XML tags within a single `<hp:run>`. You MUST consolidate these before replacement.
- Use XML parser (ElementTree) for lineSegArray removal rather than pure regex, as regex can miss namespace variations.
- The `mimetype` file must be the first entry in the ZIP and stored without compression.
- After consolidation, verify all `{{...}}` patterns are gone with a simple grep/search.

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