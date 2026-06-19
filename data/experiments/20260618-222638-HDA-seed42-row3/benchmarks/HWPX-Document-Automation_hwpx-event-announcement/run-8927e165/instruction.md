# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Objective
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Step-by-step Plan

### 1. Inspect the workspace
- List files in the current directory to locate `event_announcement_template.hwpx` and `event_data.json`.
- Read `event_data.json` fully to understand all key-value pairs.

### 2. Understand the HWPX structure
- A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML).
- Unzip the template to a temporary directory (e.g., `/tmp/hwpx_work/`).
- List all files in the extracted archive.
- Identify which XML files contain body text. Typically this is `Contents/section0.xml` (or similar), but inspect the archive to confirm. Also check `Contents/content.hpf` or any manifest.

### 3. Read and parse the section XML
- Read the section XML file(s) that contain the document body.
- Search for all `{{...}}` placeholders across ALL XML files in the archive (not just section0.xml — check every `.xml` file).
- Note: Placeholders may be split across multiple `<hp:t>` text runs within a single `<hp:run>` or across multiple `<hp:run>` elements within one `<hp:p>` paragraph. This is the critical challenge.

### 4. Perform replacements using paragraph-level text merging
For each XML file that contains placeholders, use this approach:

```python
import json, os, re, shutil, zipfile
from lxml import etree

# Load JSON data
with open('event_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract HWPX
template_path = 'event_announcement_template.hwpx'
work_dir = '/tmp/hwpx_work'
if os.path.exists(work_dir):
    shutil.rmtree(work_dir)

with zipfile.ZipFile(template_path, 'r') as z:
    z.extractall(work_dir)

# Find all XML files
xml_files = []
for root, dirs, files in os.walk(work_dir):
    for fname in files:
        if fname.endswith('.xml'):
            xml_files.append(os.path.join(root, fname))

# Define namespace map dynamically from each file
def process_xml(filepath, data):
    with open(filepath, 'rb') as f:
        tree = etree.parse(f)
    root = tree.getroot()
    nsmap = root.nsmap
    
    # Build a reverse map to find the hp namespace prefix
    # The hp namespace is typically 'http://www.hancom.co.kr/hwpml/2011/paragraph'
    # or similar. We need to find <hp:p> paragraphs.
    # Strategy: find all elements whose local name is 'p' that contain text runs
    
    modified = False
    
    # Find all paragraph elements (any namespace, local name 'p')
    for p_elem in root.iter():
        if etree.QName(p_elem.tag).localname != 'p':
            continue
        
        # Collect all text nodes (<hp:t> or similar) within this paragraph
        t_elements = []
        for elem in p_elem.iter():
            if etree.QName(elem.tag).localname == 't':
                t_elements.append(elem)
        
        if not t_elements:
            continue
        
        # Merge all text content to check for placeholders
        full_text = ''.join((t.text or '') for t in t_elements)
        
        if '{{' not in full_text:
            continue
        
        # Replace all placeholders
        new_text = full_text
        for key, value in data.items():
            new_text = new_text.replace('{{' + key + '}}', str(value))
        
        if new_text == full_text:
            continue
        
        modified = True
        
        # Put all text into the first <t> element, clear the rest
        t_elements[0].text = new_text
        for t in t_elements[1:]:
            t.text = ''
        
        # Remove lineSegArray elements (layout cache) from this paragraph
        for child in list(p_elem):
            if etree.QName(child.tag).localname == 'lineSegArray':
                p_elem.remove(child)
    
    if modified:
        tree.write(filepath, xml_declaration=True, encoding='UTF-8')
    
    return modified

for xf in xml_files:
    process_xml(xf, data)

# Verify no remaining placeholders
for xf in xml_files:
    with open(xf, 'r', encoding='utf-8') as f:
        content = f.read()
    remaining = re.findall(r'\{\{[^}]+\}\}', content)
    if remaining:
        print(f'WARNING: Remaining placeholders in {xf}: {remaining}')

# Repackage as HWPX (ZIP)
output_path = '/root/event_announcement_ready.hwpx'
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root_dir, dirs, files in os.walk(work_dir):
        for fname in files:
            full_path = os.path.join(root_dir, fname)
            arcname = os.path.relpath(full_path, work_dir)
            zout.write(full_path, arcname)

print(f'Output written to {output_path}')

# Final verification
with zipfile.ZipFile(output_path, 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            remaining = re.findall(r'\{\{[^}]+\}\}', content)
            if remaining:
                print(f'FAIL: Remaining placeholders in {name}: {remaining}')
            else:
                print(f'OK: {name}')
```

### 5. Key requirements to verify
- **No remaining `{{...}}` placeholders** in any XML file in the output archive.
- **Korean labels and static note lines are unchanged** — only placeholder text is replaced.
- **Layout cache removal**: Any `<hp:lineSegArray>` (or element with localname `lineSegArray`) in modified paragraphs must be removed to prevent overlapping character rendering.
- **Valid HWPX package**: The output must be a valid ZIP file with the same directory structure as the original.
- **Output path**: `/root/event_announcement_ready.hwpx`

### 6. Post-verification
After writing the output, run the verification script if one exists (e.g., `python test_output.py` or similar). Check for any test files in the workspace and run them.

### Important Notes
- Before running the Python script, first manually inspect `event_data.json` and at least one section XML to understand the exact placeholder keys and XML structure.
- If the `lxml` library is not available, fall back to `xml.etree.ElementTree` (adjust namespace handling accordingly).
- When repackaging, preserve the original file structure exactly. Do not add extra directories or change paths.
- If `mimetype` file exists in the archive, it should ideally be stored uncompressed (first entry), matching OPC conventions. Check if the original has this pattern and replicate it.

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