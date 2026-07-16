# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in the HWPX template with values from the JSON data file, then save the result as a valid HWPX package.

## Step-by-step Plan

### 1. Inspect the workspace
```bash
ls /root/
ls /root/HWPX-Document-Automation/hwpx-inventory-report/
```
Identify the template file (`inventory_report_template.hwpx`) and the data file (`inventory_data.json`).

### 2. Read the JSON data
```bash
cat /root/HWPX-Document-Automation/hwpx-inventory-report/inventory_data.json
```
Note every key-value pair. These keys will correspond to `{{key}}` placeholders in the template.

### 3. Explore the HWPX package structure
HWPX files are OPC/ZIP-based packages. Unzip the template to a temporary directory:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip /root/HWPX-Document-Automation/hwpx-inventory-report/inventory_report_template.hwpx -d template_extracted
find template_extracted -type f
```

### 4. Identify all files containing placeholders
Search for `{{` across all extracted files:
```bash
grep -rl '{{' template_extracted/
```
The primary content is likely in `Contents/section0.xml`, but check all files.

### 5. Examine the XML content
For each file containing `{{`, read it fully:
```bash
cat template_extracted/Contents/section0.xml
```
Note: Placeholders may be split across XML tags (e.g., `{{` in one text run and `}}` in another). Check carefully.

### 6. Write a Python script to perform the replacement
Create a Python script that:
1. Reads the JSON data file.
2. Extracts the HWPX ZIP to a temp directory.
3. For each file in the extracted directory, reads the content and replaces all `{{key}}` patterns with corresponding JSON values.
4. **Handles split placeholders**: If placeholders are split across multiple XML `<hp:t>` or similar text elements within the same paragraph/run, join the text content first, perform replacement, then write back. A robust approach: for each XML file containing `{{`, use regex on the raw XML text to find and replace `{{key}}` patterns. Since placeholders may span across XML tags within a run, consider collapsing adjacent text elements in the same `<hp:run>` before replacement.
5. **Removes stale layout cache elements**: After replacement, remove all `<hp:linesegarray>...</hp:linesegarray>` elements (and any similar layout cache like `<lineseg .../>` blocks) from any modified XML file. This ensures the document re-renders cleanly without overlapping characters.
6. Preserves all other content (Korean labels, static note lines, empty paragraphs) exactly as-is.
7. Re-packages the modified files back into a ZIP with `.hwpx` extension at `/root/inventory_report_ready.hwpx`, preserving the original directory structure and using the same compression method.

```python
import json, os, re, shutil, zipfile

# Paths
template_path = '/root/HWPX-Document-Automation/hwpx-inventory-report/inventory_report_template.hwpx'
json_path = '/root/HWPX-Document-Automation/hwpx-inventory-report/inventory_data.json'
output_path = '/root/inventory_report_ready.hwpx'
extract_dir = '/tmp/hwpx_inventory_extracted'

# Clean up any previous extraction
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)

# Load JSON data
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# If data is a list of dicts or nested, flatten appropriately
# Inspect structure first
print('JSON data type:', type(data))
print('JSON data:', json.dumps(data, ensure_ascii=False, indent=2))

# Extract HWPX
with zipfile.ZipFile(template_path, 'r') as z:
    z.extractall(extract_dir)

# Find all files with placeholders
placeholder_files = []
for root, dirs, files in os.walk(extract_dir):
    for fname in files:
        fpath = os.path.join(root, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                content = f.read()
            if '{{' in content:
                placeholder_files.append(fpath)
        except (UnicodeDecodeError, IsADirectoryError):
            pass

print('Files with placeholders:', placeholder_files)

# Build replacement map
# Handle both flat dict and nested structures
def build_replacements(data, prefix=''):
    replacements = {}
    if isinstance(data, dict):
        for k, v in data.items():
            key = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
            if isinstance(v, (dict, list)):
                replacements.update(build_replacements(v, key))
            else:
                replacements[k] = str(v)
                if prefix:
                    replacements[f'{prefix}.{k}'] = str(v)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            replacements.update(build_replacements(item, f'{prefix}[{i}]' if prefix else f'[{i}]'))
    return replacements

# First try flat keys
if isinstance(data, dict):
    flat_replacements = {k: str(v) for k, v in data.items() if not isinstance(v, (dict, list))}
    # Also handle nested
    flat_replacements.update(build_replacements(data))
else:
    flat_replacements = build_replacements(data)

print('Replacement keys:', list(flat_replacements.keys()))

# Process each file
for fpath in placeholder_files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if placeholders might be split across XML tags
    # Strategy: find {{...}} patterns that may span tags
    # First, try direct replacement
    for key, value in flat_replacements.items():
        content = content.replace('{{' + key + '}}', value)
    
    # If any {{ remain, try to handle split placeholders
    if '{{' in content:
        print(f'WARNING: Remaining placeholders in {fpath} after direct replacement')
        # Try removing XML tags between {{ and }}
        # Pattern: {{ possibly with XML tags interspersed before }}
        def fix_split_placeholders(text):
            # Remove XML tags that break up placeholder text within hp:t or similar elements
            # Find all {{ ... }} where ... might contain XML tags
            pattern = r'(\{\{)((?:(?!\}\}).)*?)(\}\})'
            def clean_match(m):
                inner = re.sub(r'</[^>]+>[^<]*<[^>]+>', '', m.group(2))
                full_key = inner.strip()
                if full_key in flat_replacements:
                    return flat_replacements[full_key]
                return m.group(0)
            return re.sub(pattern, clean_match, text, flags=re.DOTALL)
        
        content = fix_split_placeholders(content)
        
        # Show remaining placeholders for debugging
        remaining = re.findall(r'\{\{.*?\}\}', content, re.DOTALL)
        if remaining:
            print(f'Still remaining: {remaining}')
    
    # Remove stale layout cache elements (linesegarray)
    content = re.sub(r'<hp:linesegarray>.*?</hp:linesegarray>', '', content, flags=re.DOTALL)
    # Also remove standalone lineseg variants
    content = re.sub(r'<linesegarray>.*?</linesegarray>', '', content, flags=re.DOTALL)
    
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(content)

# Re-package as HWPX
if os.path.exists(output_path):
    os.remove(output_path)

# Preserve original ZIP structure
with zipfile.ZipFile(template_path, 'r') as orig_zip:
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as out_zip:
        for item in orig_zip.infolist():
            file_path = os.path.join(extract_dir, item.filename)
            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    out_zip.writestr(item, f.read())
            elif os.path.isdir(file_path):
                out_zip.writestr(item, b'')

print('Output written to', output_path)
```

Run this script. Examine the printed output carefully.

### 7. Validate the output
After the script runs:

1. **Verify no remaining placeholders**:
```bash
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify && rm -rf * && unzip /root/inventory_report_ready.hwpx
grep -r '{{' . || echo 'No placeholders remaining - GOOD'
```

2. **Verify the file is a valid ZIP**:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx','r'); print('Valid ZIP, entries:', len(z.namelist())); z.testzip(); print('No corruption')"
```

3. **Verify Korean labels and structure are preserved** by examining the section XML:
```bash
cat /tmp/hwpx_verify/Contents/section0.xml | head -200
```
Confirm Korean text is intact, empty paragraphs are preserved, and replaced values appear correctly.

4. **Verify linesegarray elements are removed from modified paragraphs**:
```bash
grep -c 'linesegarray' /tmp/hwpx_verify/Contents/section0.xml && echo 'WARNING: linesegarray still present' || echo 'linesegarray removed - GOOD'
```

5. **Confirm the output file exists at the correct path**:
```bash
ls -la /root/inventory_report_ready.hwpx
```

### 8. Handle edge cases
- If the JSON data is nested (e.g., has arrays for inventory items), inspect the XML structure to understand how rows are templated and adjust the replacement logic accordingly. The template may have repeated row patterns with indexed placeholders like `{{item_1_name}}` or `{{items[0].name}}`.
- If placeholders span multiple XML elements (split by tags), the script's fallback handler should address this. If not, manually inspect the raw XML around any remaining `{{` to understand the split pattern and fix accordingly.
- If the JSON has numeric values, ensure they are converted to strings for replacement.

### Critical Reminders
- Do NOT modify Korean text labels or the static note line.
- Do NOT remove empty `<hp:p>` paragraphs (they serve as spacing).
- DO remove `<hp:linesegarray>` blocks from any paragraph whose text was modified.
- The final file MUST be at `/root/inventory_report_ready.hwpx`.
- The final file MUST be a valid ZIP/HWPX package with no remaining `{{...}}` placeholders.

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