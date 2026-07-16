# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in the HWPX template with values from the JSON data file.

## Step-by-step Plan

### 1. Inspect the workspace
```bash
ls /root/
```
Identify `inventory_report_template.hwpx` and `inventory_data.json`.

### 2. Read the JSON data
```bash
cat /root/inventory_data.json
```
Note every key-value pair. These are the replacements for `{{key}}` placeholders.

### 3. Understand the HWPX structure
HWPX files are OPC-based ZIP packages. List the contents:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_template.hwpx','r'); print('\n'.join(z.namelist()))"
```

### 4. Extract and inspect the main content XML
The primary content is typically in `Contents/section0.xml`. Extract and print it:
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/inventory_report_template.hwpx', 'r')
for name in z.namelist():
    if 'section' in name.lower() and name.endswith('.xml'):
        print(f'--- {name} ---')
        print(z.read(name).decode('utf-8'))
"
```
Also search ALL files in the archive for `{{` to make sure no placeholders hide in other XML files:
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/inventory_report_template.hwpx', 'r')
for name in z.namelist():
    data = z.read(name)
    try:
        text = data.decode('utf-8')
    except:
        continue
    if '{{' in text:
        print(f'Placeholders found in: {name}')
"
```

### 5. Perform the replacement and clean layout caches
Write a Python script that:
1. Opens the template HWPX as a ZIP.
2. Reads `inventory_data.json` into a dict.
3. For every file in the ZIP, if it's a text/XML file containing `{{...}}` placeholders, replace each `{{key}}` with the corresponding JSON value (convert non-string values to strings).
4. **Critical**: After replacing placeholders in any XML content, remove all `<linesegarray>...</linesegarray>` elements (stale layout cache) using regex. This prevents overlapping characters when the document is opened. Use `re.sub(r'<(?:[a-zA-Z0-9_]+:)?linesegarray[^>]*>.*?</(?:[a-zA-Z0-9_]+:)?linesegarray>', '', content, flags=re.DOTALL)` to handle both namespaced and non-namespaced variants.
5. Also remove any `<lineSegArray ...>...</lineSegArray>` variants (check for camelCase too).
6. Write all files (modified and unmodified) into a new ZIP at `/root/inventory_report_ready.hwpx`, preserving the original compression type for each entry.

```python
import zipfile, json, re, os

template_path = '/root/inventory_report_template.hwpx'
data_path = '/root/inventory_data.json'
output_path = '/root/inventory_report_ready.hwpx'

with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Build replacement map: flatten if nested, handle top-level keys
def flatten(d, parent_key=''):
    items = {}
    for k, v in d.items():
        if isinstance(v, dict):
            items.update(flatten(v, k + '.'))
        else:
            items[k] = str(v)
            if parent_key:
                items[parent_key + '.' + k] = str(v)
    return items

replacements = flatten(data) if isinstance(data, dict) else {}
# Also keep top-level simple values
for k, v in data.items():
    if not isinstance(v, dict) and not isinstance(v, list):
        replacements[k] = str(v)

print('Replacement keys:', list(replacements.keys()))

with zipfile.ZipFile(template_path, 'r') as zin:
    with zipfile.ZipFile(output_path, 'w') as zout:
        for item in zin.infolist():
            raw = zin.read(item.filename)
            try:
                content = raw.decode('utf-8')
                is_text = True
            except:
                is_text = False
            
            if is_text and '{{' in content:
                # Replace all {{key}} placeholders
                for key, value in replacements.items():
                    content = content.replace('{{' + key + '}}', value)
                
                # Check for any remaining placeholders and report them
                remaining = re.findall(r'\{\{[^}]+\}\}', content)
                if remaining:
                    print(f'WARNING: remaining placeholders in {item.filename}: {remaining}')
                
                # Remove stale layout cache elements (linesegarray variants)
                content = re.sub(r'<(?:[a-zA-Z0-9_]+:)?[Ll]ine[Ss]eg[Aa]rray[^>]*>.*?</(?:[a-zA-Z0-9_]+:)?[Ll]ine[Ss]eg[Aa]rray>', '', content, flags=re.DOTALL)
                
                raw = content.encode('utf-8')
            
            zout.writestr(item, raw)

print('Output written to', output_path)
```

### 6. Validate the output
After generating the output, run these checks:

**Check 1**: Verify it's a valid ZIP/HWPX:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx','r'); print('Valid ZIP. Files:', len(z.namelist())); z.testzip()"
```

**Check 2**: No remaining `{{...}}` placeholders anywhere:
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/inventory_report_ready.hwpx', 'r')
found = False
for name in z.namelist():
    data = z.read(name)
    try:
        text = data.decode('utf-8')
    except:
        continue
    import re
    matches = re.findall(r'\{\{[^}]+\}\}', text)
    if matches:
        print(f'FAIL: {name} still has placeholders: {matches}')
        found = True
if not found:
    print('PASS: No placeholders remain.')
"
```

**Check 3**: No linesegarray elements remain in modified sections:
```bash
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/inventory_report_ready.hwpx', 'r')
for name in z.namelist():
    if 'section' in name.lower():
        text = z.read(name).decode('utf-8')
        if re.search(r'[Ll]ine[Ss]eg', text):
            print(f'WARNING: lineseg elements still in {name}')
        else:
            print(f'PASS: {name} clean of lineseg cache')
"
```

**Check 4**: Verify the output file exists at the correct path:
```bash
ls -la /root/inventory_report_ready.hwpx
```

### 7. Handle remaining placeholders
If Step 6 Check 2 reveals remaining placeholders, inspect the JSON data structure more carefully. The placeholders might use dot-notation (e.g., `{{items.0.name}}`) or nested keys. Adjust the flattening logic and re-run. Also check if placeholders span across XML tags (e.g., `<t>{{</t><t>key</t><t>}}</t>`) — if so, you need to first concatenate text runs within a paragraph, do the replacement, then put the result in a single text run.

### Important Notes
- Korean labels and static note lines must NOT be altered — only `{{...}}` placeholders get replaced.
- Empty paragraphs (spacing) must be preserved — do not remove any XML paragraph elements.
- The output must be saved exactly to `/root/inventory_report_ready.hwpx`.

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