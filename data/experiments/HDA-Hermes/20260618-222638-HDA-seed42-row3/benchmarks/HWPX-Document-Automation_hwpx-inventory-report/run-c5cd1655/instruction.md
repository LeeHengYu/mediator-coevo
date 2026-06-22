# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in the HWPX template with values from the JSON data file.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
cat /root/inventory_data.json
```
Examine the JSON to understand the keys and values available for placeholder replacement.

### 2. Inspect the HWPX template structure
```bash
cd /root
python3 -c "
import zipfile
with zipfile.ZipFile('inventory_report_template.hwpx', 'r') as z:
    for name in z.namelist():
        print(name)
"
```
Identify all files inside the HWPX archive, especially section XML files (e.g., `Contents/section0.xml`, `Contents/section1.xml`, etc.).

### 3. Examine section XML files for placeholders
For each section XML file found, print its contents to identify all `{{...}}` placeholders:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('inventory_report_template.hwpx', 'r') as z:
    for name in z.namelist():
        if 'section' in name and name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            print(f'=== {name} ===')
            print(content[:5000])
            print('...')
"
```
Also search for all placeholder patterns:
```bash
python3 -c "
import zipfile, re
with zipfile.ZipFile('inventory_report_template.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            placeholders = re.findall(r'\{\{[^}]+\}\}', content)
            if placeholders:
                print(f'{name}: {placeholders}')
"
```

### 4. Create and run the replacement script
Write a Python script `/root/process_hwpx.py` that:

a) Loads `inventory_data.json` into a dict.

b) Opens `inventory_report_template.hwpx` as a ZIP.

c) For each file in the ZIP:
   - If it's an XML file containing `{{...}}` placeholders:
     1. Decode the content as UTF-8.
     2. **Important**: Placeholders like `{{key}}` may be split across multiple XML text nodes (e.g., `<hp:t>{{ke</hp:t><hp:t>y}}</hp:t>`). To handle this robustly:
        - First, parse the XML with `lxml.etree` (preferred) or `xml.etree.ElementTree`.
        - For each paragraph element, concatenate all text content from child `<hp:t>` (or equivalent text) elements to reconstruct the full text.
        - Replace all `{{key}}` patterns in the concatenated text with the corresponding JSON values. Handle nested JSON (e.g., if a value is a dict or list, flatten appropriately; but typically values are strings or numbers).
        - Redistribute the replaced text back into the text elements (simplest: put all text in the first `<hp:t>` element and clear the rest, or replace the full paragraph text content).
     3. **Critical**: For any paragraph whose text was modified, remove all layout-cache child elements. These are typically elements like `<hp:lineSegArray>`, `<hp:lineseg>`, or similar layout/cache elements. Remove them so the HWP viewer re-renders text correctly without overlapping characters.
     4. Serialize the modified XML back to a UTF-8 string.
   - If it's not modified, keep the original bytes.

d) Write all files to `/root/inventory_report_ready.hwpx` as a new ZIP, preserving the original compression type for each entry.

e) **Validation**: After writing, re-open the output HWPX and verify:
   - No `{{...}}` patterns remain in any XML file.
   - The file is a valid ZIP.
   - Print confirmation.

**Key implementation details:**
- Handle XML namespaces correctly. The HWPX format uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph` etc. Use namespace-aware parsing.
- When searching for text elements, look for tags ending in `}t` or use the appropriate namespace prefix.
- For placeholder matching against JSON keys: strip the `{{` and `}}` and look up the key in the JSON dict. If the JSON has nested structure, also support dot-notation or flatten as needed based on what you observe in the placeholders.
- Preserve all Korean text, static notes, and empty paragraphs. Only modify text nodes that contain `{{...}}` patterns.
- Convert non-string JSON values (numbers, booleans) to strings before insertion.

### 5. Run the script
```bash
cd /root && python3 process_hwpx.py
```

### 6. Validate the output
```bash
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/inventory_report_ready.hwpx', 'r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            content = z.read(name).decode('utf-8')
            placeholders = re.findall(r'\{\{[^}]+\}\}', content)
            if placeholders:
                print(f'FAIL: {name} still has placeholders: {placeholders}')
            else:
                print(f'OK: {name}')
print('Validation complete')
"
```

Also verify the file exists and is a valid ZIP:
```bash
python3 -c "import zipfile; print('Valid ZIP:', zipfile.is_zipfile('/root/inventory_report_ready.hwpx'))"
ls -la /root/inventory_report_ready.hwpx
```

### 7. Run any available tests
```bash
cd /root && find . -name 'test_*.py' -o -name '*_test.py' | head -5
# If tests exist, run them:
# python3 -m pytest tests/ -v
```
Run any test files found to confirm the output passes verification.

## Important Reminders
- Do NOT leave any `{{...}}` placeholder in the output.
- Do NOT alter Korean labels or static note lines.
- Do NOT remove empty paragraphs.
- DO remove layout-cache elements (lineSegArray, lineseg, etc.) from any paragraph you modify.
- The output MUST be at `/root/inventory_report_ready.hwpx` and must be a valid HWPX (ZIP) package.

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