# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in the HWPX template with values from the JSON data file.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls /root/
ls /root/HWPX-Document-Automation/hwpx-inventory-report/
```
Identify all provided files: the template HWPX, the JSON data, and the test/verifier script.

### 2. Read the test/verifier script
```bash
cat /root/HWPX-Document-Automation/hwpx-inventory-report/test_output.py
```
Understand exactly what the verifier checks:
- All placeholders replaced
- Korean labels preserved
- Empty paragraphs preserved
- No `{{...}}` remaining
- Valid HWPX zip package
- **Modified paragraphs must NOT contain `hp:linesegarray` elements** (uses `paragraph.find('hp:linesegarray', NS)` with namespace dict)

Note the namespace dict `NS` used in the test so you know the exact namespace URI and prefix for `linesegarray`.

### 3. Read the JSON data
```bash
cat /root/HWPX-Document-Automation/hwpx-inventory-report/inventory_data.json
```

### 4. Examine the HWPX template structure
```bash
cd /root/HWPX-Document-Automation/hwpx-inventory-report/
python3 -c "
import zipfile, sys
zf = zipfile.ZipFile('inventory_report_template.hwpx', 'r')
for name in zf.namelist():
    print(name)
zf.close()
"
```

### 5. Extract and inspect section XML files for placeholders
For each XML file in the HWPX (especially `Contents/section0.xml` and any other section files):
```bash
python3 -c "
import zipfile
zf = zipfile.ZipFile('inventory_report_template.hwpx', 'r')
for name in zf.namelist():
    data = zf.read(name)
    if b'{{' in data or b'section' in name.encode().lower():
        print(f'=== {name} ===')
        print(data.decode('utf-8', errors='replace')[:5000])
        print('...')
zf.close()
"
```
Also search all files for `{{` to find every placeholder location.

### 6. Write the automation script
Create `/root/solve.py` that:

a) Reads `inventory_data.json` into a dict.

b) Opens the template HWPX as a zip, iterates over all entries.

c) For each XML file (especially section XML files):
   - Parse with `xml.etree.ElementTree` (register all namespaces first to preserve them on output).
   - Find all text nodes containing `{{...}}` patterns.
   - Replace each `{{key}}` with the corresponding value from the JSON (convert numbers to strings).
   - **Critical: For every `<hp:p>` paragraph element that was modified (i.e., contained a placeholder), find and REMOVE the `hp:linesegarray` child element entirely.** Use the namespace-aware `find()` with the correct namespace URI (extract from the XML root or from the test's NS dict). Also remove `linesegarray` elements using `element.remove()` on the parent, not regex.
   - Also check for `<hp:lineSegArray>` with any namespace variation.

d) Re-serialize the XML preserving the original namespace declarations. Use `ET.register_namespace()` for all namespaces found in the document before parsing.

e) For non-XML files, copy them as-is.

f) Write the result to `/root/inventory_report_ready.hwpx` as a valid zip.

### 7. Namespace handling — CRITICAL
Before parsing any XML, register all namespaces to prevent `ns0:` prefix rewriting:
```python
import xml.etree.ElementTree as ET
import re

def register_all_namespaces(xml_content):
    """Register all namespaces found in XML content."""
    for match in re.finditer(r'xmlns:(\w+)="([^"]+)"', xml_content):
        prefix, uri = match.group(1), match.group(2)
        ET.register_namespace(prefix, uri)
    # Also handle default namespace
    for match in re.finditer(r'xmlns="([^"]+)"', xml_content):
        ET.register_namespace('', match.group(1))
```

### 8. Layout cache removal — CRITICAL (this was the previous failure)
Do NOT use regex on raw XML strings. Use proper ElementTree operations:
```python
# After modifying text in a paragraph, remove linesegarray
# Find the namespace URI for 'hp' prefix from the XML
# Then: 
for lineseg in paragraph.findall('{namespace_uri}linesegarray'):
    paragraph.remove(lineseg)
# Also try case variations:
for lineseg in paragraph.findall('{namespace_uri}lineSegArray'):
    paragraph.remove(lineseg)
```
Make sure to handle BOTH `linesegarray` and `lineSegArray` tag name cases. Check the actual tag names in the XML by inspecting the parsed elements.

Alternatively, iterate over all children of the paragraph and remove any whose local tag name (case-insensitive) is `linesegarray`:
```python
for child in list(paragraph):
    local_tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
    if local_tag.lower() == 'linesegarray':
        paragraph.remove(child)
```
This is the safest approach.

### 9. Placeholder replacement strategy
Placeholders may be split across multiple `<hp:t>` runs within a single paragraph. Handle both cases:
- Simple case: entire `{{key}}` in one text node
- Split case: `{{` in one run, `key}}` in another

For the split case, concatenate all text content of `<hp:t>` elements in the paragraph, do replacements on the concatenated string, then redistribute or place all text in the first `<hp:t>` and clear the rest. However, first check if this is actually needed by inspecting the template XML.

### 10. Run the script
```bash
cd /root/HWPX-Document-Automation/hwpx-inventory-report/
python3 /root/solve.py
```

### 11. Validate
```bash
# Check it's a valid zip
python3 -c "import zipfile; print(zipfile.is_zipfile('/root/inventory_report_ready.hwpx'))"

# Check no placeholders remain
python3 -c "
import zipfile
zf = zipfile.ZipFile('/root/inventory_report_ready.hwpx', 'r')
for name in zf.namelist():
    data = zf.read(name)
    if b'{{' in data:
        print(f'PLACEHOLDER FOUND in {name}:', data.decode('utf-8', errors='replace'))
print('Validation complete')
zf.close()
"

# Check linesegarray removal in modified paragraphs
python3 -c "
import zipfile, xml.etree.ElementTree as ET
zf = zipfile.ZipFile('/root/inventory_report_ready.hwpx', 'r')
for name in zf.namelist():
    if 'section' in name and name.endswith('.xml'):
        data = zf.read(name).decode('utf-8')
        root = ET.fromstring(data)
        # Find all linesegarray elements
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            if tag.lower() == 'linesegarray':
                print(f'WARNING: linesegarray found in {name}')
                # Check if parent paragraph was modified
print('linesegarray check complete')
zf.close()
"
```

### 12. Run the official test
```bash
cd /root/HWPX-Document-Automation/hwpx-inventory-report/
python3 -m pytest test_output.py -v
```

### 13. If tests fail, debug
- Read the exact assertion error
- Re-inspect the output HWPX XML
- Fix and re-run

## Key pitfalls to avoid (from feedback)
1. **Do NOT use regex to remove linesegarray** — use proper XML parsing with `element.remove()`
2. **Handle namespace-qualified tag names** — the tag in the XML will be `{uri}linesegarray` or `{uri}lineSegArray`
3. **Remove linesegarray from ALL modified paragraphs** — a paragraph is modified if any of its descendant text nodes contained a placeholder
4. **Preserve empty paragraphs** — do not delete any `<hp:p>` elements
5. **Preserve Korean text** — only replace `{{...}}` patterns, leave everything else intact
6. **Register all namespaces before parsing** to avoid prefix rewriting

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