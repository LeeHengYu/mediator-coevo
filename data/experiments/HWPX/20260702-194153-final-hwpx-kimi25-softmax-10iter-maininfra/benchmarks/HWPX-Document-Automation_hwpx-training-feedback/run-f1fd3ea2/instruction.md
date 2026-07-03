# Task Instruction

Execute the following steps in order to fill in the training feedback HWPX template and save the result.

## Step 0 – Inspect the workspace
```bash
find /root -maxdepth 2 -type f | head -60
```
Identify the paths of `training_feedback_template.hwpx` and `training_feedback.json`.

## Step 1 – Read the JSON data
```bash
cat <path-to>/training_feedback.json
```
Note every key-value pair. You will need them all for placeholder replacement.

## Step 2 – Explore the HWPX package
A `.hwpx` file is a ZIP archive. Unzip the template into a working directory:
```bash
mkdir -p /tmp/hwpx_work
cp <path-to>/training_feedback_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d hwpx_contents
find hwpx_contents -type f
```
Identify all XML files, especially any `section*.xml` files under `Contents/`.

## Step 3 – Inspect section XML files
For every `section*.xml` found, print its full contents:
```bash
cat hwpx_contents/Contents/section0.xml
```
(Repeat for section1.xml, etc., if they exist.)

Study the XML carefully:
- Note the namespace (likely `http://www.hancom.co.kr/hwpml/2010/HWPML` or similar).
- Locate every `{{...}}` placeholder. They may be **fragmented across multiple `<hp:t>` elements** within a single `<hp:run>` or `<hp:p>`. This is critical.
- Note the presence of `<hp:linesegarray>` (or `<hp:lineSegArray>`) elements inside `<hp:p>` tags — these are layout caches.

## Step 4 – Write and run a Python script
Create `/tmp/hwpx_work/fill_template.py` with the following logic:

### 4a – Load JSON data
```python
import json, zipfile, os, re, shutil
from lxml import etree

with open('<path-to>/training_feedback.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

### 4b – Build replacement map
Build a dict mapping each placeholder key to its replacement value. Apply the following transformations:

1. **`참석자수`**: Convert to digits only. E.g., if JSON has `"25명"` → use `"25"`. If it already is a number, convert to string of digits. Strip any non-digit characters.
2. **`만족도`**: Rewrite as `"X.X점 (5.0점 만점)"` style, where X.X is the numeric score from the JSON. E.g., if JSON has `4.5` or `"4.5"` → `"4.5점 (5.0점 만점)"`.
3. **Overall opinion / comment field** (look for the key that corresponds to the final opinion/comment): Append ` 후속 심화반 검토 요망.` after the provided comment value. Make sure there is exactly one space before the appended sentence if the original doesn't end with a space.
4. All other values: use as-is from JSON.

### 4c – Placeholder fragmentation repair (CRITICAL)
For each section XML:
1. Parse with lxml, preserving the namespace.
2. For every `<hp:p>` element, collect all `<hp:run>` children.
3. Within each `<hp:p>`, concatenate the text of ALL `<hp:t>` elements (across all `<hp:run>` children) to detect if a `{{...}}` pattern spans multiple `<hp:t>` elements.
4. If fragmentation is detected (i.e., `{{` and `}}` appear in different `<hp:t>` elements, or partial placeholder text like `{{교육` in one `<hp:t>` and `명}}` in another):
   - Merge the text content of consecutive `<hp:t>` elements that together form a complete `{{...}}` placeholder into a single `<hp:t>` element.
   - Remove the now-empty `<hp:t>` elements (and their parent `<hp:run>` if it becomes empty of text).
5. A robust approach: for each `<hp:p>`, gather all `<hp:t>` elements in document order. Concatenate their `.text` values. Use regex to find all `{{...}}` patterns in the concatenated string. Then redistribute the fully-replaced text back, or (simpler) merge all `<hp:t>` text into the first `<hp:t>` and clear/remove the rest, then do replacements on the merged text.

### 4d – Perform replacements
For every `<hp:t>` element in every section XML:
- Replace every `{{key}}` with the corresponding transformed value from the replacement map.
- After all replacements, verify no `{{` or `}}` remains in any `<hp:t>` text.

### 4e – Remove layout cache from modified paragraphs (CRITICAL)
For every `<hp:p>` element where ANY text was modified (i.e., any `<hp:t>` child had a replacement):
- Find and REMOVE all child elements matching the local name `linesegarray` (case-insensitive search, or check both `linesegarray` and `lineSegArray`). Use a namespace-aware approach:
```python
for lsa in p.findall('.//{%s}linesegarray' % ns) + p.findall('.//{%s}lineSegArray' % ns):
    p.remove(lsa)
```
- Also try finding by local name iteration:
```python
for child in list(p):
    if 'lineseg' in child.tag.lower() or 'lineSegArray' in child.tag:
        p.remove(child)
```
This prevents overlapping character rendering artifacts.

### 4f – Write modified XML back
Serialize each modified section XML with `etree.tostring(..., xml_declaration=True, encoding='UTF-8')` and write it back to the unpacked directory.

### 4g – Repackage the HWPX
Re-zip the entire unpacked directory structure into `/root/training_feedback_ready.hwpx`:
```python
output_path = '/root/training_feedback_ready.hwpx'
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk('hwpx_contents'):
        for fname in files:
            full = os.path.join(root, fname)
            arcname = os.path.relpath(full, 'hwpx_contents')
            zout.write(full, arcname)
```
**Important**: If the original ZIP contains a `mimetype` file, it should be stored first and uncompressed (ZIP_STORED), as per ODF/package conventions. Check the original ZIP for this.

## Step 5 – Run the script
```bash
cd /tmp/hwpx_work
python3 fill_template.py
```

## Step 6 – Validate the output
1. Confirm the output file exists:
```bash
ls -la /root/training_feedback_ready.hwpx
```
2. Verify it is a valid ZIP:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); z.testzip(); print('Valid ZIP'); z.close()"
```
3. Extract and check section XMLs for correctness:
```bash
mkdir -p /tmp/verify
cd /tmp/verify
unzip /root/training_feedback_ready.hwpx -d verify_contents
cat verify_contents/Contents/section0.xml
```
4. Verify NO `{{` or `}}` remains:
```bash
grep -r '{{' verify_contents/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'
grep -r '}}' verify_contents/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'
```
5. Verify `참석자수` is digits only (no 명 or other text).
6. Verify `만족도` appears as `X.X점 (5.0점 만점)` format.
7. Verify the overall opinion ends with `후속 심화반 검토 요망.`
8. Verify no `<hp:linesegarray>` or `<hp:lineSegArray>` exists in any paragraph that had text modifications (ideally check that modified paragraphs have no such element).
9. Verify Korean labels and static note lines are unchanged.

## Step 7 – Run any provided test/verifier
If there is a `test_outputs.py` or similar verifier script in the task directory, run it:
```bash
find /root -name 'test_output*' -o -name 'verify*' | head -5
# Run whatever is found
python3 <verifier_script>
```
Fix any failures and re-run until the verifier passes.

## Key Reminders
- **Placeholder fragmentation**: HWPX editors split text across `<hp:t>` elements. You MUST handle this.
- **Layout cache removal**: Every `<hp:p>` with modified text MUST have its `linesegarray`/`lineSegArray` children removed.
- **Exact formatting**: `만족도` must be exactly `X.X점 (5.0점 만점)` with the score from JSON. `참석자수` must be digits only.
- **Appended sentence**: The overall opinion must end with ` 후속 심화반 검토 요망.` (space before if needed).
- **Output path**: Must be exactly `/root/training_feedback_ready.hwpx`.

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