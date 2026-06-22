# Task Instruction

You must fill in the HWPX training-feedback template and save the result. Follow these steps precisely:

## 1. Inspect the workspace
```bash
cd /root
ls -la
find . -name 'training_feedback_template.hwpx' -o -name 'training_feedback.json' 2>/dev/null
```

## 2. Read the JSON data
```bash
cat training_feedback.json
```
Note every key-value pair. You will need all of them.

## 3. Explore the HWPX template structure
HWPX is a ZIP package. Extract it to inspect:
```bash
mkdir -p /tmp/hwpx_work
cp training_feedback_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
python3 -c "
import zipfile
with zipfile.ZipFile('template.zip','r') as z:
    for info in z.infolist():
        print(info.filename, info.compress_type, info.file_size)
"
```

## 4. Find and read all XML files containing `{{` placeholders
```bash
python3 -c "
import zipfile, re
with zipfile.ZipFile('template.zip','r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8')
            if '{{' in data:
                print(f'=== {name} ===')
                print(data[:8000])
                print('...')
        except: pass
"
```
Identify every `{{placeholder}}` and which XML file(s) contain them. There may be multiple section XML files (e.g., section0.xml, section1.xml, etc.).

## 5. Build the replacement script
Write a single Python 3 script `/tmp/hwpx_work/fill.py` that does the following:

### 5a. Load JSON values
```python
import json, zipfile, re, os, shutil

with open('/root/training_feedback.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

### 5b. Build the replacement map
Create a dict mapping each `{{key}}` string to its replacement value. Apply these transformations:
- **참석자수**: Convert to digits only. E.g., if JSON has `"32명"`, replace `{{참석자수}}` with `"32"`. If it's already a number, use the digits.
- **만족도**: Rewrite as `"X.X점 (5.0점 만점)"` style. Extract the numeric score from the JSON value and format it. E.g., if JSON has `4.5` or `"4.5"` or `"4.5/5.0"`, produce `"4.5점 (5.0점 만점)"`.
- **종합의견** (or whatever key maps to the overall-opinion placeholder): Append ` 후속 심화반 검토 요망.` after the JSON-provided comment. Make sure there is exactly one space before `후속`.
- All other keys: direct substitution.

### 5c. Process every file in the ZIP
- Open the template HWPX as a ZIP.
- For each entry, if it's an XML file that contains any `{{...}}` pattern:
  - Perform all replacements.
  - **Remove all `<hp:lineSegArray>...</hp:lineSegArray>` elements** (or the equivalent with the actual namespace prefix used in the file) from any paragraph whose text was modified. Use regex or XML parsing. The safest approach: after performing text replacements, remove ALL `<(\w+):lineSegArray[^>]*>.*?</(\w+):lineSegArray>` elements from the modified XML files (using `re.DOTALL`). This prevents stale layout cache from causing overlapping text.
  - Verify no `{{` remains in the processed text.
- Write all entries to the output ZIP.

### 5d. Respect mimetype packaging
- The `mimetype` file (if present) must be the FIRST entry and stored uncompressed (`ZIP_STORED`, compression_type=0).
- All other files use `ZIP_DEFLATED`.

### 5e. Write output
Save to `/root/training_feedback_ready.hwpx`.

### 5f. Validate
After writing:
1. Open the output as a ZIP and verify it's valid.
2. Read all XML files and confirm NO `{{` pattern remains anywhere.
3. Verify the specific transformed values appear in the XML:
   - The digits-only 참석자수 value
   - The `점 (5.0점 만점)` formatted 만족도
   - The `후속 심화반 검토 요망.` suffix in the opinion text
4. Print all paragraph text content to verify correctness.

## 6. Run the script
```bash
cd /tmp/hwpx_work
python3 fill.py
```

## 7. Final verification
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_ready.hwpx','r') as z:
    print('Valid ZIP, entries:', len(z.namelist()))
    for name in z.namelist():
        data = z.read(name)
        try:
            text = data.decode('utf-8')
            if '{{' in text:
                print(f'ERROR: placeholder remains in {name}')
        except: pass
    print('Verification complete')
"
```

## Critical reminders
- Read the actual XML content carefully before writing replacements. Placeholders may be split across XML tags (e.g., `<t>{{</t><t>key</t><t>}}</t>`). If so, you must handle this by working at the raw text level or by first collapsing run elements.
- Korean labels and the static note line must remain unchanged.
- Do NOT hardcode placeholder names without first reading the template. Discover them from the actual XML.
- If any step fails, inspect the error, adjust, and retry.

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