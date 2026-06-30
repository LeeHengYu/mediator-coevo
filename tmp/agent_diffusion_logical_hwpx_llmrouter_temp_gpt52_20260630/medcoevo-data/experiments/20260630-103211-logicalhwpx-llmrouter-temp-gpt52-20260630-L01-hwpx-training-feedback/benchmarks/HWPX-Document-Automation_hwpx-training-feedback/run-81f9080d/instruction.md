# Task Instruction

## Task: Fill in the training feedback HWPX template

### Goal
Replace all `{{...}}` placeholders in `training_feedback_template.hwpx` with values from `training_feedback.json`, apply the required transformations, and save the result to `/root/training_feedback_ready.hwpx`.

### Step-by-step plan

#### 1. Understand the HWPX package structure
- A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML).
- Copy the template to the output path first, then work on it in-place as a ZIP.
```bash
cp /root/training_feedback_template.hwpx /root/training_feedback_ready.hwpx
```
- List the ZIP contents:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx','r'); print('\n'.join(z.namelist()))"
```

#### 2. Inspect the JSON data
```bash
cat /root/training_feedback.json
```
Note every key-value pair. You will need to map each `{{key}}` placeholder to its value.

#### 3. Inspect every XML file in the ZIP for `{{` placeholders
For each XML file in the archive, read its content and search for `{{`. Print the filename and matching lines. This is critical — placeholders may appear in multiple XML files (e.g., `Contents/section0.xml`, `Contents/content.hpf`, etc.).
```python
import zipfile, re
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r')
for name in z.namelist():
    try:
        data = z.read(name).decode('utf-8')
        if '{{' in data:
            print(f'=== {name} ===')
            for i, line in enumerate(data.split('\n')):
                if '{{' in line:
                    print(f'  line {i}: {line[:500]}')
    except: pass
```
Record every placeholder found (e.g., `{{교육명}}`, `{{참석자수}}`, `{{만족도}}`, `{{종합의견}}`, etc.).

#### 4. Build the replacement map with transformations
Load the JSON and build a Python dict mapping each `{{key}}` string to its replacement value, applying these rules:

- **`참석자수`**: Extract digits only. E.g., if JSON has `"참석자수": "32명"`, replace with `"32"`. Use `re.sub(r'[^0-9]', '', value)`.
- **`만족도`**: Rewrite as `"X.X점 (5.0점 만점)"` style. E.g., if JSON has `"만족도": 4.5` or `"만족도": "4.5"`, output `"4.5점 (5.0점 만점)"`. Extract the numeric score from the JSON value.
- **`종합의견`** (or whatever the overall-opinion key is): Append ` 후속 심화반 검토 요망.` after the JSON value. Make sure there's a space before the appended sentence if the original doesn't end with one.
- **All other keys**: Use the JSON value as-is.

#### 5. Perform replacements in all XML files
For every file in the ZIP that contains `{{`:
- Read the XML content as a UTF-8 string.
- For each `{{key}}` placeholder, replace it with the transformed value.
- After all replacements, verify NO `{{` remains in the content. If any `{{...}}` pattern remains, that's a bug — investigate and fix.

#### 6. Remove stale layout-cache elements from modified paragraphs
This is critical for the document to render cleanly. In HWPX XML, layout cache elements are typically `<hp:linesegarray>` or `<lineseg>` or similar elements that cache glyph positioning. After modifying paragraph text:

- Parse the XML properly (use `xml.etree.ElementTree` or `lxml`).
- For any `<hp:p>` (paragraph) element whose text content was modified (i.e., contained a placeholder), find and **remove** any child elements related to layout caching. Common tag names include:
  - `hp:linesegarray` / `linesegarray`
  - Elements with tag containing `lineseg`
  - `hp:lineBreak` cache data
- Use namespace-aware searching. First inspect the actual XML to identify the exact tag names and namespaces used for layout cache elements.
- Remove these elements entirely from modified paragraphs.

#### 7. Repack the ZIP
Write the modified XML files back into the ZIP archive. Use Python's `zipfile` module:
```python
import zipfile, os, shutil

# Read all files from original
original = zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r')
files = {}
for name in original.namelist():
    files[name] = original.read(name)
original.close()

# Update modified XML files in the dict
# files['Contents/section0.xml'] = modified_xml_bytes  # etc.

# Rewrite the ZIP
with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'w', zipfile.ZIP_DEFLATED) as zout:
    for name, data in files.items():
        zout.writestr(name, data)
```

#### 8. Validate the output
Run these checks:
1. **ZIP validity**: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx','r'); print('Valid ZIP, entries:', len(z.namelist()))"`
2. **No remaining placeholders**: Search all XML files for `{{` — must find zero.
3. **Verify transformed values appear**: Search for the digit-only attendee count, the `점 (5.0점 만점)` string, and `후속 심화반 검토 요망.` in the XML content.
4. **Verify Korean labels preserved**: Spot-check that static Korean labels from the template still exist.
5. **Verify no stale layout cache in modified paragraphs**: For paragraphs that had placeholders, confirm layout-cache elements are gone.

### Key warnings
- **Do NOT change any Korean labels or static note lines** — only replace `{{...}}` placeholders.
- **Placeholders may be split across XML tags** (e.g., `<hp:t>{{교육</hp:t><hp:t>명}}</hp:t>`). If you find this, you must handle it. Inspect the raw XML carefully. If placeholders are split, work at the string level on the serialized XML or merge the text runs first.
- **Preserve the exact ZIP structure** — same filenames, same directory layout.
- **Empty paragraphs must be preserved** — do not delete any paragraph elements.
- The output file must be at exactly `/root/training_feedback_ready.hwpx`.
- After every edit step, re-read the file to confirm changes landed correctly.

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