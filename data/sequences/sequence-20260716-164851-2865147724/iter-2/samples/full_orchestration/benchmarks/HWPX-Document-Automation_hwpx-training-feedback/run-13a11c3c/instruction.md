# Task Instruction

## Task: Fill in training_feedback_template.hwpx and save to /root/training_feedback_ready.hwpx

### Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). The text content is typically in XML files under a `Contents/` directory (e.g., `section0.xml`, `section1.xml`, etc.).

### Step 2: Inspect the workspace
```bash
ls -la /root/
find /root/ -name '*.hwpx' -o -name '*.json' 2>/dev/null
```
Locate `training_feedback_template.hwpx` and `training_feedback.json`. They may be in the current working directory or a subdirectory.

### Step 3: Read the JSON data
```bash
cat training_feedback.json
```
Note every key-value pair. Pay special attention to:
- `참석자수` — must be converted to digits only (e.g., '25명' → '25')
- `만족도` — must be reformatted as `X.X점 (5.0점 만점)` using the numeric score from JSON
- The overall opinion/comment field — must have `후속 심화반 검토 요망.` appended after the provided comment

### Step 4: Explore the HWPX template structure
```bash
mkdir -p /tmp/hwpx_work
cp training_feedback_template.hwpx /tmp/hwpx_work/
cd /tmp/hwpx_work
python3 -c "
import zipfile
with zipfile.ZipFile('training_feedback_template.hwpx', 'r') as z:
    for name in z.namelist():
        print(name)
"
```
Then extract and examine ALL XML files that contain `{{` placeholders:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('training_feedback_template.hwpx', 'r') as z:
    for name in z.namelist():
        data = z.read(name)
        try:
            text = data.decode('utf-8')
        except:
            continue
        if '{{' in text:
            print(f'=== {name} ===')
            print(text[:5000])
            print('...')
"
```

### Step 5: Identify all placeholders
For each XML file containing `{{...}}`, list every unique placeholder. Map each to the corresponding JSON key.

### Step 6: Perform replacements with a Python script
Write a Python script that:
1. Opens the template HWPX as a ZIP
2. For each file in the ZIP:
   a. If the file contains `{{` placeholders, perform substitutions
   b. Apply the special transformations:
      - `참석자수`: extract digits only from the JSON value (e.g., strip '명' or any non-digit characters)
      - `만족도`: reformat as `{score}점 (5.0점 만점)` where score is the numeric value from JSON
      - Overall opinion/comment: append ` 후속 심화반 검토 요망.` after the JSON value (ensure proper spacing — if the JSON comment doesn't end with a space, add one before appending; if it ends with a period, just add a space then append)
   c. **Remove stale layout-cache elements**: For any paragraph (`<hp:p>`) whose text content was modified, remove `<hp:linesegarray>` elements (or similar layout cache tags like `<lineseg>`, `<hp:lineSegArray>`) within that paragraph. These are pre-computed layout caches that cause overlapping characters when text length changes. Inspect the actual XML tag names used — they may be `hp:linesegarray`, `hp:lineSegArray`, `linesegarray`, etc.
3. Write the result as a new ZIP to `/root/training_feedback_ready.hwpx`, preserving the same compression method for each entry.

### Step 7: Validate the output
```bash
# Check it's a valid ZIP
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r') as z:
    print('Valid ZIP, entries:', len(z.namelist()))
"

# Check no {{...}} placeholders remain
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r') as z:
    for name in z.namelist():
        data = z.read(name)
        try:
            text = data.decode('utf-8')
        except:
            continue
        matches = re.findall(r'\{\{.*?\}\}', text)
        if matches:
            print(f'REMAINING PLACEHOLDERS in {name}: {matches}')
print('Placeholder check complete')
"

# Verify specific values are present
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r') as z:
    for name in z.namelist():
        data = z.read(name)
        try:
            text = data.decode('utf-8')
        except:
            continue
        if '점 (5.0점 만점)' in text:
            print(f'만족도 format found in {name}')
        if '후속 심화반 검토 요망' in text:
            print(f'Appended opinion found in {name}')
"
```

### Step 8: Run the verifier if available
```bash
# Check for test files
find /root/ -name 'test_*.py' -o -name '*test*.py' -o -name 'verify*' 2>/dev/null
# If found, run:
# cd <appropriate_dir> && python3 -m pytest -xvs
```

### Key Warnings (from cross-task failures):
- **Exact string matching matters**: The verifier likely checks for exact substrings. Ensure `만족도` is formatted precisely as `X.X점 (5.0점 만점)` with no extra spaces.
- **All sections must be processed**: Check ALL XML files in the HWPX, not just section0.xml. There may be section1.xml or others with placeholders.
- **Layout cache removal is critical**: Any `<hp:linesegarray>` or similar elements in modified paragraphs must be stripped to avoid rendering issues. Inspect the actual tag names in the XML before writing removal logic.
- **Korean labels and static note lines must be preserved exactly** — only replace placeholder values, not surrounding text.
- **Digits-only for 참석자수**: Use `re.sub(r'[^0-9]', '', value)` or similar to extract only digits.

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