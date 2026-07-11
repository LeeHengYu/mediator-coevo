# Task Instruction

Complete the following task step by step.

## Goal
Fill in the training feedback sheet `training_feedback_template.hwpx` using the values in `training_feedback.json`, then save the result to `/root/training_feedback_ready.hwpx`.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name 'training_feedback*' -type f
```
Identify the template HWPX file and the JSON file.

### 2. Read the JSON data
```bash
cat /root/training_feedback.json
```
Note every key-value pair. Pay special attention to:
- `참석자수` — must be converted to digits only (e.g., "25명" → "25")
- `만족도` — must be rewritten as `X.X점 (5.0점 만점)` format using the numeric score from JSON
- The overall opinion/comment field — must have `후속 심화반 검토 요망.` appended after the provided comment text

### 3. Examine the HWPX template internals
The .hwpx file is a ZIP archive. List its contents:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_template.hwpx', 'r') as z:
    for name in z.namelist():
        print(name)
"
```
Then read each XML file inside (especially files under `Contents/` like `section0.xml`, `content.hpf`, etc.) to find all `{{...}}` placeholders:
```bash
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/training_feedback_template.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8', errors='replace')
            matches = re.findall(r'\{\{.*?\}\}', data)
            if matches:
                print(f'--- {name} ---')
                for m in matches:
                    print(f'  {m}')
        except:
            pass
"
```
Also print the full XML content of files containing placeholders to understand the structure, especially how placeholders may be split across XML tags.

### 4. Write and run the automation script
Create a Python script that:

a) Reads the JSON file and builds a flat key→value mapping. If the JSON has nested objects, flatten them with dot notation (e.g., `{"a": {"b": "c"}}` → `{"a.b": "c"}`).

b) Applies special value transformations BEFORE substitution:
   - For `참석자수`: extract digits only (e.g., "25명" → "25", or if already a number, convert to string of digits)
   - For `만족도`: reformat as `X.X점 (5.0점 만점)` where X.X is the numeric score from JSON
   - For the overall opinion/comment field: append ` 후속 심화반 검토 요망.` to the end of the value (with a space before if the original doesn't end with one)

c) Opens the template HWPX as a ZIP, processes each file:
   - For XML/text files: decode to string
   - **CRITICAL**: Placeholders like `{{교육명}}` may be split across multiple XML `<hp:t>` tags (e.g., `<hp:t>{{교육</hp:t><hp:t>명}}</hp:t>`). To handle this:
     1. First, try to reconstruct placeholders by extracting all text content, finding placeholder boundaries, and mapping them back to the XML
     2. A robust approach: strip all XML tags to get plain text, find placeholder positions, then work with the raw XML string using a regex that allows XML tags between characters of the placeholder pattern
     3. Simplest robust approach: use a regex like `\{\{[^}]*\}\}` on the raw XML but also handle the split-tag case by first collapsing adjacent `</hp:t>....<hp:t>` sequences within placeholder boundaries
   - Replace each `{{key}}` with the corresponding transformed value
   - After ALL replacements in a file, remove ALL `<hp:linesegarray>...</hp:linesegarray>` elements (and variants like `<linesegarray>`) from paragraphs that were modified. To be safe, remove ALL linesegarray elements from the entire file using regex: `re.sub(r'<hp:linesegarray[^>]*>.*?</hp:linesegarray>', '', xml_content, flags=re.DOTALL)` and similarly without the `hp:` prefix.
   - Verify no `{{` remains in the processed text

d) Writes all files (modified and unmodified) into a new ZIP at `/root/training_feedback_ready.hwpx`, preserving the original compression method for each entry.

### 5. Validate the output
After running the script:

```bash
# Check it's a valid ZIP/HWPX
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r')
print('Valid ZIP, entries:', len(z.namelist()))
z.testzip()
print('No corruption detected')
"

# Check no placeholders remain
python3 -c "
import zipfile, re
with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8', errors='replace')
            matches = re.findall(r'\{\{.*?\}\}', data)
            if matches:
                print(f'FAIL: {name} still has placeholders: {matches}')
        except:
            pass
    else:
        print('PASS: No placeholders remain')
"

# Verify specific transformations
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8', errors='replace')
            if '만족도' in data or '참석자수' in data or '심화반' in data:
                print(f'--- {name} ---')
                # Print relevant lines
                for line in data.split('\n'):
                    if any(k in line for k in ['만족도', '참석자수', '심화반', '점 (', '요망']):
                        print(line[:200])
        except:
            pass
"

# Verify no linesegarray remains in modified files
python3 -c "
import zipfile
with zipfile.ZipFile('/root/training_feedback_ready.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8', errors='replace')
            if 'linesegarray' in data.lower():
                print(f'WARNING: {name} still contains linesegarray')
        except:
            pass
    print('Linesegarray check complete')
"
```

### 6. Important details
- Korean labels and the static note line must remain unchanged.
- The `참석자수` value must be DIGITS ONLY (no unit suffix like 명).
- The `만족도` must follow EXACTLY the format: `X.X점 (5.0점 만점)` — use the actual numeric score from JSON for X.X.
- The overall opinion must end with `후속 심화반 검토 요망.` appended after the JSON-provided comment.
- The output must be a valid .hwpx (ZIP) package at `/root/training_feedback_ready.hwpx`.
- Remove linesegarray elements from ALL section XML files to ensure clean rendering.
- Handle the case where placeholders are split across XML tags by examining the actual XML structure first and adapting the replacement strategy accordingly.

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