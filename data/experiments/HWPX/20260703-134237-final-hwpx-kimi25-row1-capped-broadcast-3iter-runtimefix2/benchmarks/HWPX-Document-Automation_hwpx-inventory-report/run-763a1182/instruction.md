# Task Instruction

Complete the inventory status report by replacing placeholders in a .hwpx template with values from a JSON file.

## Step-by-step Instructions

### 1. Inspect the workspace
```bash
ls -la /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```

### 2. Read the JSON data file
```bash
cat /root/inventory_data.json
```
Note all key-value pairs. These will be used to replace `{{key}}` placeholders.

### 3. Explore the .hwpx template structure
A `.hwpx` file is a ZIP archive. Extract it to inspect:
```bash
mkdir -p /tmp/hwpx_work
cp /root/inventory_report_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
python3 -c "import zipfile; z=zipfile.ZipFile('template.hwpx','r'); print(z.namelist()); z.extractall('extracted')"
```

### 4. Find all files containing placeholders
```bash
grep -rl '{{' /tmp/hwpx_work/extracted/
```
For each file found, display its full contents to understand the XML structure.

### 5. Examine the XML content files in detail
For each file containing `{{`, read and display the full content:
```bash
for f in $(grep -rl '{{' /tmp/hwpx_work/extracted/); do echo "=== $f ==="; cat "$f"; echo; done
```
Pay close attention to:
- How placeholders appear (they may be split across XML tags like `<hp:t>{{</hp:t><hp:t>key}}</hp:t>` — if so, you must handle this)
- The XML namespace and element structure
- Any layout cache elements (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, or similar cached layout data within paragraphs)

### 6. Write a Python script to perform the substitution
Create `/tmp/hwpx_work/process.py` that:

a. Reads `inventory_data.json` to get all key-value pairs.

b. Opens the template `.hwpx` as a ZIP file.

c. For each entry in the ZIP:
   - If the file contains `{{` placeholders (typically XML content files under `Contents/`):
     - Parse the XML content
     - Handle the case where `{{...}}` placeholders might be split across multiple `<hp:t>` (or similar text run) elements within the same paragraph. If split, merge the text runs first, perform replacement, then keep as a single text run.
     - Replace every `{{key}}` with the corresponding value from the JSON data. Convert numeric values to strings.
     - For any paragraph (`<hp:p>` or similar) whose text content was modified, remove all layout-cache child elements. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` or similar elements that cache glyph/character positioning. Removing them ensures the document re-renders cleanly.
     - Preserve all Korean text, empty paragraphs, and other unchanged content exactly.
   - Write all entries (modified or not) to the output ZIP.

d. Save the result as `/root/inventory_report_ready.hwpx`.

**CRITICAL**: When removing layout cache elements, inspect the actual XML namespace and element names first. Common patterns include elements like `linesegarray`, `LineSeg`, `lineSegArray` under paragraph elements. Remove these only from paragraphs where text was actually changed.

**CRITICAL**: Ensure placeholder text like `{{report_date}}` is fully replaced even if the `{{`, `report_date`, and `}}` are in separate XML text elements within the same paragraph. Concatenate sibling text runs, do the replacement, then put the result back.

### 7. Run the script
```bash
cd /tmp/hwpx_work && python3 process.py
```

### 8. Validate the output
```bash
# Verify it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx','r'); print('Valid ZIP. Files:', z.namelist())"

# Verify no placeholders remain
python3 -c "
import zipfile
z=zipfile.ZipFile('/root/inventory_report_ready.hwpx','r')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8','ignore')
        if '{{' in content:
            print(f'FAIL: placeholder still in {name}')
            # Show context around remaining placeholders
            import re
            for m in re.finditer(r'\{\{[^}]*\}\}', content):
                print(f'  Found: {m.group()} at pos {m.start()}')
    except: pass
print('Placeholder check complete')
"

# Verify Korean text is preserved (spot check)
python3 -c "
import zipfile
z=zipfile.ZipFile('/root/inventory_report_ready.hwpx','r')
for name in z.namelist():
    try:
        content = z.read(name).decode('utf-8','ignore')
        if any(ord(c) >= 0xAC00 and ord(c) <= 0xD7A3 for c in content):
            print(f'{name}: contains Korean text (good)')
    except: pass
"

# Verify layout cache elements removed from modified paragraphs
# (Display XML of content files for manual inspection)
python3 -c "
import zipfile
z=zipfile.ZipFile('/root/inventory_report_ready.hwpx','r')
for name in z.namelist():
    if 'section' in name.lower() or 'content' in name.lower():
        print(f'=== {name} ===')
        print(z.read(name).decode('utf-8','ignore')[:3000])
"
```

If any placeholder remains, debug by examining the raw XML around that placeholder, fix the script, and re-run.

### 9. Final confirmation
Confirm `/root/inventory_report_ready.hwpx` exists and is non-empty:
```bash
ls -la /root/inventory_report_ready.hwpx
```

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