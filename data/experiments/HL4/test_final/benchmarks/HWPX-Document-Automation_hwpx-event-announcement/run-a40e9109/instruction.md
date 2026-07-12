# Task Instruction

Complete the following task step by step.

## Goal
Fill in all `{{...}}` placeholders in the HWPX template `event_announcement_template.hwpx` using values from `event_data.json`, and save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Explore the workspace
```bash
ls -la /root/
find /root/ -maxdepth 2 -type f | head -40
```
Locate `event_announcement_template.hwpx` and `event_data.json`. Note their exact paths.

### 2. Read the JSON data
```bash
cat <path_to>/event_data.json
```
Record every key-value pair. These are the substitution values.

### 3. Inspect the HWPX package structure
HWPX files are ZIP archives. List contents:
```bash
python3 -c "
import zipfile, sys
with zipfile.ZipFile('<path_to>/event_announcement_template.hwpx', 'r') as z:
    for info in z.infolist():
        print(info.filename, info.file_size)
"
```

### 4. Find which XML files contain placeholders
For each file in the ZIP, search for `{{`:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('<path_to>/event_announcement_template.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            data = z.read(name).decode('utf-8')
            if '{{' in data:
                print(f'=== {name} ===')
                # Print lines containing placeholders
                for i, line in enumerate(data.split('\n')):
                    if '{{' in line:
                        print(f'  Line {i}: {line[:300]}')
        except:
            pass
"
```

### 5. Examine the XML structure around placeholders
For each file that contains `{{`, dump its full content so you understand the XML structure, especially:
- How text runs are structured (the placeholder text may be split across multiple XML text nodes within a single paragraph)
- What layout-cache elements look like (e.g., `<lc:` prefixed elements, or elements like `<hp:lineSegArray>`, `<hp:charShapeArray>`, `<linesegarray>`, `<parameterset>` with layout info)
- The namespace declarations

### 6. Write the replacement script
Write a Python script `/root/fill_template.py` that:

a) Reads `event_data.json` into a dict.

b) Opens the template HWPX as a ZIP.

c) For each file in the ZIP:
   - If it's an XML file containing `{{`, process it:
     1. Parse or read the XML content as text.
     2. **CRITICAL**: Placeholders like `{{event_name}}` might be split across multiple XML text nodes within a single run or across runs. For example: `<t>{{event</t><t>_name}}</t>`. You must handle this. Strategy: for each XML file, first do a raw text concatenation check. If simple string replacement on the full XML text works (i.e., `{{key}}` appears as a contiguous string in the raw XML), use simple replacement. If not, you need to reconstruct by joining text nodes within each paragraph, performing replacement, and redistributing.
     3. Replace every `{{key}}` with the corresponding value from the JSON. Use all keys from the JSON.
     4. **Remove layout-cache elements** from any paragraph that was modified. Look for elements like `<hp:linesegarray>`, `<hp:parameterset>`, or any element related to layout caching within the paragraph (`<hp:p>`) scope. Specifically look for `<lineseg` or `<lineSeg` elements, `<charPr` cached arrays, or a dedicated layout section. Inspect the actual XML to determine the exact element names. Remove these from modified paragraphs only.
     5. After all replacements, verify no `{{` remains in the content.
   - If it's not an XML file or has no placeholders, copy it unchanged.

d) Write all files to a new ZIP at `/root/event_announcement_ready.hwpx`, preserving the original compression method and directory structure.

e) After writing, verify:
   - The output is a valid ZIP
   - No `{{` remains in any text file within the ZIP
   - Print confirmation

### 7. Run the script
```bash
python3 /root/fill_template.py
```

### 8. Validate the output
```bash
python3 -c "
import zipfile
found_placeholder = False
with zipfile.ZipFile('/root/event_announcement_ready.hwpx', 'r') as z:
    print('Files in output:')
    for name in z.namelist():
        print(f'  {name}')
        try:
            data = z.read(name).decode('utf-8')
            if '{{' in data:
                print(f'    WARNING: placeholder found in {name}')
                found_placeholder = True
        except:
            pass
if found_placeholder:
    print('FAIL: placeholders remain')
else:
    print('PASS: no placeholders remain')
"
```

Also verify the output file exists and is non-trivially sized:
```bash
ls -la /root/event_announcement_ready.hwpx
```

### Important constraints
- **Do NOT change Korean labels or static note lines** — only replace `{{...}}` patterns.
- **Every `{{...}}` must be replaced** — none may remain.
- **Layout-cache cleanup**: For each paragraph (`<hp:p>` or equivalent) where you changed text content, remove child elements that cache glyph positions/line layout. Inspect the actual XML to identify these elements precisely before writing removal code. Common HWPX layout-cache elements include things under `<hp:linesegarray>` or similar. If you're unsure which elements are layout caches, print the full XML of one modified paragraph and identify elements that look like pre-computed layout data (arrays of positions, widths, etc.).
- The output must be a valid ZIP/HWPX package with the same internal structure as the original.
- Use `compression=zipfile.ZIP_DEFLATED` or match the original file's compression method for each entry.

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