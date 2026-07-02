# Task Instruction

## Task: Complete HWPX Inventory Report

### Goal
Replace all `{{...}}` placeholders in `inventory_report_template.hwpx` with values from `inventory_data.json`, then save the result to `/root/inventory_report_ready.hwpx`.

### Step-by-step Plan

1. **Understand the HWPX format.** A `.hwpx` file is a ZIP-based package (like OOXML). The main document content is typically in XML files inside the archive. Start by listing the contents of `inventory_report_template.hwpx`:
   ```
   cd /root
   python3 -c "import zipfile; z=zipfile.ZipFile('inventory_report_template.hwpx'); print('\n'.join(z.namelist()))"
   ```

2. **Locate placeholder text.** Search every file inside the ZIP for `{{` to find which XML files contain placeholders:
   ```python
   import zipfile
   z = zipfile.ZipFile('inventory_report_template.hwpx')
   for name in z.namelist():
       try:
           data = z.read(name).decode('utf-8', errors='replace')
           if '{{' in data:
               print(f'--- {name} ---')
               # print surrounding context for each placeholder
               import re
               for m in re.finditer(r'\{\{.*?\}\}', data):
                   start = max(0, m.start()-200)
                   end = min(len(data), m.end()+200)
                   print(data[start:end])
                   print('...')
       except: pass
   ```

3. **Read the JSON data.**
   ```python
   import json
   with open('inventory_data.json') as f:
       data = json.load(f)
   print(json.dumps(data, indent=2, ensure_ascii=False))
   ```

4. **Perform replacements carefully.** For each XML file that contains placeholders:
   - Read the raw bytes/text.
   - For each `{{key}}` placeholder, replace it with the corresponding value from the JSON. The JSON keys should match the placeholder names (e.g., `{{report_date}}` → `data["report_date"]`).
   - **Important:** After replacing text in a paragraph, remove any stale layout-cache / char-shape-run elements that would cause overlapping characters. Specifically, look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar layout cache elements within the same paragraph (`<hp:p>`) and remove them entirely. These are pre-computed glyph positions that become invalid after text changes.
   - Preserve all Korean labels, static note lines, and empty paragraphs exactly as they are.

5. **Rebuild the HWPX package.** Create the output file by copying the original ZIP structure and replacing only the modified XML files:
   ```python
   import zipfile, shutil, os
   
   src = 'inventory_report_template.hwpx'
   dst = '/root/inventory_report_ready.hwpx'
   
   # Read all entries from original
   with zipfile.ZipFile(src, 'r') as zin:
       with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
           for item in zin.infolist():
               raw = zin.read(item.filename)
               if item.filename in modified_files:
                   zout.writestr(item, modified_files[item.filename])
               else:
                   zout.writestr(item, raw)
   ```
   Preserve the original compression type and directory structure.

6. **Validate the output.**
   - Verify the output is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/inventory_report_ready.hwpx'); z.testzip()"`
   - Verify NO `{{` placeholders remain anywhere in the output:
     ```python
     z = zipfile.ZipFile('/root/inventory_report_ready.hwpx')
     for name in z.namelist():
         data = z.read(name).decode('utf-8', errors='replace')
         if '{{' in data:
             print(f'FAIL: placeholder still in {name}')
     print('All clear')
     ```
   - Verify Korean labels and empty paragraphs are preserved by comparing paragraph counts and static text between original and output.
   - Verify layout cache elements were removed from modified paragraphs.

7. **Run the verifier** (likely `pytest` or a test script in the task directory) to confirm the task passes.

### Key Pitfalls to Avoid
- **Split placeholders:** In HWPX XML, a `{{placeholder}}` might be split across multiple XML text runs (e.g., `<hp:t>{{place</hp:t><hp:t>holder}}</hp:t>`). If you find placeholders are split, you must concatenate adjacent text runs within the same paragraph, perform the replacement, then put the result back (potentially in a single run).
- **Layout cache staleness:** Any paragraph where you change text content MUST have its `<hp:linesegarray>` or equivalent layout cache child elements removed. Inspect the XML structure to identify the exact element names.
- **Encoding:** Write XML back as UTF-8 with the same XML declaration as the original.
- **Empty paragraphs:** Do not remove any `<hp:p>` elements that have no text content; they serve as spacing.
- **JSON value types:** Some values might be numbers; convert them to strings before substitution.

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