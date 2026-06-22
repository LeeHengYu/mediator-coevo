# Task Instruction

Complete the inventory status report by replacing placeholders in a `.hwpx` template with JSON data.

## Steps

### 1. Inspect the workspace
```bash
ls -la /root/
cat /root/inventory_data.json
```

### 2. Understand the `.hwpx` structure
A `.hwpx` file is a ZIP archive. Unzip it to inspect its contents:
```bash
mkdir -p /tmp/hwpx_work
cp /root/inventory_report_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f | sort
```

### 3. Find all placeholder locations
Search all XML files for `{{` patterns to find every placeholder:
```bash
grep -rn '{{' template_contents/
```
Also check if placeholders might be split across XML tags (e.g., `{{` in one element and `}}` in another). If so, you'll need to handle tag-spanning placeholders by concatenating text runs within the same paragraph before replacing.

### 4. Read and parse the JSON data
```bash
cat /root/inventory_data.json
```
Note every key-value pair. Each `{{key}}` in the template must be replaced with the corresponding value.

### 5. Write a Python script to perform the replacement
Create `/tmp/hwpx_work/process.py` that:

a. Extracts the template `.hwpx` (ZIP) into a working directory.

b. Loads `inventory_data.json` into a dictionary.

c. For each XML file in the package (especially files under `Contents/` like `section0.xml` or similar content XML files):
   - Parse the XML properly (use `lxml.etree` or `xml.etree.ElementTree` with namespace handling).
   - For each paragraph element, collect ALL text content across child runs/spans to detect placeholders that may span multiple XML elements.
   - If a placeholder spans multiple text elements within the same paragraph, merge the text into the first text element and clear the others, then perform the replacement.
   - Replace every `{{key}}` with the corresponding JSON value.
   - **Critical**: For any paragraph where text was modified, remove layout-cache elements. These are typically elements like `<lineseg>`, `<linesegarray>`, or elements within a namespace related to layout caching (look for elements containing 'lineseg', 'lineSegArray', or similar). Remove these from the modified paragraph so the document renders cleanly.
   - Preserve all empty paragraphs exactly as they are.
   - Preserve all Korean text that is not part of a placeholder.

d. Write the modified XML files back, preserving XML declarations and encoding.

e. Re-package everything into a valid ZIP file saved as `/root/inventory_report_ready.hwpx`, preserving the original ZIP structure (same directory layout, same compression method). Use `zipfile` module.

f. After creating the output, verify:
   - The output is a valid ZIP: `zipfile.is_zipfile('/root/inventory_report_ready.hwpx')`
   - No `{{` remains in any file: scan all text content in all XML files.
   - Print confirmation of replacements made.

### 6. Run the script
```bash
python3 /tmp/hwpx_work/process.py
```

### 7. Validate the output
```bash
cd /tmp && python3 -c "
import zipfile, sys
z = zipfile.ZipFile('/root/inventory_report_ready.hwpx')
for name in z.namelist():
    data = z.read(name).decode('utf-8', errors='ignore')
    if '{{' in data:
        print(f'FAIL: placeholder remains in {name}')
        sys.exit(1)
print('PASS: no placeholders remain')
z.close()
"
```

### Important Notes
- When handling XML namespaces, register them properly so they are preserved in output (don't let ElementTree mangle namespace prefixes).
- When writing XML back, ensure the same encoding (UTF-8) and XML declaration are used.
- The ZIP must contain the same files in the same paths as the original.
- Do NOT modify paragraphs that have no placeholders.
- Pay special attention to how text runs are structured in HWPX XML — the text content may be in `<hp:t>` or `<t>` elements nested within run `<hp:run>` or `<run>` elements within paragraph `<hp:p>` or `<p>` elements. Inspect the actual XML to determine the exact element names and namespaces before writing the replacement logic.

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