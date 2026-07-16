# Task Instruction

Complete the following task step-by-step.

## Goal
Update the HWPX supplier contact sheet `supplier_contact_template.hwpx` using the values in `supplier_contact.json`, then save the finished file to `/root/supplier_contact_ready.hwpx`.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name 'supplier_contact_template.hwpx' -o -name 'supplier_contact.json' 2>/dev/null
```
Locate both files. If they are inside a subdirectory (e.g., `/root/HWPX-Document-Automation/hwpx-supplier-contact-sheet/`), note the full paths.

### 2. Read the JSON data
```bash
cat <path_to>/supplier_contact.json
```
Parse and understand every key-value pair. These are the replacement values for `{{...}}` placeholders.

### 3. Unzip the HWPX template
HWPX files are ZIP archives. Unzip to a temporary working directory:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip -o <path_to>/supplier_contact_template.hwpx
```

### 4. Identify all placeholder locations
Search every file in the extracted archive for `{{` patterns:
```bash
grep -rn '{{' /tmp/hwpx_work/
```
Note every file and every placeholder found. Confirm each placeholder has a matching key in the JSON.

### 5. Read the XML files containing placeholders
For each file identified (likely under `Contents/` such as `section0.xml` or similar), read the full content:
```bash
cat <file_with_placeholders>
```
Understand the XML structure around each placeholder. Pay attention to:
- Korean field labels that must be preserved
- Static note lines that must remain unchanged
- Layout cache elements like `<hp:lineSegArray>`, `<hp:lineSeg>`, `<hp:lineSegArray>` blocks within paragraphs

### 6. Perform replacements via Python
Write a Python script that:
1. Loads the JSON file.
2. For each XML file containing placeholders:
   a. Reads the file content as a UTF-8 string.
   b. Replaces every `{{placeholder_name}}` with the corresponding JSON value. Be careful: placeholders in XML may be split across XML tags (e.g., `<hp:t>{{</hp:t><hp:t>name</hp:t><hp:t>}}</hp:t>`). If this happens, you need to handle it — first check if placeholders appear intact or split.
   c. For every paragraph (`<hp:p>...</hp:p>`) whose text content was modified, remove all layout cache elements. Specifically, remove any `<hp:lineSegArray>...</hp:lineSegArray>` blocks that appear inside modified paragraphs. This prevents overlapping characters when the document is opened.
   d. Writes the modified content back to the same file path.
3. Verifies no `{{` patterns remain in any file under `/tmp/hwpx_work/`.

IMPORTANT details for the Python script:
- Use `re` module for robust replacement.
- To handle potentially split placeholders across XML tags, first try direct string replacement. Then check if any `{{` remain. If they do, implement a strategy to collapse inline XML tags within runs to reconstruct the placeholder text, then replace.
- For layout cache removal: parse each modified paragraph and strip `<hp:lineSegArray>` elements. You can use regex like `<hp:lineSegArray>.*?</hp:lineSegArray>` with `re.DOTALL` but only within paragraphs you actually modified.
- Preserve all Korean text labels and static note lines exactly as they are.

### 7. Repackage the HWPX file
Re-zip the contents back into a valid HWPX package:
```bash
cd /tmp/hwpx_work
zip -r /root/supplier_contact_ready.hwpx . -x '.*'
```
Make sure the zip is created from inside the working directory so paths are relative (no leading `/tmp/hwpx_work/` in the archive).

### 8. Validate the output
1. Confirm the output file exists:
```bash
ls -la /root/supplier_contact_ready.hwpx
```
2. Verify it's a valid ZIP:
```bash
unzip -t /root/supplier_contact_ready.hwpx
```
3. Verify no placeholders remain:
```bash
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify
unzip -o /root/supplier_contact_ready.hwpx
grep -rn '{{' /tmp/hwpx_verify/ || echo 'NO PLACEHOLDERS FOUND - OK'
```
4. Spot-check that Korean labels are still present:
```bash
grep -rn '업체' /tmp/hwpx_verify/ || true
```

### 9. Run verifier if available
```bash
cd /root && find . -name 'test_output.py' -o -name 'verify.py' -o -name 'pytest.ini' 2>/dev/null
```
If a test file exists, run it:
```bash
cd <appropriate_dir> && python -m pytest test_output.py -v
```

## Key Constraints
- Every `{{...}}` placeholder must be replaced — zero may remain.
- Korean labels must be preserved verbatim.
- Static note lines must be unchanged.
- Layout cache (`<hp:lineSegArray>` blocks) must be removed from any paragraph whose text was modified.
- The output must be a valid `.hwpx` (ZIP) package at `/root/supplier_contact_ready.hwpx`.

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