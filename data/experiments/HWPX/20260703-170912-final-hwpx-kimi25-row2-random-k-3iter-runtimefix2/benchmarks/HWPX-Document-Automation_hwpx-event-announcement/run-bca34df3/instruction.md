# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json`, then save the result to `/root/event_announcement_ready.hwpx`.

## Steps

1. **Inspect the workspace**: List files in the current directory to find `event_announcement_template.hwpx` and `event_data.json`. If they are in a subdirectory (e.g., `/root/` or a task-specific folder), locate them.

2. **Read `event_data.json`**: Load the JSON file and examine all key-value pairs. These are the replacement values for placeholders.

3. **Understand the HWPX format**: A `.hwpx` file is a ZIP archive containing XML files (typically under `Contents/` with files like `section0.xml`, `section1.xml`, etc.). Extract the template to a temporary directory.

4. **Extract the template**:
   ```python
   import zipfile, os, json, re, shutil
   
   TEMPLATE = '<path_to>/event_announcement_template.hwpx'
   OUTPUT = '/root/event_announcement_ready.hwpx'
   TMPDIR = '/tmp/hwpx_work'
   
   if os.path.exists(TMPDIR):
       shutil.rmtree(TMPDIR)
   
   with zipfile.ZipFile(TEMPLATE, 'r') as z:
       z.extractall(TMPDIR)
   ```

5. **Load JSON data**:
   ```python
   with open('<path_to>/event_data.json', 'r', encoding='utf-8') as f:
       data = json.load(f)
   ```

6. **Replace placeholders in ALL files in the extracted archive** (not just XML — check all text-based files). For each file in the extracted directory tree:
   - Read the file content as UTF-8 text (skip binary files gracefully).
   - For each key-value pair in the JSON data, replace `{{key}}` with the corresponding value (convert non-string values to strings).
   - Also handle any nested keys if the JSON has nested structure — flatten with appropriate key names matching the placeholder patterns found in the XML.
   - After replacements, remove any `<hp:lineSegArray>...</hp:lineSegArray>` blocks (including multiline) from the content if any replacement was made in that file. Use regex: `re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', content, flags=re.DOTALL)`
   - Write the modified content back.

7. **Verify no placeholders remain**: After all replacements, scan all text files in the extracted directory for any remaining `{{...}}` patterns. If any are found, investigate the JSON keys and fix the mapping.

8. **Repackage as HWPX**:
   ```python
   with zipfile.ZipFile(OUTPUT, 'w', zipfile.ZIP_DEFLATED) as zout:
       for root, dirs, files in os.walk(TMPDIR):
           for fname in files:
               full = os.path.join(root, fname)
               arcname = os.path.relpath(full, TMPDIR)
               zout.write(full, arcname)
   ```

9. **Validate the output**:
   - Confirm `/root/event_announcement_ready.hwpx` exists and is a valid ZIP.
   - Open it with `zipfile.ZipFile` and list contents to ensure structure is intact.
   - Read the section XML files from the output ZIP and verify:
     - No `{{` or `}}` patterns remain.
     - Korean labels and static note lines are preserved.
     - The JSON values appear in the content.

10. **Run any available tests**: Check if there's a `tests/` directory or `test_output.py` and run it with `pytest` to confirm.

## Key Constraints
- Do NOT alter Korean text labels or static note lines — only replace `{{...}}` placeholders.
- ALL placeholders must be replaced; none may remain.
- Remove `<hp:lineSegArray>` blocks from any XML file where text was modified (this prevents stale layout cache causing overlapping characters).
- The output must be a valid `.hwpx` (ZIP) package with the same internal structure as the template.
- Be careful with the JSON key-to-placeholder mapping: inspect the actual placeholder names in the XML and match them exactly to JSON keys.

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