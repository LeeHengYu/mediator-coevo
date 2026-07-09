# Task Instruction

Complete the following task to update a HWPX renewal playbook document.

## Objective
Revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Steps

### 1. Inspect the workspace
- List files in the task directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, `followups.csv`, and any test/verifier files.
- Read `renewal_update.json` to learn the new field values (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
- Read `followups.csv` to learn the replacement follow-up lines and their `sequence` ordering.

### 2. Examine the HWPX structure
- A `.hwpx` file is a ZIP archive. Unzip `renewal_playbook.hwpx` to a temporary directory.
- List all files inside. Identify the main content XML (typically `Contents/section0.xml` or similar).
- Read the content XML carefully. Note all XML namespaces declared on the root element.
- Identify every paragraph where the old field values appear.
- Identify the three existing follow-up lines.
- Locate the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` and confirm its position.

### 3. Write a Python script to perform the update
Write and execute a single Python script that does all of the following:

#### 3a. Namespace registration
- Before parsing, register **every** namespace found in the XML using `xml.etree.ElementTree.register_namespace(prefix, uri)` so that serialization preserves the original prefixes exactly. This is critical to avoid prefix mangling.

#### 3b. Parse and update fields
- Parse the content XML with `ElementTree`.
- Walk all text-bearing elements (e.g., `<hp:t>` tags or equivalent) in editable sections.
- For each field in `renewal_update.json`, replace the **old** value with the **new** value everywhere it appears in paragraph text. Do not add duplicates; replace in-place.

#### 3c. Replace follow-up lines
- Identify the three existing follow-up paragraphs.
- Sort the CSV rows by the `sequence` column.
- Replace the text of the three follow-up paragraphs with the CSV items in sequence order. If there are more or fewer CSV rows than existing follow-up paragraphs, adjust accordingly (add new paragraphs or remove extras), but typically expect a 1:1 replacement.

#### 3d. Remove layout-cache elements from modified paragraphs
- For every paragraph (`<hp:p>`) whose text content was modified, find and remove any `<hp:lineSegArray>` child elements (layout cache). This prevents overlapping-character rendering issues when the document is opened.

#### 3e. Preserve the appendix sentence
- Verify that the paragraph containing `이 부록 문단은 그대로 유지해야 합니다.` is unchanged after all edits.

#### 3f. Repackage as HWPX
- Write the updated ZIP to `/root/renewal_playbook_updated.hwpx`.
- The `mimetype` file **must** be the first entry in the ZIP and must be written with `ZIP_STORED` (no compression).
- All other files from the original archive must be included, preserving their original paths. Only the modified content XML differs.

### 4. Validate
- Confirm `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
- Unzip it and re-read the content XML. Verify:
  - All six fields reflect the new values from `renewal_update.json`.
  - The follow-up lines match the CSV items in sequence order.
  - No old field values remain in editable sections.
  - The appendix sentence is intact.
  - No `<hp:lineSegArray>` elements exist in modified paragraphs.
- If a test file (e.g., `test_output.py`) exists in the task directory, run it with `pytest` to confirm the verifier passes.

### Key Technical Constraints (from prior successful runs)
- **Namespace registration**: Register all namespaces before parsing to prevent ElementTree from renaming prefixes (e.g., `ns0:` instead of `hp:`).
- **Layout cache removal**: Remove `hp:lineSegArray` from any paragraph you modify.
- **ZIP mimetype first + stored**: The mimetype entry must be first and uncompressed.
- **No stale duplicates**: Replace old values, don't append new ones alongside them.

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