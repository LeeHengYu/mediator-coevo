# Task Instruction

Complete the following task to update a HWPX renewal playbook document.

## Goal
Revise `renewal_playbook.hwpx` using data from `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Steps

### 1. Inspect source files
- List files in the task directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
- Read `renewal_update.json` to identify all field updates (customer name, current owner, renewal window, pricing band, escalation contact, pricing note, and any other fields).
- Read `followups.csv` to get the follow-up items and their `sequence` ordering.
- Extract the HWPX file (it's a ZIP archive) to a temporary directory and list its contents.
- Read `Contents/section0.xml` to understand the XML structure, namespaces, and current content.

### 2. Write a Python script to perform the update
Create a single Python script that does the following:

#### a. Parse inputs
- Load `renewal_update.json` as a dict.
- Load `followups.csv` using the `csv` module, sort rows by the `sequence` column.

#### b. Parse section0.xml
- Use `lxml.etree` to parse `Contents/section0.xml` from the original HWPX ZIP.
- Define all necessary namespaces (extract them from the root element's `nsmap`). At minimum, handle `hp`, `hc`, and any others present.

#### c. Reconstruct paragraph text and perform replacements
For each `<hp:p>` paragraph element:
1. Find all `<hp:t>` text elements (these may be split across multiple `<hp:run>` children).
2. Concatenate all `<hp:t>` text content to get the full paragraph text.
3. Determine the old values and new values from `renewal_update.json`. Build a mapping of old→new for each field. The JSON should contain both old and new values for each field, or the old values can be found by inspecting the original document.
4. Apply all text replacements on the concatenated paragraph text.
5. Handle follow-up line replacement: identify the three existing follow-up lines in the document. Replace them with the CSV items sorted by `sequence`. The follow-up lines are likely in consecutive paragraphs — match them by pattern or position.
6. If the paragraph text was modified:
   - Put the entire new text into the first `<hp:t>` element.
   - Clear (set to empty string or remove) all subsequent `<hp:t>` elements in that paragraph.
   - Remove any `<hp:lineSegArray>` child elements from the modified `<hp:p>` to clear layout cache.
   - Track that this paragraph was modified.

#### d. Preserve the appendix sentence
- Ensure the paragraph containing `이 부록 문단은 그대로 유지해야 합니다.` is NOT modified. Verify after processing that this exact text still exists in the output XML.

#### e. Reassemble the HWPX package
- Create a new ZIP file at `/root/renewal_playbook_updated.hwpx`.
- Copy all entries from the original HWPX ZIP into the new one, EXCEPT `Contents/section0.xml`.
- Write the modified `section0.xml` XML (serialized with `lxml.etree.tostring` using `xml_declaration=True` and the original encoding) into the new ZIP at `Contents/section0.xml`.
- Preserve the compression type of each entry from the original ZIP.

### 3. Run the script
Execute the Python script.

### 4. Validate the output
- Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
- Extract and read the updated `Contents/section0.xml` from the output file.
- Confirm all field replacements were applied (search for new values, confirm old values are absent).
- Confirm the follow-up lines appear in the correct sequence order.
- Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
- Confirm no `<hp:lineSegArray>` elements remain in modified paragraphs.
- If any check fails, debug and fix before finishing.

### 5. Run the verifier
If a test file exists (e.g., `test_output.py`), run `pytest` on it to confirm the verifier passes.

## Important Technical Notes
- Text in HWPX XML is commonly split across multiple `<hp:t>` elements within `<hp:run>` elements inside a single `<hp:p>`. You MUST concatenate all text in a paragraph before doing string matching/replacement, then redistribute into a single `<hp:t>` element.
- Namespace handling is critical. Use the document's own `nsmap` for XPath queries.
- Layout cache elements (`<hp:lineSegArray>`) must be removed from any paragraph whose text content was changed, to prevent overlapping character rendering.
- The output must be a proper ZIP file (not just renamed directory).
- Do NOT add duplicate content — remove old values when inserting new ones.
- Read each source file carefully before writing the transformation logic. The exact field names and old values in the JSON and the column names in the CSV must be matched precisely.

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