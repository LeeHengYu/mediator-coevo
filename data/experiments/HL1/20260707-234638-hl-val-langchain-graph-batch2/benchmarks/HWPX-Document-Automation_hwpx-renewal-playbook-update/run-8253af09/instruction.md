# Task Instruction

Complete the following task to update a Hancom HWPX renewal playbook document.

## Objective
Revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### Step 1: Inspect the workspace
- List files in the current working directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
- Read `renewal_update.json` to identify the new values for: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
- Read `followups.csv` to get the follow-up items; note the `sequence` column for ordering.

### Step 2: Unzip the HWPX package
- HWPX is an OPC (ZIP-based) package. Unzip `renewal_playbook.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`).
- List the extracted contents to understand the package structure. The main content XML is typically under `Contents/section0.xml` (or similar).

### Step 3: Inspect the content XML
- Read the section XML file(s) to understand the document structure.
- Identify all paragraphs containing the old values that need replacement.
- Identify the three existing follow-up lines.
- Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` exists and note its location.

### Step 4: Write a Python script to perform all modifications
Create and run a Python script that does the following:

1. **Parse XML with lxml**, registering all namespaces (especially `hp`) to prevent prefix corruption during serialization. Use `lxml.etree.parse()` and register namespaces via `etree.register_namespace()` or by extracting them from the root element.

2. **Load update data**: Read `renewal_update.json` and `followups.csv` (sorted by `sequence`).

3. **Build a mapping of old→new values** from the JSON. For each field (customer name, current owner, renewal window, pricing band, escalation contact, pricing note), identify the old value from the existing XML text and the new value from the JSON.

4. **Text replacement in editable sections**: Iterate over all text runs (`<hp:t>` elements or equivalent) in the content XML. For each run's text, replace old values with new values. Do NOT modify the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`.

5. **Follow-up replacement**: Locate the three existing follow-up lines. Remove them and insert the new follow-up items from the CSV in `sequence` order. Ensure no duplicate lines remain. Reuse the paragraph structure/style of the original follow-up lines for the new ones.

6. **Layout cache invalidation**: For every `<hp:p>` paragraph whose text content was modified, remove any `<hp:lineSegArray>` child elements. This forces the word processor to recalculate layout and prevents overlapping characters.

7. **Preserve the appendix sentence** exactly as-is. Do not modify its paragraph or any of its attributes.

8. **Serialize the modified XML** back to the file, using `etree.tostring()` with `xml_declaration=True` and `encoding='UTF-8'`.

### Step 5: Re-zip into a valid HWPX package
- Using Python's `zipfile` module, create `/root/renewal_playbook_updated.hwpx` with `ZIP_DEFLATED` compression.
- Walk the temporary directory and add all files using **relative paths** (matching the original archive structure).
- Ensure no extra directory prefixes are introduced.

### Step 6: Validate the output
- Confirm `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
- Unzip it to a second temp directory and read back the content XML.
- Verify:
  - All old field values are gone (no stale customer name, owner, etc.).
  - All new field values are present.
  - Follow-up items appear in correct `sequence` order.
  - No duplicate follow-up lines from the old version remain.
  - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is intact.
  - Modified paragraphs have no `<hp:lineSegArray>` elements.
  - The ZIP structure matches the original package layout.

## Critical Technical Notes
- **Namespace handling**: When parsing HWPX XML with lxml, collect all namespace declarations from the root element and register them before any serialization. This prevents lxml from generating `ns0:`, `ns1:` prefixes that would corrupt the document.
- **lineSegArray removal**: This is mandatory for any modified paragraph. Without it, the document will display overlapping or garbled text.
- **CSV sorting**: Sort `followups.csv` rows by the `sequence` column (ascending) before inserting into the document.
- **No test/verifier weakening**: Do not skip any of the requirements. All old values must be replaced, not appended alongside.

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