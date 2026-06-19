# Task Instruction

Complete the following task to update a Korean HWPX document. HWPX is a ZIP-based package format containing XML content files.

## Step 1: Inspect the workspace

- List files in the current directory and locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
- Read `renewal_update.json` to identify all field mappings (customer name, current owner, renewal window, pricing band, escalation contact, pricing note — both old and new values).
- Read `followups.csv` to see the follow-up items and their `sequence` column for ordering.

## Step 2: Explore the HWPX package structure

- Copy `renewal_playbook.hwpx` to a working location and unzip it to inspect the package structure.
- List all files in the extracted package. Identify the main content XML file(s) — typically under a path like `Contents/` with names like `section0.xml` or similar.
- Read the content XML file(s) carefully. Identify:
  - The XML namespaces used.
  - Where the editable field values appear (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
  - Where the three follow-up lines appear.
  - Where the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` appears.
  - Any layout-cache elements (look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, or similar cached layout data within paragraph elements).

## Step 3: Plan the edits

- From `renewal_update.json`, extract the old→new value mappings for each field.
- From `followups.csv`, read all rows and sort by the `sequence` column. These will replace the existing three follow-up lines.
- Identify exactly which XML elements/text nodes need modification.

## Step 4: Apply the edits using Python

Write a Python script that:

1. Extracts the HWPX ZIP to a temp directory (preserving all files).
2. Parses the content XML file(s) using `xml.etree.ElementTree` (with proper namespace handling — register namespaces before parsing to avoid ns0/ns1 prefix mangling).
3. For each field in the update JSON:
   - Finds all text nodes containing the old value and replaces with the new value.
   - Ensures replacements happen everywhere the old value appears in editable sections.
4. For the follow-up lines:
   - Identifies the three existing follow-up line paragraphs.
   - Replaces their text content with the CSV items sorted by `sequence`.
   - If there are more or fewer CSV items than existing lines, add or remove paragraph elements accordingly.
5. For every paragraph element whose text was modified:
   - Removes any child elements that represent layout cache data (e.g., `linesegarray`, `lineSegArray`, `lineseg`, or similar elements). These are typically direct children of paragraph `<p>` elements. Inspect the actual element names in the XML before removing.
6. Verifies the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is still present and unchanged.
7. Serializes the modified XML back (with proper XML declaration and encoding).
8. Repackages everything into a new ZIP file saved as `/root/renewal_playbook_updated.hwpx`, preserving the original ZIP structure (mimetype file first if present, no extra compression on mimetype).

## Step 5: Validate the output

- Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
- Extract it and re-read the content XML to confirm:
  - All old field values are gone (search for each old value — none should remain in editable sections).
  - All new field values are present.
  - Follow-up lines match the CSV items in sequence order.
  - The appendix sentence is intact.
  - No layout-cache elements remain on modified paragraphs.
  - The XML is well-formed.

## Important Notes

- Before editing XML, register ALL namespaces found in the document to prevent ElementTree from rewriting namespace prefixes.
- Be careful with Korean text encoding (UTF-8).
- Do NOT modify the appendix paragraph or any non-editable structural content.
- Remove old values entirely — do not leave duplicates.
- The final .hwpx must be a valid ZIP package with the same internal structure as the original.

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