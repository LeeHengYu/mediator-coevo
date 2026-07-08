# Task Instruction

Complete the following task to update a HWPX renewal playbook document.

## Goal
Revise `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### 1. Inspect input files
- Read `renewal_update.json` to identify the field mappings (customer name, current owner, renewal window, pricing band, escalation contact, pricing note — both old and new values).
- Read `followups.csv` to get the follow-up items and their `sequence` ordering.
- Unzip `renewal_playbook.hwpx` to a temp directory (e.g., `/tmp/hwpx_work/`).
- List the archive contents to understand structure. The main content is typically in `Contents/section0.xml`.

### 2. Parse the XML with full namespace registration
- Read `Contents/section0.xml`.
- Before parsing with ElementTree, extract ALL namespace declarations from the root element and register each one with `ET.register_namespace(prefix, uri)`. This is critical to avoid namespace corruption.
- Parse the XML.

### 3. Apply field updates (text replacements)
- For each field in `renewal_update.json`, find every `<hp:t>` element (or text-bearing element) in the document whose text contains the OLD value and replace it with the NEW value.
- This covers: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
- Make sure replacements happen everywhere the old values appear (there may be multiple occurrences).

### 4. Replace follow-up lines
- Identify the three existing follow-up paragraph elements (`<hp:p>`) in the document. They should be identifiable by their text content (look for follow-up related text patterns — inspect the actual content first).
- Sort the CSV follow-up items by the `sequence` column.
- Strategy: Clone the first follow-up paragraph's structure for each new item. Remove the old follow-up paragraphs. Insert the new ones at the same position in the parent element, preserving the XML tree structure.
- If there are more or fewer CSV items than original follow-ups, handle accordingly (add or remove paragraph elements).

### 5. Remove layout cache from modified paragraphs
- For EVERY `<hp:p>` element whose text was modified (either by field replacement or follow-up replacement), find and REMOVE any `<hp:linesegarray>` child element. This prevents overlapping character rendering when the document is opened.
- This is a critical step — do not skip it.

### 6. Preserve the appendix sentence
- Verify that the paragraph containing `이 부록 문단은 그대로 유지해야 합니다.` is unchanged. Do NOT modify this paragraph's text or remove its layout cache.

### 7. Write the updated XML
- Write the modified XML back to `Contents/section0.xml` in the temp directory.
- Use `xml_declaration=True, encoding='UTF-8'` when writing.

### 8. Repackage as HWPX
- Create `/root/renewal_playbook_updated.hwpx` as a ZIP file.
- The `mimetype` file MUST be the FIRST entry in the ZIP and MUST be stored uncompressed (`compression=ZIP_STORED`, no extra field). This is required for HWPX/EPUB format validity.
- Add all other files from the temp directory with normal ZIP compression (`ZIP_DEFLATED`).

### 9. Validate
- Verify the output file exists at `/root/renewal_playbook_updated.hwpx`.
- Unzip it to a separate temp location and re-read section0.xml to confirm:
  - All old field values are gone (no stale values).
  - All new field values are present.
  - Follow-up items match the CSV in sequence order.
  - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is intact.
  - No `hp:linesegarray` elements remain in modified paragraphs.
  - The mimetype entry is first in the archive.

## Important Reminders
- Do NOT add new paragraphs alongside old ones for follow-ups — REPLACE them.
- Do NOT modify the appendix paragraph.
- Register ALL namespaces before parsing to avoid prefix mangling.
- Remove `hp:linesegarray` from every modified `hp:p` — this is the #1 cause of layout corruption.

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