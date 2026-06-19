# Task Instruction

Revise the existing renewal playbook `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, then save the updated file to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### 1. Inspect input files
- Read `renewal_update.json` to identify all field updates (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
- Read `followups.csv` to get the follow-up items and their `sequence` ordering.
- Unzip `renewal_playbook.hwpx` into a temporary directory (e.g., `/tmp/hwpx_work/`).
- List the extracted contents to understand the package structure.
- Read the main content XML file (likely `Contents/section0.xml` or similar) and any other XML files to understand the document structure, namespaces, and where editable content lives.
- Identify the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` and note its location so it is preserved.
- Identify the three existing follow-up lines that need to be replaced.

### 2. Write a Python script to perform all modifications
Create a Python script that:

a. **Registers all XML namespaces** found in the document before parsing (use `xml.etree.ElementTree` with namespace registration, or parse with `lxml` if available). This prevents namespace prefix loss.

b. **Performs text replacements** for all fields from `renewal_update.json`:
   - First, identify the OLD values currently in the document by inspecting the XML.
   - Replace every occurrence of each old value with the corresponding new value from the JSON.
   - Handle the case where text may be split across multiple `<hp:t>` tags within a single paragraph's run elements. Strategy: for each `<hp:p>` paragraph, concatenate all `<hp:t>` text, check if it contains an old value, and if so, consolidate the text into fewer runs and perform the replacement.

c. **Replaces the three follow-up lines** with CSV items sorted by `sequence`:
   - Read `followups.csv`, sort by `sequence` column.
   - Identify the three existing follow-up paragraphs in the XML.
   - Replace their text content with the new follow-up items from the CSV, maintaining the paragraph XML structure.
   - If there are more or fewer CSV items than existing lines, add or remove paragraph elements accordingly.

d. **Removes layout cache elements** (`<hp:lineSegArray>` or similar `lineSegArray` elements) from every `<hp:p>` paragraph whose text content was modified. This is critical to prevent overlapping character rendering.

e. **Preserves the appendix sentence** `이 부록 문단은 그대로 유지해야 합니다.` exactly as-is. Do not modify the paragraph containing this text.

f. **Writes the modified XML** back to the extracted directory.

g. **Re-packages the HWPX file** by zipping from within the root of the extracted directory to `/root/renewal_playbook_updated.hwpx`, ensuring the directory structure (`mimetype`, `Contents/`, etc.) is at the archive root. Use `zipfile` module with appropriate compression.

### 3. Execute and validate
- Run the Python script.
- Verify the output file exists at `/root/renewal_playbook_updated.hwpx`.
- Unzip the output and inspect the content XML to confirm:
  - All old field values are gone (no duplicates).
  - All new field values from `renewal_update.json` are present.
  - Follow-up lines match the CSV items in sequence order.
  - The appendix sentence is unchanged.
  - No `lineSegArray` elements remain in modified paragraphs.
  - The file is a valid ZIP archive.

### Important Notes
- When re-zipping, if a `mimetype` file exists, store it first with `ZIP_STORED` (no compression) as required by ODF-like package formats.
- Use string-level replacement on the serialized XML as a fallback if ElementTree-based replacement misses split tokens, but always re-parse to validate well-formedness afterward.
- Do NOT modify any paragraph that you don't need to change.
- Remove old values completely; do not leave them alongside new values.

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