# Task Instruction

Complete the following task to update a Korean HWPX document.

## Background
A `.hwpx` file is a ZIP-based package containing XML content files. You need to update `renewal_playbook.hwpx` using data from `renewal_update.json` and `followups.csv`, then save the result to `/root/renewal_playbook_updated.hwpx`.

## Step-by-step Plan

### 1. Explore the workspace
- List files in the working directory to locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
- Read `renewal_update.json` fully to understand all field mappings (customer name, current owner, renewal window, pricing band, escalation contact, pricing note — both old and new values).
- Read `followups.csv` fully. Note the `sequence` column for ordering.

### 2. Inspect the HWPX package structure
- Unzip `renewal_playbook.hwpx` into a temporary directory (e.g., `/tmp/hwpx_orig/`).
- List all files in the extracted archive to understand the package structure.
- Read each XML content file (especially files under `Contents/` such as `section0.xml` or similar) to find where the editable text lives.
- Identify the XML namespace(s) used (likely `http://www.hancom.co.kr/hwpml/...` or similar).

### 3. Understand the content structure
- Locate all paragraphs containing the old values for: customer name, current owner, renewal window, pricing band, escalation contact, and pricing note.
- Locate the three existing follow-up lines that need to be replaced.
- Locate the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` and note its position — this must NOT be modified.

### 4. Perform the edits using Python
Write a Python script that:

a) **Extracts** the hwpx ZIP to a temp directory.

b) **Parses** `renewal_update.json` to get old→new value mappings for all six fields.

c) **Parses** `followups.csv` and sorts rows by the `sequence` column.

d) **For each XML content file** in the package:
   - Parse it as XML (use `lxml.etree` or `xml.etree.ElementTree`).
   - For each paragraph element, extract its full text content.
   - **Field replacements**: For each of the six fields, find text runs containing old values and replace them with new values. Be careful to handle cases where a value might span multiple text runs within a paragraph — if so, consolidate or replace appropriately. Replace ALL occurrences everywhere in editable sections.
   - **Follow-up replacement**: Identify the three consecutive follow-up line paragraphs (by matching their old text content). Replace them with the CSV items in sequence order. If there are more or fewer CSV items than original lines, add or remove paragraph elements accordingly.
   - **Layout-cache cleanup**: For any paragraph whose text was modified, remove all child elements that represent layout cache data. These are typically elements like `<hp:linesegarray>`, `<lineseg>`, `<hp:lineSegArray>`, or similar layout/positioning cache elements within the paragraph. Search for elements with names containing 'lineseg', 'LineSeg', 'lineSegArray', or similar patterns and remove them from modified paragraphs.
   - **Preserve the appendix sentence** `이 부록 문단은 그대로 유지해야 합니다.` — do not modify the paragraph containing this text.

e) **Write** modified XML back to the same file paths.

f) **Repackage** the directory into a valid ZIP file saved as `/root/renewal_playbook_updated.hwpx`. Ensure:
   - The ZIP uses the same compression method as the original.
   - All original files (including non-XML files like `mimetype`, `META-INF/`, etc.) are preserved.
   - The `mimetype` file, if present, should be stored first and uncompressed (as per OPC conventions).

### 5. Validate the result
- Unzip `/root/renewal_playbook_updated.hwpx` to a new temp directory.
- Read the content XML files and verify:
  - All six fields show new values, not old values.
  - The follow-up lines match the CSV items in sequence order.
  - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is present and unchanged.
  - No old field values remain in the document.
  - Modified paragraphs do not contain layout-cache elements.
  - The ZIP is structurally valid.

### Important Notes
- Work carefully with XML namespaces. Print namespace maps when parsing to ensure correct element selection.
- When replacing text in XML runs, be precise: the text might be split across multiple `<hp:t>` or `<t>` elements within a run. Handle this by examining the full concatenated text of a paragraph to find matches, then performing the replacement at the appropriate text node level.
- Do NOT modify any paragraph containing the appendix sentence.
- Remove layout-cache elements ONLY from paragraphs you actually modified.
- Ensure the output is a proper ZIP file with `.hwpx` extension.

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