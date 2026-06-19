# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in `inventory_report_template.hwpx` with values from `inventory_data.json`, then save the result to `/root/inventory_report_ready.hwpx`.

## Step-by-step Plan

### 1. Inspect the input files
- Read `inventory_data.json` to understand all available key-value pairs.
- List the contents of `inventory_report_template.hwpx` (it's a ZIP archive) to identify which XML files contain content (typically `Contents/section0.xml` or similar).
- Extract and read each XML section file to locate all `{{...}}` placeholders.

### 2. Build a placeholder-to-value mapping
- Parse `inventory_data.json` into a Python dictionary.
- Identify every unique `{{key}}` placeholder found across all XML section files.
- Confirm every placeholder key exists in the JSON data. If any key is missing, report it before proceeding.

### 3. Replace placeholders in XML
Use Python with `zipfile` and `lxml.etree` (or `xml.etree.ElementTree`):

**Critical: Handle fragmented placeholders.** The `{{` and `}}` delimiters and the key name may be split across multiple `<hp:t>` (or similar text run) elements within a single paragraph. You must:
- For each paragraph element, concatenate all text-run contents to form the full paragraph text.
- Find all `{{...}}` patterns in the concatenated text.
- Replace them with the corresponding JSON values.
- Redistribute the replaced text back into the run elements. A simple approach: put all replaced text into the first run's text element and clear the remaining runs (but keep the elements to preserve structure), OR collapse into a single run per paragraph where placeholders were found.

### 4. Remove layout cache elements
For every paragraph whose text content was modified:
- Remove `<hp:lineSegArray>` elements (and any child elements) from that paragraph.
- This prevents overlapping/stale character rendering when the document is opened.
- Do NOT remove lineSegArray from paragraphs that were not modified.

### 5. Preserve document integrity
- Keep all Korean labels and static note lines unchanged.
- Preserve empty paragraphs (paragraphs with no text or only whitespace) — do not delete them.
- Preserve the XML namespace declarations exactly as they appear.
- Preserve all other files in the ZIP archive unchanged (e.g., `header.xml`, `content.hpf`, images, etc.).

### 6. Write the output HWPX file
- Create `/root/inventory_report_ready.hwpx` as a new ZIP file.
- Copy all entries from the original template ZIP into the new ZIP.
- For modified XML section files, write the updated XML content instead of the original.
- Use `ZIP_DEFLATED` compression to match the original archive.

### 7. Validate the output
- Re-open `/root/inventory_report_ready.hwpx` as a ZIP and verify it's a valid archive.
- Extract and scan ALL XML section files for any remaining `{{` or `}}` patterns. There must be NONE.
- Verify that the JSON values appear in the XML content.
- Verify that Korean labels and static note text are still present and unchanged.
- Verify empty paragraphs are preserved (count paragraphs before and after should match).
- Run the test suite if available: `cd /root && python -m pytest test_output.py -v`

### Important Notes
- Do NOT skip any placeholder — every single `{{...}}` must be replaced.
- Do NOT modify paragraphs that don't contain placeholders.
- Do NOT add or remove paragraph elements.
- Handle namespace prefixes carefully when searching for elements (use namespace maps from the XML root).
- If `lxml` is not available, fall back to `xml.etree.ElementTree` with proper namespace handling.

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