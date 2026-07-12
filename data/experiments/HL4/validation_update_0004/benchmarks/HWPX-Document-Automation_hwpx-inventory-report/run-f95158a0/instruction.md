# Task Instruction

Complete the inventory status report by replacing all `{{...}}` placeholders in the HWPX template with values from the JSON data file, then save the result as a valid HWPX package.

## Steps

1. **Inspect the workspace**: List files in the task directory to locate `inventory_report_template.hwpx` and `inventory_data.json`.

2. **Read the JSON data**: Load `inventory_data.json` and examine all key-value pairs. These are the replacement values for the placeholders.

3. **Unpack the HWPX template**: A `.hwpx` file is a ZIP archive. Extract it to a temporary directory (e.g., `/tmp/hwpx_work/`). List all files inside to understand the package structure.

4. **Identify the main content XML**: The primary document content is typically in `Contents/section0.xml` (or similar). Find all XML files that may contain `{{...}}` placeholders by grepping recursively.

5. **Process each XML file containing placeholders**:
   a. Parse the XML file.
   b. For each `<hp:p>` paragraph element, concatenate all text content from its descendant `<hp:t>` (text run) elements to reconstruct the full paragraph text. This is critical because HWPX may fragment a single `{{placeholder}}` across multiple `<hp:t>` nodes.
   c. Check if the concatenated text contains any `{{...}}` pattern.
   d. If it does, perform all `{{key}}` → value replacements using the JSON data.
   e. Replace the paragraph's text runs: clear all existing `<hp:t>` text content, then set the first `<hp:t>` element's text to the fully replaced string (or create a single run if needed). Remove text from subsequent `<hp:t>` elements (set to empty string).
   f. **Remove `<hp:linesegarray>` elements** from any modified paragraph. These are layout-cache elements that cause overlapping characters if stale.
   g. Preserve all other elements, attributes, Korean labels, static note lines, and empty paragraphs unchanged.
   h. Write the modified XML back to disk, preserving the XML declaration and encoding.

6. **Verify no placeholders remain**: Grep all extracted files for `{{` to confirm zero remaining placeholders.

7. **Repack the HWPX as a valid ZIP**:
   a. The `mimetype` file **must** be the first entry in the ZIP archive and must be stored **uncompressed** (compression method `ZIP_STORED`).
   b. All other files should be added with `ZIP_DEFLATED` compression.
   c. Save the result to `/root/inventory_report_ready.hwpx`.

8. **Validate the output**:
   a. Confirm `/root/inventory_report_ready.hwpx` exists and is a valid ZIP.
   b. Open it and verify `mimetype` is the first entry and is uncompressed.
   c. Extract and grep for `{{` — expect zero matches.
   d. Verify the content XML parses without errors.
   e. If a test suite exists (e.g., `test_output.py`), run `pytest` to confirm all tests pass.

## Key Technical Details
- HWPX text fragmentation: `{{report_date}}` might be split as `{{repo` in one `<hp:t>` and `rt_date}}` in another. Always concatenate all text in a paragraph before matching.
- `<hp:linesegarray>` removal: Only remove from paragraphs where text was modified. Leave untouched paragraphs as-is.
- Empty paragraphs (no text or only whitespace) must be preserved for document spacing.
- The ZIP mimetype-first requirement is essential for HWPX validity — use Python's `zipfile` module with explicit ordering.
- Use Python for all processing (xml.etree.ElementTree or lxml for XML, zipfile for archive handling).

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