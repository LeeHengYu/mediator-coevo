# Task Instruction

Complete the project proposal document by following these steps:

1. **Inspect the workspace**: List files in the task directory to find `project_proposal_template.hwpx` and `project_proposal.json`. Read `project_proposal.json` to understand all placeholder values.

2. **Understand the HWPX structure**: A `.hwpx` file is a ZIP archive containing XML files. Unzip `project_proposal_template.hwpx` to a temporary directory (e.g., `/tmp/hwpx_work/`). List the extracted contents and identify the XML files that contain document body content (typically under `Contents/` — look for files like `section0.xml` or similar).

3. **Examine the content XML files**: Read each content XML file to find `{{...}}` placeholders. Note how they appear — in HWPX files, a single placeholder like `{{project_name}}` is often split across multiple `<hp:t>` elements within a single `<hp:p>` paragraph element (e.g., `<hp:t>{{</hp:t>`, `<hp:t>project_name</hp:t>`, `<hp:t>}}</hp:t>`).

4. **Write a Python script** (`/tmp/process_hwpx.py`) that does the following:

   a. **Parse each content XML** using `lxml.etree` (or `xml.etree.ElementTree` if lxml unavailable).
   
   b. **For each `<hp:p>` paragraph element**, merge all `<hp:t>` text nodes into a single concatenated string to reconstruct any fragmented placeholders. Store the merged text.
   
   c. **Perform placeholder replacement** on the merged text using the JSON values:
      - For the budget field: remove commas from the numeric portion but keep the leading currency symbol (e.g., `₩1,500,000,000` → `₩1500000000`).
      - Replace all `{{key}}` patterns with corresponding JSON values.
   
   d. **Append month spans to phase lines**: For lines containing `단계1`, `단계2`, `단계3` with date ranges (format like `2025.01 ~ 2025.03`), calculate the inclusive month span and append it as `(N개월)`. The inclusive month count = (end_year - start_year) * 12 + (end_month - start_month) + 1. Use regex to find date patterns like `(\d{4})\.(\d{2})\s*~\s*(\d{4})\.(\d{2})` and compute accordingly.
   
   e. **After modifying a paragraph's text**: 
      - Consolidate the merged text back into a single `<hp:t>` element (remove extra `<hp:t>` siblings, keep one with the full text).
      - **Remove any `<hp:lineSegArray>` child element** from that `<hp:p>` to prevent stale layout-cache from causing overlapping characters when the document is opened.
   
   f. **Verify no `{{` or `}}` remain** anywhere in the processed XML text content.
   
   g. **Write the modified XML back** to the same file path, preserving XML declaration and encoding.

5. **Repackage the HWPX**: Re-zip the modified contents from `/tmp/hwpx_work/` into `/root/project_proposal_ready.hwpx`. Use Python's `zipfile` module. Make sure to preserve the directory structure exactly as it was in the original archive (same member names, no extra nesting). Use `ZIP_DEFLATED` compression.

6. **Validate the output**:
   - Confirm `/root/project_proposal_ready.hwpx` exists and is a valid ZIP.
   - Unzip it to a separate temp location and grep/search all XML content for any remaining `{{` — there should be none.
   - Print the text content of the main section XML to visually confirm placeholders are replaced, month spans are appended, budget is formatted correctly, and Korean labels are intact.
   - Check that the file can be opened as a ZIP without errors.

7. **Run the verifier** if a test script exists (look for `test_output.py` or similar in the task directory): `cd /path/to/task && python -m pytest test_output.py -v`

Key details to watch for:
- Namespace handling: HWPX XML uses namespaces (e.g., `hp:` prefix). Make sure your XML parsing respects these namespaces when finding elements.
- The ZIP repackaging must use the exact same archive member paths as the original.
- Do not modify any Korean label text or the static note line — only replace placeholders and append month spans.
- Ensure the budget normalization regex handles the currency symbol correctly (it might be ₩ or another symbol — check the JSON).
- When merging `<hp:t>` nodes, preserve the `<hp:run>` structure: keep one `<hp:run>` with one `<hp:t>` containing the full merged text, and handle the run-level properly.

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