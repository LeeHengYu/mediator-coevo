# Task Instruction

Complete the HWPX document automation task. Follow these steps precisely:

## Step 1: Inspect the input files
- Read `/root/project_proposal.json` to understand the data structure and values.
- Examine the HWPX template: `ls /root/project_proposal_template.hwpx` — note this is a ZIP archive.
- Unpack it to a temp directory to inspect the XML contents: `mkdir /tmp/hwpx_work && cd /tmp/hwpx_work && unzip /root/project_proposal_template.hwpx`
- List all files in the unpacked archive.
- Read the XML section files (likely under `Contents/` — look for files like `section0.xml` or similar) to identify all `{{...}}` placeholders and understand the XML structure.

## Step 2: Write a Python transformation script
Create a Python script `/tmp/transform.py` that does the following:

1. **Load the JSON data** from `/root/project_proposal.json`. Flatten nested structures so that keys like `project.title` map to `{{project.title}}` placeholders. Handle all nesting levels.

2. **Budget normalization**: For any budget value, remove commas but keep the leading currency symbol (e.g., `₩1,500,000` → `₩1500000`).

3. **Process all XML files** in the unpacked HWPX directory (find all `.xml` files recursively):
   - For each XML file, read its content as text.
   - Replace all `{{...}}` placeholders with the corresponding JSON values.
   - **Month span calculation**: For lines containing phase markers (단계1, 단계2, 단계3), parse the date range already present in that line (e.g., `2025.01 ~ 2025.03`), calculate the number of months between start and end dates (inclusive of both endpoints — count the months), and append ` (N개월)` after the phase line content. Use regex to find date patterns like `YYYY.MM ~ YYYY.MM` and compute: `(end_year - start_year) * 12 + (end_month - start_month + 1)` for inclusive month count, or `(end_year - start_year) * 12 + (end_month - start_month)` — verify against expected values: 단계1→3개월, 단계2→3개월, 단계3→1개월.
   - **Remove stale layout caches**: After modifying any paragraph's text content, remove `<linesegarray>...</linesegarray>` and `<lineSeg .../>` elements from that paragraph. Use regex or XML parsing. This is critical — without this, the document will render with overlapping characters.

4. **Verify no `{{...}}` placeholders remain** in any XML file after transformation.

5. **Repackage the HWPX archive**:
   - Use Python's `zipfile` module to create `/root/project_proposal_ready.hwpx`.
   - Add all files from the unpacked directory, preserving the relative path structure.
   - Use `ZIP_DEFLATED` compression.
   - Important: When creating the ZIP, the paths inside must match the original archive structure (no leading directory prefix from the temp extraction path).

## Step 3: Execute the script
Run: `cd /tmp/hwpx_work && python3 /tmp/transform.py`

## Step 4: Validate the output
1. Verify `/root/project_proposal_ready.hwpx` exists and is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/project_proposal_ready.hwpx'); z.testzip(); print('Valid ZIP'); z.close()"`
2. Unpack to a verification directory and grep for any remaining `{{`: `mkdir /tmp/verify && cd /tmp/verify && unzip /root/project_proposal_ready.hwpx && grep -r '{{' . || echo 'No placeholders remaining'`
3. Check that month spans appear correctly: `grep -r '개월' /tmp/verify/`
4. Check that budget values have no commas: verify the budget field in the XML.
5. Run the verifier if available: `cd /root && python -m pytest test_output.py -v` or similar.

## Key Details to Watch
- The JSON may have nested keys — flatten them with dot notation for placeholder matching.
- The month span text must be appended within the same XML text run/element as the phase line, not as a separate paragraph.
- linesegarray/lineSeg removal should target only modified paragraphs to minimize changes, but removing from all paragraphs is also acceptable.
- Ensure Korean text and static note lines are preserved exactly.
- The HWPX internal structure must be preserved exactly (mimetype file, Contents/, etc.).

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