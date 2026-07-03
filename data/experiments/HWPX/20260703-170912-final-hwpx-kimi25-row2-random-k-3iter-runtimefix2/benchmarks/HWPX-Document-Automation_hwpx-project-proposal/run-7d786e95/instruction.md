# Task Instruction

You must produce the file `/root/project_proposal_ready.hwpx` by completing the template HWPX document with values from the JSON file.

Step-by-step plan:

1. **Inspect the workspace.** List files in the task directory to locate `project_proposal_template.hwpx` and `project_proposal.json`. Also look for any `test_output.py` or verifier script so you understand what will be checked.

2. **Read the JSON data.** Load `project_proposal.json` and print its contents so you know every key-value pair available for substitution.

3. **Unzip the HWPX template.** HWPX is a ZIP archive. Extract it to a temporary working directory (e.g., `/tmp/hwpx_work/`). List the extracted tree.

4. **Identify XML sections to edit.** The content lives in files like `Contents/section0.xml`, `Contents/section1.xml`, etc. Read each section XML file and print its full text so you can see every `{{...}}` placeholder and every phase line.

5. **Write a Python script** (`/tmp/process_hwpx.py`) that does the following:

   a. **Load the JSON** into a dict.

   b. **Budget normalization:** For the budget value, remove commas but keep the leading currency symbol (e.g., `₩1,500,000,000` → `₩1500000000`). Store the normalized value back in the dict under its key before substitution.

   c. **Read each section XML file** as a UTF-8 string.

   d. **Replace all `{{key}}` placeholders** with the corresponding JSON value. Use a loop over all JSON keys, replacing `{{key}}` with the value. After all replacements, assert no `{{` remains in the text.

   e. **Append month spans to phase lines.** For lines containing `단계1`, `단계2`, `단계3`, compute the month span from the date range already present in that line. The date format is typically `YYYY.MM.DD ~ YYYY.MM.DD` or `YYYY-MM-DD ~ YYYY-MM-DD`. Parse both dates, compute the difference in months (round to nearest integer: `(end.year - start.year)*12 + end.month - start.month`; if there are extra days, consider rounding up), and append ` (N개월)` to the text run that contains the phase info. Based on the task description, expect: 단계1 → (3개월), 단계2 → (3개월), 단계3 → (1개월).

   f. **Remove stale layout-cache elements.** This is critical. For every `<hp:p>` paragraph element whose text content was modified (placeholder replaced or month span appended), find and remove any `<hp:linesegarray>` child element (use the HWPML namespace). This prevents overlapping characters when the document is opened. Be thorough: it's safest to remove `hp:linesegarray` from ALL paragraphs that contain any modified `<hp:run>`, or even from all paragraphs globally if simpler, since removing layout cache from unmodified paragraphs is harmless (the application regenerates it).

   g. **Write the modified XML** back to the same file paths (UTF-8, with XML declaration).

   h. **Repackage the HWPX.** Use `zipfile.ZipFile` to create `/root/project_proposal_ready.hwpx`. Walk the extracted directory and add every file, preserving the relative path structure inside the ZIP. Make sure `mimetype` (if present) is stored first and uncompressed (compression=ZIP_STORED), and all other files use ZIP_DEFLATED. This mirrors standard OPC/HWPX packaging.

6. **Run the script.** Execute `python3 /tmp/process_hwpx.py` and check for errors.

7. **Validate the output.**
   - Confirm `/root/project_proposal_ready.hwpx` exists and is a valid ZIP.
   - Unzip it to a temp location, read each section XML, and verify:
     - No `{{` remains anywhere.
     - Phase lines contain the `(N개월)` suffixes.
     - Budget value has no commas but retains currency symbol.
     - No `hp:linesegarray` elements exist in modified paragraphs.
   - If a test script exists in the task directory, run it: `cd /path/to/task && python -m pytest test_output.py -v`

8. **If any check fails**, inspect the specific failure, fix the script, and re-run.

Key pitfalls to avoid:
- **Do NOT forget to remove `hp:linesegarray`** from modified paragraphs. This was the #1 failure mode in similar HWPX tasks.
- **Namespace handling:** HWPX XML uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph` for `hp:`. When using ElementTree, register and use the correct namespace prefixes. Print the root tag of each XML to discover the actual namespace URIs.
- **Preserve Korean text and static note lines exactly.** Only modify placeholder text and phase-line suffixes.
- **When repackaging**, preserve the exact directory structure from the original ZIP.

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