# Task Instruction

Complete the project proposal document by following these steps precisely:

1. **Inspect the workspace**: List files in the current directory and `/root/` to locate `project_proposal_template.hwpx` and `project_proposal.json`.

2. **Read the JSON data**: Cat `project_proposal.json` and note all key-value pairs. Pay special attention to the budget field (will need comma removal while keeping currency symbol like ₩).

3. **Examine the HWPX template structure**: The `.hwpx` file is a ZIP archive. Unzip it to a temporary directory (e.g., `/tmp/hwpx_template/`) and list all files. Identify the section XML files (likely under `Contents/` — files like `section0.xml`, `section1.xml`, etc.).

4. **Read all section XML files**: Cat each section XML to find all `{{...}}` placeholders and understand the document structure. Also look for the phase lines (단계1, 단계2, 단계3) with their date ranges.

5. **Write a Python script** that does the following:
   a. Copies the template HWPX to a working location.
   b. Opens it as a ZIP, reads each entry.
   c. For each XML file (especially section XMLs):
      - Replaces all `{{placeholder}}` tokens with corresponding values from the JSON file.
      - For the budget value: removes commas from the numeric portion while preserving the leading currency symbol (e.g., `₩1,200,000,000` → `₩1200000000`).
      - For phase lines containing date ranges, calculates the inclusive month span and appends it in parentheses. The formula is: `(end_year - start_year) * 12 + (end_month - start_month) + 1`. For example, if 단계1 has dates like `2025.01 ~ 2025.03`, that's 3 months → append `(3개월)`. Do this for 단계1, 단계2, and 단계3.
      - **Critical**: For any `<hp:p>` paragraph element whose text content was modified, remove all `<hp:lineSegArray>...</hp:lineSegArray>` elements within that paragraph. This prevents stale layout cache from causing overlapping characters when the document is opened.
      - Preserves all Korean labels and static note lines unchanged.
   d. Writes the modified ZIP to `/root/project_proposal_ready.hwpx`, preserving the original ZIP structure and compression.

6. **Validation**: After generating the output:
   a. Unzip `/root/project_proposal_ready.hwpx` to a temp directory.
   b. Grep recursively for `{{` — there must be zero matches.
   c. Grep for key values from the JSON to confirm they appear in the XML.
   d. Grep for `개월` to confirm the month spans were appended.
   e. Verify the budget value appears without commas.
   f. Confirm the file is a valid ZIP.

**Important implementation details for the Python script:**
- Use `zipfile` module. Read the template, process entries, write to a new zip.
- Use `re` module for replacements. Be careful with XML — the placeholder text may be split across multiple `<hp:t>` elements within a single `<hp:run>`. If simple string replacement on the raw XML doesn't find a placeholder, you may need to handle cases where XML tags are interspersed within placeholder text.
- For the month span appending: parse the date range from the line text (look for patterns like `YYYY.MM ~ YYYY.MM` or `YYYY.MM~YYYY.MM`), compute the duration, and append ` (N개월)` after the date range on the same line/paragraph.
- For lineSegArray removal: use regex like `re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', xml_content, flags=re.DOTALL)` but only within paragraphs that were actually modified. A safe approach: track which `<hp:p>` blocks had text changes and remove lineSegArray only from those. Alternatively, if simpler, remove lineSegArray from ALL paragraphs (this is safe — the viewer will recalculate all layouts).
- Ensure the output path is exactly `/root/project_proposal_ready.hwpx`.

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