# Task Instruction

Complete the following task step by step.

## Goal
Fill in the training feedback template `training_feedback_template.hwpx` using values from `training_feedback.json`, and save the result to `/root/training_feedback_ready.hwpx`.

## Steps

### 1. Inspect the workspace
- List files in the current directory to locate `training_feedback_template.hwpx` and `training_feedback.json`.
- Read and display the full contents of `training_feedback.json` to understand all keys and values.

### 2. Examine the HWPX structure
- A `.hwpx` file is a ZIP archive. Unzip the template to a temporary working directory (e.g., `/tmp/hwpx_work/`).
- List all files in the extracted archive.
- Identify which XML files contain the document content (typically `Contents/section0.xml`, possibly `section1.xml`, etc.).
- Read and display the full raw XML content of each section file. Also check `Contents/header*.xml` and `Contents/footer*.xml` if they exist.

### 3. Identify all placeholders
- Search for `{{` across ALL extracted files (not just section XMLs) to find every placeholder.
- Note that placeholders may be split across multiple `<hp:t>` elements within `<hp:run>` elements (e.g., `<hp:t>{{</hp:t>`, `<hp:t>key</hp:t>`, `<hp:t>}}</hp:t>`). You must handle this.

### 4. Write a Python script to perform the replacement
Create and run a Python script that:

a) **Reads** `training_feedback.json` into a dictionary.

b) **Applies business-logic transformations** to the JSON values BEFORE substitution:
   - `참석자수`: Convert to digits only. If the JSON value is something like `스물다섯 명` or `25명`, extract/convert to just the numeric digits (e.g., `25`). Handle Korean number words if present.
   - `만족도`: Reformat as `X.X점 (5.0점 만점)` where X.X is the numeric score from the JSON. For example, if the JSON has `4.5` or `4.5/5.0`, output `4.5점 (5.0점 만점)`.
   - `종합의견` (or whatever key maps to the overall-opinion field): Append ` 후속 심화반 검토 요망.` after the provided comment value (with a space before `후속`).
   - All other values: use as-is from JSON.

c) **For each XML file that contains placeholders**, perform replacement using this robust approach:
   - Parse the XML as raw text (not with an XML parser that might alter formatting).
   - First, try to consolidate fragmented placeholders: scan for patterns where `{{`, the key name, and `}}` are split across adjacent `<hp:t>...</hp:t>` elements within the same paragraph or run group. Merge them into a single `<hp:t>{{key}}</hp:t>` element (removing the now-empty sibling `<hp:t>` elements and their parent `<hp:run>` if they become empty).
   - Then do a simple string replacement of `{{key}}` with the transformed value for each key.
   - After all replacements, verify NO `{{` or `}}` remain in the file.

d) **Remove layout-cache elements** from any paragraph (`<hp:p>`) whose text content was modified:
   - Remove `<hp:lineSegArray>...</hp:lineSegArray>` elements (and any similar layout-cache elements like `<hp:linesegarray>...</hp:linesegarray>`) from modified paragraphs.
   - This is critical to prevent overlapping/corrupted text rendering.

e) **Write back** the modified XML files to the working directory.

f) **Repackage** the working directory into a valid ZIP file saved as `/root/training_feedback_ready.hwpx`:
   - Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression.
   - Preserve the exact directory structure and all original files.
   - Make sure to include ALL files from the original archive, not just the modified ones.

### 5. Validate the output
- Unzip `/root/training_feedback_ready.hwpx` to a new temp directory.
- Read each section XML and print its text content.
- Verify:
  1. No `{{` or `}}` placeholders remain anywhere in any file.
  2. The `참석자수` value appears as digits only (e.g., `25`, not `25명` or Korean words).
  3. The `만족도` value appears in the format `X.X점 (5.0점 만점)`.
  4. The overall opinion text ends with `후속 심화반 검토 요망.`
  5. All Korean labels and static note lines are preserved unchanged.
  6. All JSON values (after transformation) appear in the output.
  7. The file is a valid ZIP archive.
- If any check fails, diagnose and fix before finalizing.

### Important Notes
- Do NOT use an XML parser that normalizes whitespace or reorders attributes; work with raw text or use `lxml` with care to preserve the original XML structure.
- The `[Content_Types].xml` and other metadata files in the HWPX package must be preserved exactly.
- If you encounter Korean numeral words for 참석자수 (e.g., 스물다섯), convert them to Arabic numerals.
- Double-check the ZIP by listing its contents and comparing against the original template's file list.

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