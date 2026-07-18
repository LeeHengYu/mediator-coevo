# Task Instruction

Complete the following task step by step.

## Goal
Fill in the training feedback template `training_feedback_template.hwpx` using values from `training_feedback.json`, and save the result to `/root/training_feedback_ready.hwpx`.

## Steps

### Step 1: Inspect the input files
1. Find the template file `training_feedback_template.hwpx` and the JSON file `training_feedback.json`. They are likely under `/root/` or a subdirectory. Use `find / -name 'training_feedback_template.hwpx' 2>/dev/null` and similarly for the JSON file.
2. Read and display the full contents of `training_feedback.json`.
3. Since `.hwpx` is a ZIP archive, list its contents (`zipfile.namelist()`). Identify the XML files inside, especially those under `Contents/` (commonly `section0.xml` or similar).
4. Extract and display the full XML content of each content XML file to find all `{{...}}` placeholders. Record every placeholder you find.

### Step 2: Understand the data mapping and transformations
Map each `{{placeholder}}` to its JSON key. Apply these transformations:
- **참석자수**: Convert to digits only (e.g., if JSON has `"25명"` or `25`, output just the number like `25`).
- **만족도**: Rewrite as `X.X점 (5.0점 만점)` format, using the numeric score from JSON (e.g., if JSON has `4.5`, output `4.5점 (5.0점 만점)`).
- **Overall opinion / 종합의견**: After inserting the JSON comment value, append ` 후속 심화반 검토 요망.` at the end (with a space before it if needed, ensuring the sentence reads naturally).
- All other placeholders: direct substitution from JSON values.
- Korean labels and the static note line must remain unchanged.

### Step 3: Write a Python script to perform the replacement
Write a Python script that:
1. Copies the template HWPX (ZIP) to the output path `/root/training_feedback_ready.hwpx`.
2. Opens the ZIP, reads each file entry.
3. For XML files containing `{{` placeholders, performs string-level replacements first:
   - Build a replacement dictionary from JSON keys to their (possibly transformed) values.
   - For 참석자수: extract digits only (use `re.sub(r'[^0-9]', '', str(value))` if it contains non-digit chars, or just `str(int(value))` if numeric).
   - For 만족도: format as `"{score}점 (5.0점 만점)"` where score is the numeric value from JSON.
   - For the overall opinion field: append ` 후속 심화반 검토 요망.` to the JSON value before substituting.
   - Replace all `{{key}}` patterns with corresponding values.
4. After string replacement, parse the XML with `lxml.etree` (or `xml.etree.ElementTree`).
5. **Remove `linesegarray` elements** from any paragraph (`<hp:p>` or similar) whose text content was modified. This is critical to prevent overlapping characters. Use namespace-aware or local-name() based XPath to find these elements. Remove them from their parent.
6. Serialize the modified XML back to bytes (with XML declaration and proper encoding).
7. Verify no `{{` remains anywhere in any XML file in the output.
8. Write all files (modified and unmodified) into a new ZIP at `/root/training_feedback_ready.hwpx`, preserving the original ZIP structure and compression.

### Step 4: Validate the output
1. Open `/root/training_feedback_ready.hwpx` as a ZIP and verify it's valid.
2. Read every XML file in the archive and confirm:
   - No `{{...}}` placeholder text remains anywhere.
   - The 참석자수 value is digits only.
   - The 만족도 value matches the `X.X점 (5.0점 만점)` format.
   - The overall opinion sentence ends with `후속 심화반 검토 요망.`
   - No `linesegarray` elements exist in paragraphs that were modified.
   - Korean labels and static note lines are preserved.
3. Print a summary of all replaced values for verification.

## Important Notes
- HWPX files use HWP-specific XML namespaces (e.g., `hp`, `hc`, etc.). Handle namespaces carefully.
- The `linesegarray` element (or elements with local name `linesegarray` or `lineSegArray`) caches layout info. It MUST be removed from every paragraph you modify, or the document will display with overlapping text.
- When rewriting the ZIP, preserve compression method and all non-XML entries (images, settings, etc.) as-is.
- Do NOT use `zipfile.open()` for writing; instead, create a new ZIP and copy all entries, replacing only the modified XML files.
- Double-check the exact placeholder names by inspecting the XML before coding the replacements.

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