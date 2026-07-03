# Task Instruction

Complete the following steps in order to fill in the training feedback sheet and produce `/root/training_feedback_ready.hwpx`.

## 1. Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (ODF-like structure used by Hancom Office). The main body text is typically in `Contents/section0.xml` (or similar). You will need to:
- Unzip the template
- Edit the XML content to replace placeholders
- Re-zip into a valid `.hwpx` package

## 2. Inspect the input files
1. Read `/root/training_feedback.json` to understand all available key-value pairs.
2. Copy or unzip `/root/training_feedback_template.hwpx` into a temporary working directory (e.g., `/root/hwpx_work/`):
   ```
   mkdir -p /root/hwpx_work
   cd /root/hwpx_work
   unzip /root/training_feedback_template.hwpx
   ```
3. List all extracted files to understand the package structure.
4. Search all XML files for `{{` to find every placeholder. Use: `grep -r '{{' /root/hwpx_work/`
5. Record every placeholder found and which file(s) contain them.

## 3. Understand the JSON data
Read the JSON file carefully. Note:
- The key names inside `{{...}}` in the template should correspond to JSON keys.
- `참석자수` (number of attendees): extract only the digits. If the JSON value is e.g. `"32명"`, write `32`. If it's already a number, just use the number.
- `만족도` (satisfaction): rewrite as `X.X점 (5.0점 만점)` format, where X.X is the numeric score from JSON. For example, if JSON has `4.5`, write `4.5점 (5.0점 만점)`.
- For the overall-opinion/종합의견 field: take the comment from JSON and append ` 후속 심화반 검토 요망.` at the end (with a space before 후속).

## 4. Replace placeholders in the XML
For each XML file containing `{{...}}` placeholders:
1. Read the file content.
2. Replace each `{{key}}` with the corresponding transformed value from JSON.
3. **CRITICAL**: After replacing text in any paragraph element, remove any layout-cache / char-shape-positioning / glyph-run elements that cache character positions. These are typically elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, or similar elements within `<hp:p>` paragraph tags that store pre-computed layout. Look for elements that contain pixel-position or character-width data within modified paragraphs and remove them so the document re-renders cleanly.
   - Specifically, look for `<hp:linesegarray>` or `<linesegarray>` blocks inside any `<hp:p>` whose text you changed, and delete them entirely.
4. Write the modified XML back.

## 5. Verify no placeholders remain
Run `grep -r '{{' /root/hwpx_work/` and confirm zero matches. If any remain, fix them.

## 6. Verify Korean labels and static note line are unchanged
Compare the structure to ensure you only changed placeholder values, not labels or static content.

## 7. Repackage as .hwpx
Re-create the HWPX (ZIP) archive:
```
cd /root/hwpx_work
zip -r /root/training_feedback_ready.hwpx . -x '.*'
```
IMPORTANT: The ZIP must preserve the original directory structure exactly. The `mimetype` file (if present) should ideally be stored first and uncompressed (use `zip -0` for it first, then add the rest), similar to ODF packaging. Check if a `mimetype` file exists; if so:
```
cd /root/hwpx_work
zip -0 /root/training_feedback_ready.hwpx mimetype
zip -r /root/training_feedback_ready.hwpx . -x mimetype -x '.*'
```
If no `mimetype` file exists, just zip everything normally.

## 8. Final validation
1. Verify `/root/training_feedback_ready.hwpx` exists and is a valid ZIP: `unzip -t /root/training_feedback_ready.hwpx`
2. Unzip to a temp location and grep for `{{` — must find zero occurrences.
3. Verify the transformed values are present: grep for the digit-only attendee count, the `점 (5.0점 만점)` satisfaction format, and `후속 심화반 검토 요망` in the output.
4. Confirm no original Korean labels were altered by spot-checking a few.

## Key constraints recap
- Every `{{...}}` must be replaced — none may remain.
- `참석자수`: digits only (strip any non-digit characters like 명).
- `만족도`: format as `X.X점 (5.0점 만점)`.
- Overall opinion: append ` 후속 심화반 검토 요망.` after the JSON comment.
- Korean labels and static note lines must be preserved exactly.
- Remove stale layout-cache elements (`linesegarray` or similar) from any modified paragraph.
- Output must be a valid .hwpx ZIP package at `/root/training_feedback_ready.hwpx`.

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