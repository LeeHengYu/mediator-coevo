# Task Instruction

Complete the following steps in order to fill in the training feedback HWPX template and produce the final document.

## 1. Understand the HWPX format
A `.hwpx` file is a ZIP-based ODF-like package used by Hancom Office. Inside the ZIP you will find XML files (typically under `Contents/`) that contain the document body. The placeholders `{{...}}` will appear in these XML files.

## 2. Inspect the workspace
```bash
find /root -maxdepth 3 -type f | head -60
```
Locate `training_feedback_template.hwpx` and `training_feedback.json`. Read the JSON file:
```bash
cat /root/training_feedback.json
```
(or wherever it is located)

## 3. Extract and inspect the template
```bash
mkdir -p /tmp/hwpx_work
cp /root/training_feedback_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_extracted
find template_extracted -type f
```
Then read every XML file to find all `{{...}}` placeholders:
```bash
grep -rn '{{' template_extracted/
```
Also look at the full content of each XML file that contains placeholders so you understand the surrounding XML structure.

## 4. Read the JSON values
Parse the JSON file carefully. Note every key and its value. You will need to map each `{{key}}` placeholder to its corresponding JSON value.

## 5. Apply replacements with these specific rules
For each XML file that contains placeholders, perform the following replacements:

- **General rule**: Replace every `{{key}}` with the corresponding value from the JSON.
- **`참석자수` (attendee count)**: Convert the value to digits only. For example, if the JSON has `"참석자수": "32명"`, write `32` (strip any non-digit characters). If it is already a plain number, keep it as digits.
- **`만족도` (satisfaction)**: Rewrite the value in the format `X.X점 (5.0점 만점)` where X.X is the numeric score from the JSON. For example, if JSON has `"만족도": 4.5` or `"만족도": "4.5"`, write `4.5점 (5.0점 만점)`.
- **Overall opinion / 종합의견**: Find the placeholder for the overall opinion/comment. After inserting the JSON value, append ` 후속 심화반 검토 요망.` (with a space before it) at the end of that text. Make sure both the original comment and the appended sentence are in the same text run/paragraph.
- **Korean labels and static note lines**: Do NOT modify any existing Korean label text or static note lines. Only replace `{{...}}` placeholders.

## 6. Remove stale layout-cache elements
After modifying paragraph text in the XML, look for layout-cache elements associated with those paragraphs. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` elements (or similar caching elements like `<lineseg>` entries) within or near the modified `<hp:p>` paragraphs. **Delete the entire `<hp:linesegarray>...</hp:linesegarray>` element** (or equivalent layout cache block) from any paragraph whose text content you changed. This prevents overlapping characters when the document is opened. Do NOT remove these elements from paragraphs you did not modify.

To identify the exact element names:
```bash
grep -i 'lineseg\|lineSeg\|LINESEG\|layoutcache\|LayoutCache' template_extracted/Contents/*.xml
```
Remove matching elements only in modified paragraphs.

## 7. Verify no placeholders remain
```bash
grep -rn '{{' template_extracted/
```
This must return zero results. If any `{{...}}` remain, fix them.

## 8. Verify Korean labels and static notes are intact
Spot-check that Korean label text (like column/row headers) has not been altered.

## 9. Repackage the HWPX
Repackage the modified files back into a valid ZIP with the `.hwpx` extension:
```bash
cd /tmp/hwpx_work/template_extracted
zip -r /root/training_feedback_ready.hwpx . -x '*.DS_Store'
```
Note: Use `zip -r` from inside the extracted directory root so that the internal paths are correct (e.g., `Contents/...` not `template_extracted/Contents/...`).

## 10. Final validation
```bash
# Confirm it's a valid ZIP
file /root/training_feedback_ready.hwpx
unzip -l /root/training_feedback_ready.hwpx | head -30

# Confirm no placeholders
unzip -p /root/training_feedback_ready.hwpx | grep -c '{{'
# Must be 0

# Confirm the file exists at the required path
ls -la /root/training_feedback_ready.hwpx
```

## Critical constraints
- The output file MUST be at `/root/training_feedback_ready.hwpx`.
- It MUST be a valid ZIP (hwpx package) with the same internal structure as the original.
- Zero `{{...}}` placeholders may remain anywhere in any file inside the package.
- Layout cache elements (`linesegarray` or similar) MUST be removed from modified paragraphs only.
- All Korean labels and static note text must be preserved exactly.
- The `참석자수` value must be digits only (no units like 명).
- The `만족도` value must follow the exact format: `X.X점 (5.0점 만점)`.
- The overall opinion must end with `후속 심화반 검토 요망.` appended after the JSON comment value.

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