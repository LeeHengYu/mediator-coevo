# Task Instruction

Complete the following steps in order to fill in the training feedback sheet and produce `/root/training_feedback_ready.hwpx`.

## 1. Explore the workspace
```bash
find /root -maxdepth 2 -type f | head -40
```
Identify the locations of `training_feedback_template.hwpx` and `training_feedback.json`.

## 2. Understand the HWPX structure
HWPX files are ZIP packages. Unzip the template into a working directory:
```bash
mkdir -p /root/hwpx_work
cp /root/training_feedback_template.hwpx /root/hwpx_work/template.zip
cd /root/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
List every file in the package to understand the structure.

## 3. Read the JSON data
```bash
cat /root/training_feedback.json
```
Note every key-value pair. You will need all of them.

## 4. Inspect all XML content files for placeholders
Search for `{{` across all extracted files:
```bash
grep -r '{{' /root/hwpx_work/template_contents/
```
Also read the full content of each XML file that contains placeholders (likely under `Contents/` — files like `section0.xml`, `content.hpf`, etc.):
```bash
find /root/hwpx_work/template_contents -name '*.xml' -exec echo '=== {} ===' \; -exec cat {} \;
```
Carefully note every `{{placeholder_name}}` and which file it appears in.

## 5. Perform substitutions with the required transformations
Write a Python script that:

a. Loads `training_feedback.json`.

b. For each XML file that contains `{{...}}` placeholders, reads the file and performs replacements:
   - For every `{{key}}` placeholder, replace it with the corresponding value from the JSON.
   - **`참석자수` (attendee count):** Convert the value to digits only (strip any non-digit characters like '명' or other Korean unit suffixes). For example, if the value is `32명`, write `32`.
   - **`만족도` (satisfaction):** Rewrite the value in the format `X.X점 (5.0점 만점)`, where X.X is the numeric score from the JSON. For example, if the JSON has `4.5` or `4.5/5.0`, output `4.5점 (5.0점 만점)`.
   - **Overall opinion / 종합의견:** Find the placeholder for the overall comment. After inserting the JSON value, append ` 후속 심화반 검토 요망.` (with a space before it) to the end of that sentence.
   - All other placeholders: substitute the JSON value directly.

c. **Remove stale layout-cache elements** from any paragraph (`<hp:p>` or similar) whose text content was modified. Specifically, look for elements like `<hp:linesegarray>`, `<lineseg>`, `<hp:lineSegArray>`, or similar layout-cache/line-segment elements within modified paragraphs and remove them entirely. This prevents overlapping characters when the document is opened. Use an XML parser (e.g., `lxml` or `xml.etree.ElementTree`) for this step — do NOT use plain string replacement for removing XML elements.

d. Verify that no `{{` or `}}` remains in any file.

e. Write the modified XML files back.

## 6. Repackage as HWPX
Repackage the modified contents back into a valid HWPX (ZIP) file. HWPX uses the same packaging as OPC/ZIP. The `mimetype` file (if present) should be stored first without compression (like in ODF):
```bash
cd /root/hwpx_work/template_contents
# If mimetype exists, add it first uncompressed
if [ -f mimetype ]; then
  zip -0 -X /root/training_feedback_ready.hwpx mimetype
  zip -r -X /root/training_feedback_ready.hwpx . -x mimetype
else
  zip -r -X /root/training_feedback_ready.hwpx .
fi
```
Alternatively, use Python's `zipfile` module to replicate the original ZIP structure exactly (preserving compression methods from the original archive).

## 7. Validate the output
```bash
# Confirm it's a valid ZIP
unzip -t /root/training_feedback_ready.hwpx

# Confirm no placeholders remain
unzip -p /root/training_feedback_ready.hwpx | grep -c '{{'
# Expected: 0

# Spot-check the substituted values
unzip -p /root/training_feedback_ready.hwpx | grep -o '후속 심화반 검토 요망'
unzip -p /root/training_feedback_ready.hwpx | grep -o '점 (5.0점 만점)'

# Confirm the file exists at the right path
ls -la /root/training_feedback_ready.hwpx
```

## Key constraints — do NOT violate these:
- Every `{{...}}` placeholder must be replaced. Zero may remain.
- Korean labels and the static note line must be unchanged.
- `참석자수` must be digits only (no Korean unit suffixes).
- `만족도` must follow the exact format: `X.X점 (5.0점 만점)`.
- The overall opinion must end with `후속 심화반 검토 요망.` appended after the JSON comment.
- Any paragraph with modified text must have its layout-cache / lineseg elements removed.
- The output must be a valid `.hwpx` ZIP package at `/root/training_feedback_ready.hwpx`.

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