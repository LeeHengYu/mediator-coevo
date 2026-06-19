# Task Instruction

Complete the following steps in order to fill in the training feedback sheet and produce `/root/training_feedback_ready.hwpx`.

## 1. Explore the workspace
```bash
find /root -maxdepth 2 -type f | head -40
```
Identify the locations of `training_feedback_template.hwpx` and `training_feedback.json`.

## 2. Understand the HWPX package structure
A `.hwpx` file is a ZIP archive (ODF-like package). Unzip the template into a working directory:
```bash
mkdir -p /root/hwpx_work
cp /root/training_feedback_template.hwpx /root/hwpx_work/template.zip
cd /root/hwpx_work
unzip template.zip -d template_contents
```
List all files inside:
```bash
find template_contents -type f
```

## 3. Read the JSON data
```bash
cat /root/training_feedback.json
```
Note every key-value pair. You will need all of them.

## 4. Find all XML files containing `{{` placeholders
```bash
grep -rl '{{' template_contents/
```
For each file found, display its full contents so you can see every placeholder and the surrounding XML structure.

## 5. Understand the placeholder-to-value mapping
Read every `{{...}}` token in the XML files. Match each to the corresponding JSON key. Pay special attention to:
- `참석자수` – must be converted to digits only (e.g., if JSON says "25명" write "25"; if it says 25 as a number, write "25").
- `만족도` – must be rewritten in the format `X.X점 (5.0점 만점)` where X.X is the numeric score from JSON.
- The overall-opinion / 종합의견 field – after inserting the JSON comment value, append a space then `후속 심화반 검토 요망.` so the final text is `<original comment> 후속 심화반 검토 요망.`

## 6. Perform replacements carefully
Using Python (preferred for precision), write a script that:
1. Reads each XML file that contains `{{`.
2. Loads the JSON data.
3. Replaces every `{{placeholder}}` with the correct value, applying the three special transformations above.
4. **Removes stale layout-cache elements**: For every `<hp:linesegarray>` (or similar `linesegarray` / `lineSegArray` element) that is a child/descendant of any paragraph (`<hp:p>`) whose text content was modified, delete that entire `linesegarray` element. This prevents overlapping-character rendering. Inspect the actual XML tag names first — they may be namespaced differently (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, etc.). Remove whichever layout-cache elements exist in modified paragraphs.
5. Writes the modified XML back, preserving encoding (UTF-8) and XML declaration.
6. Verifies no `{{` remains anywhere in any file in the package.

Key implementation notes:
- Use `xml.etree.ElementTree` or `lxml` for XML parsing so you don't corrupt tags.
- Register all namespaces found in the file before parsing so they are preserved on write-back.
- If the placeholder text is split across multiple XML inline elements (e.g., `<hp:t>{{</hp:t><hp:t>name}}</hp:t>`), you must handle that: concatenate adjacent text runs, perform replacement, then write back. Inspect the raw XML carefully to see if this occurs.
- Alternatively, if placeholders are each within a single `<hp:t>` or `<t>` element, simple string replacement on that element's text is fine.
- After replacement, do a raw-text grep for `{{` to confirm zero remaining placeholders.

## 7. Re-pack the HWPX archive
```bash
cd /root/hwpx_work/template_contents
zip -r /root/training_feedback_ready.hwpx . -x '*.DS_Store'
```
Use `zip -r` from inside the extracted directory root so paths are correct (mimetype, META-INF/, Contents/, etc.).

## 8. Validate the output
1. Confirm the output file exists and is a valid ZIP:
```bash
unzip -t /root/training_feedback_ready.hwpx
```
2. Grep the entire archive for any remaining `{{`:
```bash
mkdir -p /root/hwpx_verify
cd /root/hwpx_verify
unzip /root/training_feedback_ready.hwpx -d verify
grep -r '{{' verify/ || echo 'NO PLACEHOLDERS REMAIN - OK'
```
3. Spot-check that the special fields are correct:
   - `참석자수` is digits only (no 명, no other text)
   - `만족도` matches `X.X점 (5.0점 만점)` format
   - 종합의견 ends with `후속 심화반 검토 요망.`
   - Korean labels and the static note line are unchanged
   - No `linesegarray` (or equivalent) elements remain in paragraphs whose text was modified

## Critical constraints
- Do NOT remove or weaken any part of the original template structure except the placeholders and stale layout caches in modified paragraphs.
- Keep all Korean labels and static note lines exactly as they are.
- The final file must be at exactly `/root/training_feedback_ready.hwpx`.
- The file must be a valid ZIP (HWPX package).

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