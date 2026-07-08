# Task Instruction

Complete the following steps in order.

## 1 – Inspect the inputs
1. `cd /root && ls` – confirm `training_feedback_template.hwpx` and `training_feedback.json` exist.
2. `cat training_feedback.json` – read every key/value pair; note exact keys and values.
3. A `.hwpx` file is a ZIP package (like OOXML). Unzip the template into a working directory:
   ```
   mkdir -p /root/hwpx_work
   cp training_feedback_template.hwpx /root/hwpx_work/template.hwpx
   cd /root/hwpx_work
   unzip template.hwpx -d template_contents
   ```
4. List the extracted tree (`find template_contents -type f`) to understand the package structure.
5. Identify which XML files contain the document body text. Typically this is something like `Contents/section0.xml` or similar. Search for `{{` across all extracted files:
   ```
   grep -rl '{{' template_contents/
   ```
6. For every file that contains `{{`, `cat` it in full so you can see every placeholder and the surrounding XML structure.

## 2 – Understand the placeholder contract
From the JSON and the XML content, build a mapping of every `{{placeholder}}` → replacement value. Apply these transformations:
- **`참석자수`**: strip any non-digit characters; write digits only (e.g., `25명` → `25`, `25` stays `25`).
- **`만족도`**: rewrite as `X.X점 (5.0점 만점)` where X.X is the numeric score from the JSON. For example if the JSON value is `4.5` or `4.5/5.0`, output `4.5점 (5.0점 만점)`.
- **Overall-opinion / 종합의견 field**: after substituting the JSON comment value, append a space then `후속 심화반 검토 요망.` at the end of that same text run.
- All other placeholders: substitute the JSON value verbatim.

## 3 – Edit the XML files
For each XML file that contains placeholders:
1. Read the current file content.
2. Perform all `{{...}}` replacements using the mapping from step 2.
3. **Remove stale layout-cache elements**: In HWPX, layout caches are typically `<hp:linesegarray>` or `<hp:lineSegArray>` elements (and their children). For every `<hp:p>` (paragraph) element whose text you modified, delete the entire `<hp:linesegarray>...</hp:linesegarray>` (or equivalent) block inside that paragraph. This prevents overlapping-character rendering. Use a script or careful sed/Python to do this.
4. Write the modified XML back.
5. Re-read the file and confirm:
   - No `{{` remains anywhere.
   - The XML is well-formed (use `python3 -c "import xml.etree.ElementTree as ET; ET.parse('filename')"`).
   - Korean labels and the static note line are unchanged.

## 4 – Repackage the HWPX
Repackage the modified contents back into a valid ZIP (HWPX):
```bash
cd /root/hwpx_work/template_contents
zip -r /root/training_feedback_ready.hwpx . -x '*.DS_Store'
```
Note: if the original package has a `mimetype` file that must be first and uncompressed (like ODF), handle that. Check whether such a file exists; if so:
```bash
zip -0 -X /root/training_feedback_ready.hwpx mimetype
zip -r /root/training_feedback_ready.hwpx . -x mimetype -x '*.DS_Store'
```
If no mimetype file exists, the simple `zip -r` is fine.

## 5 – Final validation
1. Verify the output exists: `ls -la /root/training_feedback_ready.hwpx`
2. Verify it's a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/training_feedback_ready.hwpx'); z.testzip(); print('Valid ZIP, entries:', len(z.namelist()))"`
3. Search the entire package for any remaining placeholders:
   ```bash
   mkdir -p /root/hwpx_verify
   cd /root/hwpx_verify
   unzip /root/training_feedback_ready.hwpx -d verify
   grep -r '{{' verify/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'
   ```
4. Spot-check a few substituted values in the XML to confirm correctness (참석자수 is digits only, 만족도 is in the required format, 종합의견 ends with `후속 심화반 검토 요망.`).
5. Verify that Korean labels (column/row headers) are intact by grepping for a few expected ones.

Do NOT mark the task complete until all validation checks pass.

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