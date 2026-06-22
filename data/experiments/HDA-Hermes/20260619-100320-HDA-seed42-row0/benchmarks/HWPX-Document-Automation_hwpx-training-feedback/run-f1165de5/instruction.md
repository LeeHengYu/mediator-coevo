# Task Instruction

Execute the following steps in a single Python script to fill in the training feedback HWPX template and save the result.

### Step 1 – Inspect inputs
- Read and print `training_feedback.json` so you know every key and value.
- List the files inside `training_feedback_template.hwpx` (it is a ZIP archive) so you know the package structure (look for `mimetype`, `Contents/`, etc.).
- Extract the archive to a temp directory (e.g., `/tmp/hwpx_work`).
- Find the main content XML file (likely `Contents/content.hpf` or `Contents/section0.xml` – check both). Print its full text so you can see every `{{...}}` placeholder and the XML namespace declarations.

### Step 2 – Plan replacements from JSON
Map each `{{...}}` token to the corresponding JSON value, applying these transformations:
1. **참석자수** – strip any non-digit characters; write digits only (e.g., `"25명"` → `"25"`, `"25"` stays `"25"`).
2. **만족도** – rewrite as `"<score>점 (5.0점 만점)"` where `<score>` is the numeric value from JSON (e.g., `4.5` → `"4.5점 (5.0점 만점)"`).
3. **종합의견 / overall opinion** – append ` 후속 심화반 검토 요망.` (with a leading space) after the JSON-provided comment text.
4. All other placeholders – substitute the JSON value as-is.

### Step 3 – Robust placeholder replacement
Because HWPX editors sometimes split `{{placeholder}}` across multiple `<hp:t>` or `<hc:t>` text-run tags:
1. Read the raw XML as a UTF-8 string.
2. First, do a regex consolidation pass: collapse sequences of adjacent closing/opening text-run tags that together form a `{{...}}` token. Specifically, find patterns where `{{`, the key name, and `}}` are split across tags, and merge them into a single tag's text.
3. Then perform simple `str.replace` for each `{{key}}` → value.
4. After all replacements, assert that no `{{` or `}}` remain anywhere in the XML string. If any do, print them and abort so you can debug.

### Step 4 – Remove stale layout cache from modified paragraphs
Parse the resulting XML string with `lxml.etree` (register all namespaces found in the root element first, so prefixes are preserved on output).
- For every `<hp:p>` element (or whatever paragraph tag the document uses), check whether its serialized text content matches any of the replacement values (i.e., the paragraph was modified). For safety, simply remove ALL `<hp:lineSegArray>` elements (and any `<hp:linesegarray>` case variants) throughout the document. This is the proven safe approach for HWPX.
- Re-serialize the XML tree back to a UTF-8 string, preserving the XML declaration.

### Step 5 – Repackage the HWPX
1. Write the modified XML back to the correct file path inside the extracted directory.
2. Also check if there are other XML files in the package (e.g., other section files). If any contain `{{...}}` placeholders, apply the same replacement and cache-clearing to them.
3. Re-zip the directory into `/root/training_feedback_ready.hwpx`:
   - `cd` into the extracted root directory so paths are relative.
   - If a `mimetype` file exists, add it first with `ZIP_STORED` (no compression).
   - Add all remaining files with `ZIP_DEFLATED`.
   - This preserves the HWPX package structure.

### Step 6 – Validate
1. Open `/root/training_feedback_ready.hwpx` as a ZIP and list its contents – confirm structure matches the original.
2. Read back every XML file from the new archive and confirm:
   - No `{{` or `}}` tokens remain.
   - No `<hp:lineSegArray>` elements remain.
   - The 참석자수 value is digits only.
   - The 만족도 value matches the `X.X점 (5.0점 만점)` pattern.
   - The overall opinion sentence ends with `후속 심화반 검토 요망.`
3. Print "VALIDATION PASSED" if all checks succeed.

Write and run the complete Python script. If any step fails, inspect the actual file contents and fix before proceeding.

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