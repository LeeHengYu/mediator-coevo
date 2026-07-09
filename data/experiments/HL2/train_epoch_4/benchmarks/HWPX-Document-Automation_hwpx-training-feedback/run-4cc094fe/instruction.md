# Task Instruction

Complete the following task to fill in a training feedback HWPX document from JSON data.

## Goal
Fill in `training_feedback_template.hwpx` using values from `training_feedback.json`, saving the result to `/root/training_feedback_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `training_feedback.json` to understand all available keys and values.
- Extract `training_feedback_template.hwpx` (it's a ZIP archive) to a temporary directory (e.g., `/tmp/hwpx_work/`).
- List the extracted contents. Identify all XML files that may contain `{{...}}` placeholders — typically files under `Contents/` such as `section0.xml` (or similar). Check ALL XML files, not just one.
- Print the raw XML content of each file that contains `{{` to understand placeholder locations and any fragmentation.

### 2. Build the replacement map from JSON
- Load the JSON file into a Python dict.
- For the key `참석자수`: extract digits only from the value (e.g., '32명' → '32').
- For the key `만족도`: reformat as `{score}점 (5.0점 만점)` style. For example, if the JSON value is `4.5` or `4.5/5.0`, output `4.5점 (5.0점 만점)`.
- For the overall opinion/comment field (종합의견 or similar): append ` 후속 심화반 검토 요망.` after the JSON-provided comment text. Make sure there's a space before the appended sentence if the original doesn't end with one.
- All other keys map directly to their JSON values.

### 3. Replace placeholders handling XML fragmentation
For each XML file containing placeholders:
- Read the entire XML content as a string.
- **Critical**: Placeholders like `{{교육명}}` may be fragmented across multiple XML tags, e.g., `<hp:t>{</hp:t><hp:t>{교육명</hp:t><hp:t>}}</hp:t>` or similar splits. To handle this:
  - First, collect all text content within each `<hp:p>` paragraph element by concatenating all `<hp:t>` text nodes.
  - Check if the concatenated text contains any `{{...}}` pattern.
  - If it does, perform a reconstruction approach: use regex to find and replace placeholders even when XML tags intervene between the `{{` and `}}` markers.
  - A proven approach: use a regex like `\{\{(?:<[^>]*>)*([^}](?:<[^>]*>|[^}])*)(?:<[^>]*>)*\}\}` or similar to match `{{key}}` patterns that may have XML tags interspersed. Alternatively, strip all XML tags from the paragraph's inner content to find placeholders, then rebuild the paragraph with a single `<hp:t>` containing the replaced text.
  - The simplest reliable method: for each `<hp:p>` block, extract all `<hp:t>` text, concatenate, perform replacements on the concatenated string, then replace all `<hp:t>...</hp:t>` sequences within that `<hp:p>` with a single `<hp:t>` containing the final text. Preserve the attributes of the first `<hp:t>` tag if any.

### 4. Remove layout cache from modified paragraphs
- For every `<hp:p>` paragraph that was modified (had placeholder replacements), remove any `<hp:lineSegArray>...</hp:lineSegArray>` element within it. Use regex: `<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>` with `re.DOTALL` flag.
- This prevents overlapping character rendering when the document is opened.

### 5. Validate no placeholders remain
- After all replacements, scan the entire XML content for any remaining `{{` or `}}` patterns.
- If any remain, investigate and fix them before proceeding.
- Print confirmation that no placeholders remain.

### 6. Write modified XML back and repack HWPX
- Write the modified XML content back to the extracted file(s).
- Repack the entire extracted directory into a ZIP file at `/root/training_feedback_ready.hwpx`.
- Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression.
- Walk the extracted directory and add each file with the correct relative archive path (no leading directory prefix — the archive root should contain `[Content_Types].xml`, `Contents/`, etc. directly).

### 7. Final validation
- Open the output file as a ZIP and list its contents to confirm it's a valid archive.
- Read back the XML file(s) that were modified and print them to confirm:
  - No `{{...}}` placeholders remain
  - 참석자수 is digits only
  - 만족도 is in the `X.X점 (5.0점 만점)` format
  - The overall opinion ends with `후속 심화반 검토 요망.`
  - Korean labels and static note lines are unchanged
  - No `<hp:lineSegArray>` elements exist in modified paragraphs

## Important Notes
- Do NOT modify Korean labels or static instructional text — only replace `{{...}}` placeholders.
- The output must be a valid HWPX (ZIP) package.
- Handle ALL XML files in the package that contain placeholders, not just the first one found.
- Use Python for the entire workflow.

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