# Task Instruction

You must produce the file `/root/training_feedback_ready.hwpx` by filling in the HWPX template with values from the JSON data file.

## Steps

1. **Locate source files.** List the working directory and find `training_feedback_template.hwpx` and `training_feedback.json`. Read the JSON file to understand all key-value pairs.

2. **Inspect the HWPX template.** The `.hwpx` file is a ZIP archive. Extract it to a temporary directory (e.g., `/tmp/hwpx_work/`). List all entries. The main content file to edit is typically `Contents/section0.xml`. Read that file fully.

3. **Identify all `{{...}}` placeholders** in `section0.xml` (and check any other XML files for placeholders too, just in case). Map each placeholder to the corresponding JSON key.

4. **Apply replacements with the following special rules:**
   - **`참석자수`**: Extract digits only from the JSON value (e.g., if the value is `"32명"`, write `"32"`). Use a regex like `re.sub(r'[^0-9]', '', value)` to strip non-digit characters.
   - **`만족도`**: Reformat as `"X.X점 (5.0점 만점)"` where X.X is the numeric score from the JSON. For example, if the JSON has `4.5`, write `"4.5점 (5.0점 만점)"`.
   - **Overall-opinion / 종합의견 field**: After substituting the JSON comment value, append ` 후속 심화반 검토 요망.` (with a space before it) at the end of that sentence.
   - **All other placeholders**: Replace with the literal JSON value.

5. **Remove stale layout caches.** For every `<hp:p>` paragraph element whose text content you modified, remove any `<hp:lineSegArray>` child element (and its descendants) from that paragraph. This prevents overlapping-character rendering issues in Hancom Office. Use an XML parser (e.g., `lxml.etree`) with proper namespace handling to do this reliably.

6. **Validate the result:**
   - Parse the modified XML to confirm it is well-formed.
   - Search the entire XML string for any remaining `{{` — there must be none.
   - Confirm all expected JSON values appear in the XML text content.

7. **Repackage the HWPX.** Write the modified `section0.xml` (and any other changed files) back into the extracted directory. Then re-create a ZIP file at `/root/training_feedback_ready.hwpx` containing all the original entries. Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression. Make sure the directory structure inside the ZIP matches the original exactly (no extra root folder).

8. **Final verification.** Open `/root/training_feedback_ready.hwpx` with `zipfile.ZipFile` to confirm it is a valid ZIP. Read back the section XML from inside the ZIP and verify no `{{` placeholders remain and that the special formatting rules were applied correctly.

## Important Notes
- Keep all Korean labels and any static note lines unchanged.
- Do not modify paragraphs you didn't need to change.
- Only remove `<hp:lineSegArray>` from paragraphs where you actually changed text.
- The output path must be exactly `/root/training_feedback_ready.hwpx`.

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