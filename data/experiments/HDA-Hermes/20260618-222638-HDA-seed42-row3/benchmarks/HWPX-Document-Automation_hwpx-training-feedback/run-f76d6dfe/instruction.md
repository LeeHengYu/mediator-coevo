# Task Instruction

Execute the following steps to fill in the training feedback HWPX template:

1. **Inspect the workspace.** List files in the task directory to locate `training_feedback_template.hwpx` and `training_feedback.json`. Read the JSON file to understand all available keys and values.

2. **Explore the HWPX structure.** The `.hwpx` file is a ZIP archive. Unzip it to a temporary working directory (e.g., `/tmp/hwpx_work/`). List all files inside. Identify the section XML file(s) that contain the document body text (typically `Contents/section0.xml` or similar). Read the section XML to find all `{{...}}` placeholders and understand the document structure.

3. **Write and run a Python script** that does the following:

   a. **Load the JSON** from `training_feedback.json`.

   b. **Parse the section XML** using `lxml.etree`.

   c. **Walk every text node** in the XML tree (both `.text` and `.tail` of every element). For each text node containing a `{{...}}` pattern, replace it with the corresponding JSON value, applying these transformations:
      - For `참석자수`: extract digits only (e.g., if the JSON says `"32명"`, write `32`; if it says `32`, write `32`).
      - For `만족도`: rewrite as `X.X점 (5.0점 만점)` where X.X is the numeric score from JSON.
      - For the overall-opinion / 종합의견 field: after substituting the JSON comment value, append ` 후속 심화반 검토 요망.` (with a space before it) at the end of that text.
      - All other placeholders: substitute the JSON value directly.

   d. **Verify no `{{` or `}}` remains** anywhere in the serialized XML. If any remain, log them and abort.

   e. **Remove layout-cache elements.** For every `<hp:p>` (paragraph) element whose text content was modified, find and remove any child `<hp:lineSegArray>` element (and its descendants). This prevents stale glyph-position data from causing overlapping characters when the file is opened in Hancom Office. Use the HWPX namespace (`urn:hancom:hwpx:...` or whichever namespace prefix `hp` maps to in the file). If the namespace differs, detect it dynamically from the root element's `nsmap`.

   f. **Serialize the modified XML** back to the same file path inside the extracted directory, preserving the XML declaration and encoding.

4. **Repack the HWPX archive.** Using Python's `zipfile` module, create `/root/training_feedback_ready.hwpx` by adding every file from the extracted directory back into a new ZIP, preserving the original relative paths. Use `ZIP_DEFLATED` compression. Make sure the `mimetype` file (if present) is stored first and uncompressed, as per ODF/HWPX packaging conventions.

5. **Validate the output.**
   - Confirm `/root/training_feedback_ready.hwpx` exists and is a valid ZIP.
   - Re-extract the section XML from the output and verify:
     - No `{{` or `}}` strings remain.
     - The `참석자수` value is digits only.
     - The `만족도` value matches the `X.X점 (5.0점 만점)` pattern.
     - The overall-opinion text ends with `후속 심화반 검토 요망.`
     - No `<hp:lineSegArray>` elements exist in modified paragraphs.
   - Print a summary of all checks.

**Important details:**
- Korean labels and the static note line must remain unchanged.
- Only modify text nodes; do not alter element structure except for removing `lineSegArray` from modified paragraphs.
- If the HWPX namespace prefix is not `hp`, detect it dynamically from the XML.
- Handle both sections if the template has content split across multiple section XML files.

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