# Task Instruction

This task has already been solved successfully (reward 1.0) with a proven strategy. Repeat the same approach:

1. **Inspect the workspace**: List files in the task directory. Read `patient_intake.json` to understand all field values. Examine `clinic_intake_template.hwpx` by unzipping it and identifying the XML content file(s) (typically under `Contents/section*.xml`).

2. **Understand the template**: Read the raw XML of each section file. Identify all `{{...}}` placeholders. Note that placeholders may be split across multiple `<hp:t>` tags due to HWPX's internal tagging.

3. **Build the replacement logic in Python**:
   - Parse `patient_intake.json` to get all field values.
   - **Phone number normalization**: Convert the callback phone number to `000-0000-0000` format (digits only, hyphen-grouped as 3-4-4).
   - **Korean age calculation**: Compute 만 나이 (full-year age) as of the visit date. Age = visit_year - birth_year, minus 1 if the visit date is before the birthday in that year. Format as `(<N>세)` and append it after the birth date value.
   - Read the entire XML content as a raw string.
   - **Critical**: Before doing placeholder replacement, collapse split placeholders. Use regex to merge sequences of `</hp:t></hp:run><hp:run ...><hp:t>` (and variants with attributes) that break up `{{...}}` tokens, so each placeholder becomes a single continuous string within one `<hp:t>` element. Specifically, use a pattern like `re.sub(r'</hp:t></hp:run><hp:run[^>]*><hp:t>', '', xml_string)` iteratively or in a targeted way around `{{` and `}}` regions.
   - Then do simple string replacement for each `{{placeholder}}` with its corresponding value.
   - **Verify no `{{` or `}}` remains** in the XML after all replacements.

4. **Strip layout caches**: Remove all `<hp:lineSegArray>...</hp:lineSegArray>` elements from any paragraph whose text was modified. The safest approach is to remove ALL `<hp:lineSegArray>` elements from the entire document using `re.sub(r'<hp:lineSegArray>.*?</hp:lineSegArray>', '', xml_string, flags=re.DOTALL)`.

5. **Reassemble the HWPX package**:
   - Copy `clinic_intake_template.hwpx` to `/root/clinic_intake_ready.hwpx`.
   - Use Python's `zipfile` module to open the copy, replace the modified XML file(s) while preserving all other entries (mimetype, metadata, etc.).
   - Ensure the mimetype entry (if present) is stored without compression (ZIP_STORED) as the first entry, per ODF/HWPX conventions.

6. **Validate**:
   - Re-open `/root/clinic_intake_ready.hwpx` as a ZIP and read back the XML.
   - Confirm no `{{` remains anywhere in any XML file.
   - Confirm the age note `(<N>세)` appears.
   - Confirm the phone number is in `000-0000-0000` format.
   - Print a summary of checks passed.

Write all of this as a single Python script and execute it. If any step fails, debug and re-run.

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