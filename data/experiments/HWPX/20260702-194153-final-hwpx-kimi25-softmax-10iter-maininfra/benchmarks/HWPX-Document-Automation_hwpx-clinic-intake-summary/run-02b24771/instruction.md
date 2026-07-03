# Task Instruction

Complete the clinic intake summary by filling in the HWPX template with patient data and saving the result.

## Steps

### 1. Inspect the input files
- Read `patient_intake.json` to understand all available patient data fields.
- List the contents of `clinic_intake_template.hwpx` (it's a ZIP archive) to find the main document XML (likely `Contents/section0.xml` or similar).
- Extract and read the main section XML file(s) to identify every `{{...}}` placeholder and understand the XML structure, namespaces, and layout-cache elements.

### 2. Plan replacements
- Map each `{{placeholder}}` in the XML to the corresponding field in `patient_intake.json`.
- Note any placeholder that appears more than once (e.g., patient name confirmation line) — all occurrences must be replaced.
- Calculate Korean full-year age (만 나이): `age = visit_year - birth_year`, subtract 1 if the visit date is before the birthday in that year. Format as `(<N>세)` and append it after the birth-date value.
- Normalize the callback phone number: strip all non-digit characters, then format as `000-0000-0000` (3-4-4 grouping).

### 3. Handle split placeholders in HWPX XML
HWPX editors often split text across multiple `<hp:t>` elements within a single `<hp:run>` or across runs in a paragraph. To handle this:
- For each `<hp:p>` (paragraph), concatenate the text content of all `<hp:t>` elements to get the full paragraph text.
- Perform placeholder replacement on the concatenated text.
- If a replacement was made, rewrite the paragraph's `<hp:t>` elements so the first `<hp:t>` contains the full replaced text and remove or empty the remaining `<hp:t>` elements in that run sequence.
- Alternatively, perform regex-based replacement on the raw XML string, but be careful to handle cases where XML tags appear between placeholder characters (e.g., `{{na</hp:t><hp:t>me}}`).

A robust approach: use regex on the raw XML string to collapse split placeholders. For example, for a placeholder like `{{name}}`, build a pattern that allows optional closing/opening `</hp:t>...</hp:t>` tags between each character of the placeholder name, then replace the entire match with the single value in a single `<hp:t>` element.

### 4. Remove stale layout-cache elements
For every `<hp:p>` paragraph whose text content was modified:
- Remove all `<hp:lineSegArray>` child elements (and their contents) from that paragraph.
- This prevents overlapping character rendering when the document is opened in a viewer.

### 5. Verify no placeholders remain
- After all replacements, search the entire XML for any remaining `{{` or `}}` patterns.
- If any remain, investigate and fix them (they may be in a different section XML file or split in an unexpected way).

### 6. Repackage the HWPX file
- The HWPX format is a ZIP archive. Repackage the modified XML back into a new ZIP file at `/root/clinic_intake_ready.hwpx`.
- Preserve all other files from the original archive exactly as-is (images, metadata, other XML files).
- Use `zipfile.ZIP_DEFLATED` compression to match the original packaging.
- Make sure the output is a valid ZIP that can be opened as a `.hwpx` document.

### 7. Final validation
- Unzip `/root/clinic_intake_ready.hwpx` and verify:
  - The section XML is well-formed XML (parse it with an XML parser).
  - No `{{...}}` placeholders remain anywhere in any XML file in the archive.
  - The age note `(<N>세)` appears after the birth date.
  - The phone number is in `000-0000-0000` format.
  - Korean labels and the handwritten-signature note are preserved.
  - `<hp:lineSegArray>` elements are absent from modified paragraphs.

## Key Reminders
- Preserve all XML namespaces exactly as they appear in the original.
- Do not modify paragraphs that don't contain placeholders.
- Keep the existing document structure, styles, and formatting intact.
- The only output file is `/root/clinic_intake_ready.hwpx`.

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