# Task Instruction

Complete the clinic intake summary HWPX document by filling in patient data from JSON. Follow these steps precisely:

## Step 1: Inspect inputs
- Read `/root/patient_intake.json` to understand all available data fields.
- Extract `/root/clinic_intake_template.hwpx` (it's a ZIP archive) to a temporary directory, e.g., `/tmp/hwpx_work/`.
- List the extracted files and identify the main content XML file(s) — typically under `Contents/` with names like `section0.xml` or similar.
- Read the content XML file(s) to find all `{{placeholder}}` patterns and understand the document structure.

## Step 2: Prepare replacement values
From the JSON data, prepare all replacement strings:
- **Patient name**: Use as-is for all occurrences (there may be a repeated patient-name confirmation line).
- **Birth date**: Use the value from JSON. Additionally, compute Korean full-year age (만 나이) as of the visit date: `age = visit_year - birth_year - (1 if (visit_month, visit_day) < (birth_month, birth_day) else 0)`. Append the age note in the format ` (<N>세)` immediately after the birth date value.
- **Phone number**: Clean to digits only, then format as `000-0000-0000` (for 11-digit Korean mobile numbers like 010-XXXX-XXXX). If 10 digits, use `00-000-0000` or `000-000-0000` as appropriate.
- **All other fields**: Replace with corresponding JSON values as-is.

## Step 3: Handle split-tag placeholders
HWPX XML often splits placeholder text across multiple `<hp:t>` elements within different `<hp:run>` elements. To handle this robustly:
1. For each paragraph (`<hp:p>`) element, concatenate all text content to check if it contains a `{{...}}` placeholder.
2. If placeholders are split across runs, consolidate the text into a single run or use a text-level replacement strategy that works across element boundaries.
3. A practical approach: work on the raw XML string, using regex to find and replace placeholders that may be split by XML tags. For example, build a regex pattern for each placeholder key that allows arbitrary XML tags between characters.
4. Alternatively, extract all text from a paragraph, perform replacements, then rebuild the paragraph with the replaced text in a single `<hp:run>` while preserving the formatting attributes from the first run.

## Step 4: Perform replacements
- Replace every `{{placeholder}}` with its corresponding value.
- Ensure the birth date line includes the age note: e.g., if birth date is `1985-03-15`, the output should be `1985-03-15 (39세)` (with the correct computed age).
- Ensure the phone number is in `000-0000-0000` format.
- Verify that the patient name appears in ALL locations, including any confirmation/signature line.
- Keep all existing Korean labels (e.g., `환자명:`, `생년월일:`, etc.) intact.
- Keep the handwritten-signature note (자필서명 or similar) intact.

## Step 5: Remove layout-cache elements from modified paragraphs
This is CRITICAL. For every `<hp:p>` paragraph whose text content was modified:
- Remove the `<hp:lineSegArray>` element (and its children) from that paragraph.
- Also check for case variations: `<hp:linesegarray>`, `<hc:lineSegArray>`.
- This forces the Hangul word processor to recalculate character positions, preventing overlapping/garbled text.
- Do NOT remove lineSegArray from paragraphs that were NOT modified.

## Step 6: Repack the HWPX archive
- Write the modified XML back to the content file(s) in the extracted directory.
- Repack into a ZIP file at `/root/clinic_intake_ready.hwpx`.
- If a `mimetype` file exists, add it first to the ZIP with `ZIP_STORED` (no compression). All other files use `ZIP_DEFLATED`.
- Preserve the original directory structure exactly.

## Step 7: Validate
- Extract the newly created `/root/clinic_intake_ready.hwpx` and read the content XML.
- Verify with a regex search that NO `{{` or `}}` patterns remain anywhere in any XML file in the archive.
- Verify the age note `세)` appears in the content.
- Verify the phone number matches the `000-0000-0000` pattern.
- Verify the patient name appears at least twice (original location + confirmation line).
- Verify the file is a valid ZIP archive.
- Print confirmation of all checks passing.

If any check fails, diagnose and fix before finalizing.

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