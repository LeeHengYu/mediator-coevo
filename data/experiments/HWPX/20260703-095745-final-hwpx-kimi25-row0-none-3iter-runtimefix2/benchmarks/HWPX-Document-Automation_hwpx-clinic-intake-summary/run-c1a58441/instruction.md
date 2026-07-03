# Task Instruction

Complete the clinic intake summary HWPX document by filling in placeholders from the patient JSON data. Follow these steps precisely:

## Step 1: Inspect the input files

1. List the contents of the task directory to find `clinic_intake_template.hwpx` and `patient_intake.json`.
2. Read `patient_intake.json` fully to understand all available fields.
3. Extract `clinic_intake_template.hwpx` (it's a ZIP archive) and list its contents.
4. Read the main document XML file (likely `Contents/section0.xml` or similar) and identify all `{{...}}` placeholders. Note which placeholders appear multiple times.
5. Also check any other XML files in the archive for placeholders (e.g., header/footer XMLs).

## Step 2: Write a Python script to perform the transformation

Create a Python script `/root/fill_intake.py` that does the following:

### 2a: Load patient data
- Parse `patient_intake.json` into a dict.

### 2b: Compute derived values
- **Age calculation**: Compute Korean full-year age (만 나이) as of the visit date. This is: `visit_year - birth_year`, minus 1 if the visit date is before the birthday in that year. Format the age note as `(<N>세)` — this string will be appended after the birth date value.
- **Phone normalization**: Convert the callback phone number to `000-0000-0000` format (strip all non-digit characters, then format as 3-4-4 digit groups with hyphens).

### 2c: Build a replacement map
- Map each placeholder name (without the `{{` `}}` delimiters) to its replacement value from the JSON.
- For the birth date placeholder, the replacement should be the birth date value followed by a space and the age note, e.g., `1985-03-15 (38세)`.
- For the phone placeholder, use the normalized phone number.
- Include all other fields as-is from the JSON.

### 2d: Process the HWPX archive
- Open the template HWPX as a ZIP.
- For each file in the ZIP:
  - If it's an XML file that could contain placeholders (check all `.xml` files), read its content as a UTF-8 string.
  - **Consolidate fragmented placeholders**: HWPX/Hancom often splits text across multiple `<hp:t>` elements within a `<hp:run>`. Before doing replacements, for each `<hp:p>` paragraph element, extract all text content, check if it contains a `{{...}}` pattern, and if the placeholder is split across tags, consolidate the text into a single `<hp:t>` element (removing the extra `<hp:t>` elements and their parent `<hp:run>` wrappers if they become empty).
  - **Specific robust approach**: Use a regex-based strategy on the raw XML string. For each placeholder like `{{patient_name}}`, build a regex pattern that allows optional XML tags (`<[^>]+>`) between each character of the placeholder string (including the `{{` and `}}` delimiters). Replace the entire match (including any interleaved tags) with just the replacement value text (XML-escaped). This handles arbitrary tag splitting.
  - After all replacements, verify no `{{` remains in the text content of the XML.
  - **Remove layout cache from modified paragraphs**: Parse the modified XML with `xml.etree.ElementTree` or use regex to find every `<hp:p>` element whose text content contains any of the replacement values. For each such paragraph, remove any `<hp:lineSegArray>` or `<hp:linesegarray>` child element (case-insensitive search). **Important**: To be safe, remove `<hp:lineSegArray>` (and any case variant like `<hp:linesegarray>`) from ALL paragraphs that were modified. A practical approach: after doing text replacements, use regex to remove `<hp:lineSegArray>...</hp:lineSegArray>` (with a non-greedy match and case-insensitive flag) from every `<hp:p>` block that contains any replacement value substring. Alternatively, simply remove ALL `<hp:lineSegArray>` elements from the entire document XML — this is safe because the word processor will regenerate them on open.
  - Write the processed content to the output ZIP.
- For non-XML files, copy them as-is.
- Save the result to `/root/clinic_intake_ready.hwpx`.

### 2e: Validation
- After creating the output, open it as a ZIP and verify:
  1. It's a valid ZIP file.
  2. The main XML contains none of the original `{{...}}` placeholder patterns.
  3. The main XML contains the expected replacement values (spot-check patient name, phone number, age note).
  4. No `<hp:lineSegArray>` elements remain in paragraphs containing replacement text (or in the entire document if you chose the aggressive removal approach).
  5. Print confirmation of each check.

## Step 3: Execute the script

Run `python3 /root/fill_intake.py` and check the output.

## Step 4: Final verification

1. Unzip `/root/clinic_intake_ready.hwpx` and grep for `{{` in all XML files — there must be zero matches.
2. Grep for the age note pattern `(\d+세)` to confirm it's present.
3. Grep for the normalized phone number pattern `\d{3}-\d{4}-\d{4}` to confirm it's present.
4. Confirm no `<hp:lineSegArray>` elements exist in paragraphs you modified (or in the whole document).

## Critical reminders
- The `{{...}}` placeholders WILL be split across multiple XML tags. You MUST handle this. The regex approach (allowing `<[^>]+>` between characters) has been proven to work in prior tasks.
- Removing `<hp:lineSegArray>` is MANDATORY for modified paragraphs. Prior tasks have failed verification on this point. Be aggressive — removing them from the entire document is acceptable and safe.
- The patient name may appear multiple times (e.g., a confirmation line). ALL occurrences must be replaced.
- Preserve all existing Korean labels and the handwritten-signature note — do not modify any text that isn't a placeholder.
- The output must be a valid `.hwpx` ZIP package with the same directory structure as the input.

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