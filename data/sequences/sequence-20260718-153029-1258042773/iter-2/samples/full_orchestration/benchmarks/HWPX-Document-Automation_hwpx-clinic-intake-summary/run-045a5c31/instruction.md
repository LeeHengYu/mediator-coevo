# Task Instruction

## Task: Complete clinic intake summary HWPX document

### Goal
Fill in the clinic intake template `clinic_intake_template.hwpx` with values from `patient_intake.json` and save the result to `/root/clinic_intake_ready.hwpx`.

### Step-by-step plan

#### 1. Understand the HWPX format
- A `.hwpx` file is a ZIP-based package (like OOXML). It contains XML files inside.
- List the contents of the template: `unzip -l clinic_intake_template.hwpx`
- Identify which XML files contain the document body text (likely something like `Contents/section0.xml` or similar).

#### 2. Inspect the patient data
- Read `patient_intake.json` to learn all available field values.
- Note the visit date (needed for age calculation).
- Note the phone number (needs normalization to `000-0000-0000` digit-hyphen format).
- Note the birth date (needed for Korean full-year age calculation).

#### 3. Inspect the template content
- Extract the hwpx to a temporary directory: `mkdir /tmp/hwpx_work && cp clinic_intake_template.hwpx /tmp/hwpx_work/template.hwpx && cd /tmp/hwpx_work && unzip template.hwpx -d template_extracted`
- Search for all `{{` placeholders across all XML files: `grep -rn '{{' template_extracted/`
- Document every unique placeholder found and which files they appear in.
- Also look for any layout-cache elements (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar cached layout tags) in paragraphs that contain placeholders.

#### 4. Prepare replacement values
- Map each placeholder to its value from the JSON.
- **Phone normalization**: Strip all non-digit characters from the callback phone number, then format as `NNN-NNNN-NNNN` (Korean mobile format: 3-4-4 digits).
- **Age calculation**: Compute Korean full-year age (만 나이) as of the visit date. This is: `visit_year - birth_year`, minus 1 if the visit date is before the birthday in that year. Format as `(<N>세)` and append it after the birth date value.
- **Patient name**: Ensure ALL occurrences are replaced, including any confirmation line.

#### 5. Perform replacements
- Write a Python script that:
  1. Extracts the hwpx ZIP to a working directory.
  2. Reads `patient_intake.json`.
  3. For each XML file in the extracted package, reads the content and performs all placeholder substitutions.
  4. For the birth date placeholder, appends the age note ` (<N>세)` after the date value.
  5. For paragraphs where text was modified, removes any layout-cache elements. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` elements (and their children) within `<hp:p>` paragraph elements. Check the actual element names in the XML namespace before removing.
  6. Verifies that NO `{{` or `}}` remains in any file in the package.
  7. Re-packages the directory back into a ZIP file saved as `/root/clinic_intake_ready.hwpx`, preserving the original directory structure and using the same compression method.

#### 6. Validation
- After creating the output file, verify:
  - `unzip -t /root/clinic_intake_ready.hwpx` succeeds (valid ZIP).
  - `unzip -p /root/clinic_intake_ready.hwpx | grep -c '{{'` returns 0 (no remaining placeholders across all files).
  - Grep for the patient name, phone number, age note to confirm they appear.
  - Confirm the Korean labels and handwritten-signature note are preserved (grep for a known Korean label from the template).
  - Confirm layout-cache elements were removed from modified paragraphs.

### Important details
- When re-zipping, the `mimetype` file (if present) should be stored first without compression (this is a convention for OPC/ZIP packages). Other files can use deflate.
- Use `zipfile` module in Python for precise control over the archive.
- Be careful with XML namespaces when searching for and removing layout-cache elements. Parse with `lxml.etree` or `xml.etree.ElementTree` and handle namespaces properly.
- Do NOT strip or alter content outside of placeholder replacement and layout-cache cleanup.
- The age note goes directly after the birth date value in the same text run or immediately after it, e.g., `1990-05-15 (34세)` — check the exact placeholder location to place it naturally.
- Double-check that repeated placeholders (like patient name appearing multiple times) are ALL replaced.

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