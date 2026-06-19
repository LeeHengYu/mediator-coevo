# Task Instruction

Complete the clinic intake summary by filling the HWPX template with patient data. Follow these steps precisely:

## Step 1: Inspect inputs
- Read `patient_intake.json` to understand all available field values.
- List the contents of `clinic_intake_template.hwpx` (it's a ZIP) to understand its structure.
- Extract the HWPX to a temporary directory (e.g., `/tmp/hwpx_work/`).
- Inspect the XML content files (likely under `Contents/`) to find all `{{...}}` placeholders and understand the XML namespace structure.

## Step 2: Write and run a Python script that does the following:

### 2a: Parse patient_intake.json
Load all patient data fields from the JSON file.

### 2b: Compute Korean full-year age
- Parse the patient's birth date and the visit date from the JSON.
- Calculate age as: `visit_year - birth_year`, minus 1 if the visit date is before the birthday in that year.
- Format as `(<N>세)` (e.g., `(45세)`).

### 2c: Normalize phone number
- Strip all non-digit characters from the callback/phone number.
- Reformat as `000-0000-0000` (3-4-4 grouping). If the number has 10 digits, use `00-0000-0000` or `000-000-0000` as appropriate — but the standard Korean mobile format for 11 digits is `000-0000-0000`.

### 2d: Process XML files with split-placeholder awareness
For each XML file in the extracted HWPX (especially section XML files under `Contents/`):

1. **Register all namespaces** before parsing to preserve prefixes like `hp:`, `hc:`, `hh:`, etc. Use `xml.etree.ElementTree.iterparse` or register namespaces from the root element attributes.

2. **Handle split placeholders**: Placeholders like `{{patient_name}}` may be split across multiple `<hp:t>` (or similar text) tags within a single paragraph `<hp:p>`. For each paragraph:
   - Concatenate all text content from child text runs.
   - Check if the concatenated text contains any `{{...}}` placeholder.
   - If so, perform replacements on the concatenated string, then put the full replaced text into the first text run and clear the remaining text runs.

3. **Replacement mapping**: Build a mapping from placeholder names to values. Include:
   - Patient name (may appear multiple times, including a confirmation line)
   - Birth date, with the age note `(<N>세)` appended after it
   - Phone/callback number in normalized format
   - All other fields from the JSON (address, visit date, symptoms, allergies, medications, insurance info, emergency contact, etc.)

4. **Verify no `{{...}}` remains**: After all replacements, scan the entire XML text to confirm zero remaining `{{` or `}}` sequences.

### 2e: Remove stale layout cache
For every `<hp:p>` paragraph element that was modified (had text replaced):
- Find and remove any `<hp:lineSegArray>` child element (and all its descendants).
- This prevents overlapping character rendering when the document is opened.

### 2f: Write modified XML back
Serialize the modified XML tree back to the file, preserving the XML declaration and encoding.

### 2g: Repackage as HWPX
- Create `/root/clinic_intake_ready.hwpx` as a ZIP file.
- **Important**: ZIP from within the extracted root directory so that paths like `mimetype`, `Contents/`, `META-INF/` are at the archive root (not nested under an extra directory).
- If a `mimetype` file exists, add it first with `ZIP_STORED` compression (no compression) as required by ODF-based formats.
- Add all other files with `ZIP_DEFLATED` compression.

## Step 3: Validate the output
1. Verify `/root/clinic_intake_ready.hwpx` exists and is a valid ZIP.
2. Extract it and check that no XML file contains any `{{` placeholder text.
3. Confirm the age note appears in the expected format `(<N>세)`.
4. Confirm the phone number appears in `000-0000-0000` format.
5. Confirm Korean labels and the handwritten-signature note are preserved.
6. Confirm `<hp:lineSegArray>` elements are removed from modified paragraphs.

If any validation fails, diagnose and fix before finishing.

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