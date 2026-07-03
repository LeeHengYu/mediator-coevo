# Task Instruction

Complete the following task to fill in a clinic intake summary HWPX template.

## Goal
Replace all placeholders in `clinic_intake_template.hwpx` with values from `patient_intake.json` and save the result to `/root/clinic_intake_ready.hwpx`.

## Steps

### 1. Explore the workspace and understand the inputs
- List all files in the current working directory.
- Read `patient_intake.json` to understand all available field values.
- Since `.hwpx` is a ZIP-based package format (Korean Hangul word processor), unzip `clinic_intake_template.hwpx` to a temporary directory (e.g., `/tmp/hwpx_work/`) to inspect its contents.
- List all files in the extracted package. The main document content is typically in `Contents/section0.xml` (or similar path). Identify all XML files.
- Read the main section XML file(s) carefully to find all `{{...}}` placeholders and understand the XML structure.

### 2. Understand the placeholder-to-value mapping
- Map each `{{placeholder}}` found in the XML to the corresponding field in `patient_intake.json`.
- Note that patient name may appear multiple times (including a confirmation line).
- Identify the birth date field and the visit date field — you'll need both to calculate age.
- Identify the phone/callback number field that needs normalization.

### 3. Handle special requirements

#### Age calculation
- Calculate Korean full-year age (만 나이 / international age): the number of full years completed as of the visit date. This is: `visit_year - birth_year - 1` if the birthday hasn't occurred yet in the visit year, otherwise `visit_year - birth_year`.
- Format as `(<N>세)` and insert it immediately after the birth date value in the document text.

#### Phone number normalization
- Strip all non-digit characters from the callback phone number.
- Re-format as `000-0000-0000` (3-4-4 grouping with hyphens). If the number has 10 digits, use `00-0000-0000` or `000-000-0000` as appropriate, but 11-digit Korean mobile numbers should be `000-0000-0000`.

### 4. Perform the replacements in the XML
- **Critical**: Placeholders like `{{patient_name}}` may be split across multiple XML text runs/elements (e.g., `<hp:t>{{patient</hp:t><hp:t>_name}}</hp:t>`). You must handle this:
  - First, try simple string replacement on the full XML text.
  - Then check if any `{{` or `}}` fragments remain. If so, you need to merge adjacent text runs within the same paragraph, perform the replacement, then output the merged text in a single run.
- Replace ALL occurrences of every placeholder. No `{{...}}` text may remain.
- After replacement, verify by searching the entire XML for `{{` — there must be zero matches.

### 5. Remove stale layout-cache elements from modified paragraphs
- In HWPX XML, paragraphs may contain layout cache elements such as `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar elements that cache glyph positions.
- For any paragraph (`<hp:p>`) whose text content you modified, remove these layout-cache child elements entirely. This prevents overlapping characters when the document is opened.
- Common layout cache element names to look for and remove: `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineSeg>`, or any element that appears to store character position/width caching data. Inspect the XML structure to identify the correct element names.

### 6. Reassemble the HWPX package
- After modifying the XML file(s), repackage everything back into a ZIP file saved as `/root/clinic_intake_ready.hwpx`.
- The ZIP must preserve the original directory structure of the HWPX package.
- Use `zip` command or Python's `zipfile` module. Make sure to include all original files from the package.

### 7. Validate the output
- Unzip `/root/clinic_intake_ready.hwpx` to a new temp directory.
- Search all XML files for `{{` — confirm zero occurrences.
- Confirm the age note `(<N>세)` appears after the birth date.
- Confirm the phone number is in `000-0000-0000` format.
- Confirm the patient name appears in all expected locations.
- Confirm Korean labels and the handwritten-signature note are preserved.
- Confirm the file is a valid ZIP.

## Important Notes
- Do NOT skip the step of checking for split placeholders across XML elements. This is the most common failure mode.
- Do NOT leave any `{{...}}` placeholders in the output.
- When removing layout cache elements, be precise — only remove from paragraphs you actually modified, and only remove the caching elements, not content elements.
- Use Python for the XML manipulation if possible, as it gives you the most control with `xml.etree.ElementTree` or `lxml`. Be careful with XML namespaces in HWPX files.
- When working with ElementTree and namespaces, register namespaces before parsing to avoid `ns0:` prefix pollution in the output.

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