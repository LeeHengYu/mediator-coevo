# Task Instruction

Complete the clinic intake summary by filling the HWPX template with patient data. Follow these steps precisely:

## Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). The main document content is typically in `Contents/section0.xml` (or similar path). Use `unzip` to explore the structure.

## Step 2: Inspect the inputs
1. List and examine the files in the working directory to find `clinic_intake_template.hwpx` and `patient_intake.json`.
2. Read `patient_intake.json` to understand all available field values.
3. Extract the template HWPX to a temporary directory: `mkdir -p /tmp/hwpx_work && cp clinic_intake_template.hwpx /tmp/hwpx_work/template.zip && cd /tmp/hwpx_work && unzip template.zip -d template_extracted`
4. List all files in the extracted archive: `find /tmp/hwpx_work/template_extracted -type f`
5. Read ALL XML files in the extracted archive, especially any under `Contents/` — look for files like `section0.xml`, `content.hpf`, etc. Read every XML file to find where `{{...}}` placeholders appear.

## Step 3: Identify all placeholders
Search all extracted files for `{{` patterns: `grep -r '{{' /tmp/hwpx_work/template_extracted/`
Document every unique placeholder found and which files contain them. Placeholders may be split across multiple XML elements/runs — check carefully.

## Step 4: Prepare replacement values from patient_intake.json
- For the patient name: use as-is, replace ALL occurrences including any confirmation line.
- For birth date: after inserting the birth date value, append ` (<N>세)` where N is the Korean full-year age. Korean full-year age (만 나이 is Western age, but '세' with full-year typically means Korean age = current_year - birth_year). Calculate: age = visit_year - birth_year. Format as `(<age>세)` appended right after the birth date.
- For phone/callback number: normalize to `000-0000-0000` format (strip all non-digits, then format as 3-4-4 groups with hyphens).
- For all other fields: substitute directly from the JSON.

## Step 5: Perform replacements
Write a Python script to do the replacements. Key considerations:
- Read each XML file that contains placeholders.
- Placeholders like `{{patient_name}}` might be split across multiple XML `<hp:t>` or `<hp:run>` elements. You must handle this: scan the concatenated text of sibling runs, find the placeholder spanning runs, merge or consolidate the text into one run, and remove the now-empty runs.
- After modifying any paragraph's text, remove any layout-cache elements (`<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, or similar layout cache elements within the same paragraph or parent). These are stale layout caches that cause overlapping characters. Search for elements with names containing 'lineseg', 'lineSegArray', 'linesegarray', 'charShapeArray', or similar cache elements within modified paragraphs. Use `lxml` or `xml.etree.ElementTree` with proper namespace handling.
- Preserve all Korean labels and the handwritten-signature note — only replace `{{...}}` patterns.
- Ensure NO `{{...}}` placeholder text remains anywhere.

## Step 6: Repackage the HWPX
Repackage the modified files back into a valid HWPX (ZIP) archive:
```
cd /tmp/hwpx_work/template_extracted
zip -r /root/clinic_intake_ready.hwpx . -x '*.DS_Store'
```
Note: HWPX files may need `mimetype` as the first uncompressed entry (like EPUB). Check if a `mimetype` file exists — if so, add it first with `zip -0` (stored, no compression), then add the rest.

## Step 7: Validate
1. Verify the output exists: `ls -la /root/clinic_intake_ready.hwpx`
2. Verify it's a valid ZIP: `unzip -t /root/clinic_intake_ready.hwpx`
3. Extract and verify no `{{` remains: `mkdir -p /tmp/verify && unzip -o /root/clinic_intake_ready.hwpx -d /tmp/verify && grep -r '{{' /tmp/verify/` — this must return nothing.
4. Verify the age note is present: `grep -r '세)' /tmp/verify/`
5. Verify the phone number format (000-0000-0000): `grep -r '[0-9]\{3\}-[0-9]\{4\}-[0-9]\{4\}' /tmp/verify/`
6. Verify Korean labels and signature note are preserved.
7. Verify layout cache elements were removed from modified paragraphs.

## Important Notes
- Use Python with `lxml` or `xml.etree.ElementTree` for XML manipulation to handle namespaces correctly.
- When parsing HWPX XML, identify the namespace prefixes used (e.g., `hp:`, `hc:`, etc.) and handle them properly.
- Be very careful with placeholder splitting across XML runs — this is the most common failure point.
- The age calculation: read the visit_date and birth_date from the JSON, extract years, compute visit_year - birth_year for Korean age.
- Phone normalization: strip everything except digits, then format as NNN-NNNN-NNNN.

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