# Task Instruction

Complete the clinic intake summary HWPX document by filling in placeholders from patient data.

## Steps

### 1. Inspect the input files
- Read `/root/patient_intake.json` to understand all available data fields.
- Extract the HWPX template: `unzip -o /root/clinic_intake_template.hwpx -d /tmp/hwpx_work`
- List the extracted files and read the XML content files (especially `Contents/section*.xml` or similar) to identify all `{{...}}` placeholders and understand the document structure.

### 2. Write and run a Python script to perform the transformation

Create `/root/solve.py` that does the following:

#### a. Load patient data
- Parse `/root/patient_intake.json`.

#### b. Extract the template
- Use `zipfile` to extract `/root/clinic_intake_template.hwpx` to a temp directory.

#### c. Process each XML file in the package
For every `.xml` file in the extracted archive:

1. **Parse with `xml.etree.ElementTree`** using the `hp` namespace (register namespace `hp` = `http://www.hancom.co.kr/hwpml/2011/paragraph` or whichever URI is declared; inspect the file first to get the exact namespace URIs).

2. **Consolidate split placeholders**: For each `<hp:p>` (paragraph) element, collect all `<hp:t>` text nodes across all `<hp:run>` children. Concatenate them to detect placeholders that span multiple runs. If a placeholder is found:
   - Put the full replaced text into the first `<hp:t>` element.
   - Clear the text of all subsequent `<hp:t>` elements in that paragraph (set to empty string).
   - This handles cases where `{{patient_name}}` is split like `{{pat` + `ient_na` + `me}}`.

3. **Perform replacements**: Replace all `{{placeholder}}` patterns with the corresponding values from the JSON. Specific rules:
   - **Age note**: After the birth date value, append ` (<N>세)` where N is the Korean full-year age. Calculate as: `age = visit_year - birth_year` then subtract 1 if the visit date is before the birthday in that year. Use the visit date from the JSON data.
   - **Phone normalization**: Convert the callback/phone number to `000-0000-0000` format (strip all non-digits, then format as 3-4-4 groups with hyphens).
   - **All other placeholders**: Direct substitution from JSON values.
   - **Repeated placeholders** (e.g., patient name confirmation): Ensure ALL occurrences are replaced, not just the first.

4. **Remove layout cache from modified paragraphs**: Using the XML parser (NOT regex), find and remove ALL `<hp:lineSegArray>` elements within any `<hp:p>` element whose text content was modified. This is critical — the safety-audit-brief task failed because regex missed some instances. Use `element.findall()` with proper namespace and `element.remove()` for each found lineSegArray.

5. **Verify no placeholders remain**: After all replacements, scan the full XML text for any remaining `{{` or `}}` patterns. If found, raise an error.

#### d. Repackage as HWPX
- Create `/root/clinic_intake_ready.hwpx` using `zipfile.ZipFile`.
- **Critical**: Add the `mimetype` file FIRST with `compression=zipfile.ZIP_STORED` (no compression).
- Add all other files with `compression=zipfile.ZIP_DEFLATED`.
- Preserve the original directory structure.

### 3. Run the script
```bash
python3 /root/solve.py
```

### 4. Validate the output
- Verify `/root/clinic_intake_ready.hwpx` exists and is a valid ZIP: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_ready.hwpx'); z.testzip(); print('Valid ZIP'); z.close()"`
- Extract and check for remaining placeholders: `unzip -p /root/clinic_intake_ready.hwpx | grep -c '{{' ` — should be 0.
- Check that the mimetype entry is first and stored: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/clinic_intake_ready.hwpx'); i=z.infolist()[0]; print(f'First entry: {i.filename}, compress_type: {i.compress_type}'); assert i.filename=='mimetype' and i.compress_type==0"`
- Grep for `lineSegArray` in modified content to confirm removal.
- Check that Korean labels and handwritten-signature note are preserved.
- Verify the age note format `(<N>세)` appears in the output.
- Verify the phone number matches `000-0000-0000` pattern.

## Important Notes
- Before writing the script, you MUST inspect the actual namespace URIs in the XML files. HWPX namespaces vary. Register all namespaces found so they are preserved in output.
- When using ElementTree, register namespaces BEFORE parsing to avoid `ns0:` prefix pollution in output.
- Handle the case where the JSON might use different key names than the placeholders — map them carefully by inspecting both files.
- If the birth date or visit date format needs parsing, use `datetime` module.
- Korean age calculation: age = visit_year - birth_year, then subtract 1 if visit month/day is before birth month/day.

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