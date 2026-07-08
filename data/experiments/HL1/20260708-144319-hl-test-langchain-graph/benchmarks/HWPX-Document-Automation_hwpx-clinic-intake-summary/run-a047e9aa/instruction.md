# Task Instruction

Complete the clinic intake summary by filling the template with patient data and saving to /root/clinic_intake_ready.hwpx.

## Step 1: Inspect the workspace
```
ls -la /root/
find /root/ -name 'patient_intake.json' -o -name 'clinic_intake_template.hwpx' 2>/dev/null
```
Locate both `clinic_intake_template.hwpx` and `patient_intake.json`. They may be in /root/ or a subdirectory.

## Step 2: Read the patient data
```
cat <path_to>/patient_intake.json
```
Note all field names and values. Pay special attention to: patient name, birth date, visit date, phone number, and any other fields.

## Step 3: Explore the .hwpx template structure
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). Extract it to a temp directory:
```
mkdir -p /tmp/hwpx_work
cp <path_to>/clinic_intake_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
List all files. The main document content is typically in files like `Contents/section0.xml` or similar paths. Examine the directory structure.

## Step 4: Identify all placeholders
Search every XML file for `{{` patterns:
```
grep -rn '{{' template_contents/
```
Record every placeholder found (e.g., `{{patient_name}}`, `{{birth_date}}`, `{{phone}}`, etc.), noting which files contain them and how many occurrences of each. Also note any repeated placeholders (the instruction mentions patient-name confirmation line appearing twice).

## Step 5: Understand the XML structure around placeholders
Read the XML files containing placeholders carefully. Pay attention to:
- The XML element structure (likely `<hp:t>` or `<hp:run>` elements containing text)
- Whether placeholders might be split across multiple XML text nodes (e.g., `{{patient` in one node and `_name}}` in another) — if so, you must handle this
- Any layout-cache elements (like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:charPr>` with positioning, or similar cached layout data) within or near paragraphs containing placeholders

Read each relevant XML file fully:
```
cat template_contents/Contents/section0.xml
```
(Adjust path based on what you found in Step 3.)

## Step 6: Write a Python script to perform all replacements
Create a Python script that:

1. **Reads patient_intake.json** and loads all values.

2. **Computes Korean full-year age**: Calculate age as of the visit date. Korean full-year age (만 나이) = visit_year - birth_year, adjusted down by 1 if the visit date is before the birthday in that year. Format as `(<N>세)`. This note must be inserted right after the birth date value in the document text.

3. **Normalizes the phone number**: Strip all non-digit characters, then format as `000-0000-0000` (3-4-4 grouping for Korean mobile numbers). If the number is 11 digits like 01012345678, format as 010-1234-5678.

4. **Processes each XML file** in the extracted hwpx:
   - Parse with `xml.etree.ElementTree` (use the correct namespace handling — inspect the XML to find namespaces)
   - Handle the case where placeholders might be split across multiple text runs within a paragraph. If a placeholder like `{{name}}` is split across runs, you need to concatenate text within a paragraph, find placeholders, and redistribute text back. A safer approach: for each paragraph element, collect all text content, perform replacements on the concatenated string, then put the result back (possibly in a single text run if the original was split).
   - For the birth date field specifically, after replacing `{{birth_date}}` (or whatever the placeholder is) with the actual date, append ` (<N>세)` to that same text.
   - **Remove stale layout-cache elements**: For any paragraph whose text content was modified, remove child elements that represent layout caches. These are typically elements like `<lineseg>`, `<lineSegArray>`, `<hp:linesegarray>`, or similar. Inspect the actual XML to identify the exact element names. Removing these ensures the document re-renders cleanly.
   - Preserve all namespaces, XML declarations, and encoding.

5. **Writes modified XML files** back to the extracted directory.

6. **Repackages as .hwpx**: Re-zip the contents maintaining the original directory structure. The zip must be created from within the extracted root so paths are relative (no leading directory). Use:
   ```python
   import zipfile, os
   with zipfile.ZipFile('/root/clinic_intake_ready.hwpx', 'w', zipfile.ZIP_DEFLATED) as zf:
       for root, dirs, files in os.walk('template_contents'):
           for f in files:
               full = os.path.join(root, f)
               arcname = os.path.relpath(full, 'template_contents')
               zf.write(full, arcname)
   ```

**Important namespace handling**: When parsing XML with ElementTree, register all namespaces found in the document BEFORE parsing to avoid namespace prefix mangling. Look at the root element's namespace declarations and register them with `ET.register_namespace(prefix, uri)`. Also preserve any XML processing instructions or DOCTYPE declarations.

## Step 7: Run the script
```
python3 /tmp/hwpx_work/fill_template.py
```

## Step 8: Verify the output
1. Confirm the output file exists:
   ```
   ls -la /root/clinic_intake_ready.hwpx
   ```
2. Verify it's a valid ZIP:
   ```
   unzip -t /root/clinic_intake_ready.hwpx
   ```
3. Extract and verify no placeholders remain:
   ```
   mkdir -p /tmp/hwpx_verify
   unzip /root/clinic_intake_ready.hwpx -d /tmp/hwpx_verify
   grep -rn '{{' /tmp/hwpx_verify/
   ```
   This MUST return no results.
4. Verify the age note is present:
   ```
   grep -rn '세)' /tmp/hwpx_verify/
   ```
5. Verify the phone number is in correct format (digits-only with hyphens in 000-0000-0000):
   ```
   grep -rn '[0-9]\{3\}-[0-9]\{4\}-[0-9]\{4\}' /tmp/hwpx_verify/
   ```
6. Verify Korean labels and signature note are preserved:
   ```
   grep -rn '서명' /tmp/hwpx_verify/ || grep -rn '자필' /tmp/hwpx_verify/
   ```
7. Verify layout-cache elements were removed from modified paragraphs. Check that no `lineseg` or similar cache elements exist in paragraphs with replaced content.

## Critical Notes
- **Do NOT skip the XML namespace registration step.** ElementTree will mangle namespace prefixes without it, producing invalid hwpx XML.
- **Placeholder splitting**: Carefully check if `{{` and `}}` delimiters are in the same text node. If split, handle by paragraph-level text concatenation.
- **The age calculation must use Korean full-year (만 나이) convention**: years between birth and visit, minus 1 if birthday hasn't occurred yet in the visit year.
- **Layout cache removal is mandatory** for any modified paragraph. Inspect the actual XML element names before writing removal code.
- If the template has a `mimetype` file or `[Content_Types].xml`, ensure these are preserved exactly.
- If the original ZIP has a specific compression method or stored entries (like `mimetype`), try to match that. Some package formats require `mimetype` to be stored uncompressed as the first entry.

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