# Task Instruction

You must produce the file `/root/clinic_intake_ready.hwpx` by filling the HWPX template with patient data.

## Step-by-step plan

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```
Identify the template (`clinic_intake_template.hwpx`) and data file (`patient_intake.json`). Read the JSON:
```bash
cat /root/patient_intake.json
```
(or wherever it is located)

### 2. Understand the HWPX package
A `.hwpx` file is a ZIP archive (like DOCX/ODP). Unzip it to inspect its structure:
```bash
mkdir -p /tmp/hwpx_work
cp /root/clinic_intake_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip template.hwpx -d template_extracted
find template_extracted -type f
```
Identify which XML files contain the document body text. Typically there is a `Contents/section0.xml` or similar. Search for placeholder patterns:
```bash
grep -r '{{' template_extracted/
```
This tells you every placeholder and which files contain them.

### 3. Read the JSON and understand placeholders
Parse `patient_intake.json` to identify all key-value pairs. Map each `{{placeholder}}` found in step 2 to its corresponding JSON value.

### 4. Write a Python script to perform the replacement
Create `/tmp/hwpx_work/fill_template.py` that:

a. **Loads the JSON** data file.

b. **Computes derived values:**
   - **Age**: Calculate Korean full-year age (만 나이) as of the visit date. Korean full-year age = `visit_year - birth_year`, but subtract 1 if the visit date is before the birthday in that year. Format as `(<N>세)`. This note must be **appended** right after the birth-date text in the same paragraph (or immediately following it), e.g., `1990-05-15 (34세)`.
   - **Phone normalization**: Take the callback phone number, strip all non-digit characters, then format as `NNN-NNNN-NNNN` (3-4-4 grouping). For example `01012345678` → `010-1234-5678`.

c. **Processes every XML file** inside the extracted HWPX directory:
   - Read each XML file as UTF-8 text.
   - For each placeholder `{{key}}`, replace ALL occurrences with the corresponding value (including the computed age note and normalized phone).
   - **Handle the age note insertion**: After replacing the birth-date placeholder, append ` (<N>세)` to the birth date value before substitution so it appears naturally in the text.
   - **Handle repeated placeholders**: e.g., patient name may appear multiple times (including a confirmation line). All must be replaced.
   - **Remove stale layout-cache elements**: After modifying any paragraph's text content, remove `<hp:linesegarray>` elements (and their children) from that paragraph. These are layout cache elements that cause overlapping characters if left stale. Use an XML parser (e.g., `lxml` or `xml.etree.ElementTree`) for this — do NOT use pure regex for XML manipulation of layout caches.

d. **Verify no `{{` remains** in any file in the package. If any remain, report them and abort.

e. **Re-pack the HWPX**: Re-create the ZIP archive from the modified extracted directory, preserving the original directory structure and using deflate compression. Save to `/root/clinic_intake_ready.hwpx`.

### 5. Important details for the script

- Use `lxml` or `xml.etree.ElementTree` to parse the XML files that contain placeholders. This is critical for properly removing `<hp:linesegarray>` (or similar layout-cache tags) from modified paragraphs.
- When repacking the ZIP, iterate the original ZIP's namelist to preserve entry order and paths. Use `zipfile.ZIP_DEFLATED`.
- The `mimetype` entry, if present, should be stored uncompressed (ZIP_STORED) as first entry — check if the original has this convention.
- Preserve all Korean labels and the handwritten-signature note — only replace `{{...}}` patterns.
- Do NOT add or remove paragraphs; only modify text content and remove stale caches from edited paragraphs.

### 6. Execute and validate
```bash
python3 /tmp/hwpx_work/fill_template.py
```
Then validate:
```bash
# Check it's a valid ZIP
unzip -t /root/clinic_intake_ready.hwpx

# Check no placeholders remain
unzip -p /root/clinic_intake_ready.hwpx | grep -c '{{'
# Should be 0

# Inspect the filled content to verify replacements
unzip -p /root/clinic_intake_ready.hwpx | grep -i 'linesegarray' | head -5
# Ideally none in modified paragraphs

# Quick visual check of text content
unzip -p /root/clinic_intake_ready.hwpx '*/section*.xml' 2>/dev/null || unzip -p /root/clinic_intake_ready.hwpx | head -200
```

### 7. Edge cases to handle
- Placeholders may be split across XML elements (e.g., `<t>{{patient</t><t>_name}}</t>`). If `grep -r '{{' ...` shows clean placeholders, simple text replacement suffices. If placeholders are split across tags, you must concatenate adjacent text runs, replace, then re-split or consolidate.
- The namespace for layout-cache elements may vary. Inspect the actual XML namespace declarations and tag names before writing removal logic. Common patterns: `<hp:linesegarray>`, `<linesegarray>`, or with a namespace URI.
- Ensure the age calculation handles edge cases (visit date exactly on birthday, etc.).

Deliver `/root/clinic_intake_ready.hwpx` as the final artifact.

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