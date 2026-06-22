# Task Instruction

Complete the clinic intake summary by filling a HWPX template with patient data. Follow these steps precisely:

## Step 1: Inspect the workspace
```bash
ls -la /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -20
```

## Step 2: Read the patient data
```bash
cat /root/patient_intake.json
```
Note all field names and values. Pay special attention to: patient name, birth date, visit date, phone number.

## Step 3: Explore the HWPX template structure
```bash
mkdir -p /tmp/hwpx_work
cp /root/clinic_intake_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip -o template.zip -d template_contents
find template_contents -type f
```
HWPX is a ZIP package containing XML files. List all files to understand the structure.

## Step 4: Find and examine all XML content files
```bash
find /tmp/hwpx_work/template_contents -name '*.xml' -exec echo '=== {} ===' \; -exec cat {} \;
```
Identify ALL `{{...}}` placeholders across ALL XML files. Note which files contain them. Also look for any content in files with other extensions.

## Step 5: Search exhaustively for all placeholders
```bash
grep -rn '{{' /tmp/hwpx_work/template_contents/
```
Record every placeholder and its location. Note repeated occurrences (e.g., patient name may appear multiple times including a confirmation line).

## Step 6: Build and run a Python script to perform all modifications

Write a Python script `/tmp/hwpx_work/fill_template.py` that:

a) Reads `patient_intake.json` to get all values.

b) Computes Korean full-year age: age = visit_year - birth_year, subtract 1 if the visit date is before the birthday in that year. Format as `(<N>세)` where <N> is the integer age.

c) Normalizes the phone number: strip all non-digit characters, then format as `NNN-NNNN-NNNN` (3-4-4 grouping). If the number starts with a country code like +82, convert appropriately.

d) Builds a mapping from placeholder names to replacement values.

e) For each XML file in the extracted template:
   - Parse with lxml or xml.etree.ElementTree, preserving namespaces.
   - Walk all text nodes (including `.text` and `.tail` of every element).
   - Replace any `{{placeholder}}` patterns with the corresponding values.
   - For the birth date field, append ` (<N>세)` after the birth date value.
   - CRITICAL: For any paragraph element (typically `hp:p` or similar) where text was modified, remove all child elements that represent layout cache. These are typically elements like `hp:linesegarray`, `hp:lineSegArray`, or elements in a layout namespace. Inspect the actual XML to identify the correct element names.
   - Write the modified XML back, preserving the XML declaration and encoding.

f) Repackages the modified contents into `/root/clinic_intake_ready.hwpx` as a proper ZIP file, preserving the directory structure and using deflate compression. Important: use `os.walk` from the extracted root and add files with their relative paths matching the original structure. If there's a `mimetype` file, store it uncompressed first (like ODF convention) — but check the original ZIP to see if this convention applies.

## Step 7: Run the script
```bash
python3 /tmp/hwpx_work/fill_template.py
```

## Step 8: Validate the output
```bash
# Verify it's a valid ZIP
unzip -t /root/clinic_intake_ready.hwpx

# Verify no placeholders remain
unzip -p /root/clinic_intake_ready.hwpx | grep -c '{{'
# This should output 0

# More thorough check across all files
mkdir -p /tmp/hwpx_verify
unzip -o /root/clinic_intake_ready.hwpx -d /tmp/hwpx_verify
grep -rn '{{' /tmp/hwpx_verify/
# Should produce no output

# Verify the age note is present
grep -rn '세)' /tmp/hwpx_verify/

# Verify phone format
grep -rn '[0-9]\{3\}-[0-9]\{4\}-[0-9]\{4\}' /tmp/hwpx_verify/

# Verify Korean labels are preserved
grep -rn '서명' /tmp/hwpx_verify/ || echo 'Check for signature note preservation'
```

## Step 9: If any validation fails, debug and fix
- If placeholders remain, check if they span multiple XML elements (e.g., `{{` in one element and `}}` in another). If so, update the script to handle split placeholders by concatenating text runs within a paragraph before replacement, then setting the merged text on the first run and clearing subsequent runs.
- If the ZIP is invalid, check the packaging step.
- If layout issues are suspected, verify that layout-cache elements were properly removed from modified paragraphs.

## Key cautions:
- Placeholders may be split across multiple XML text runs within the same paragraph. This is very common in word processor formats. Handle this by examining the raw XML carefully.
- Preserve all namespace declarations in XML files.
- Do NOT modify the `[Content_Types].xml` or relationship files unless they contain placeholders.
- The handwritten-signature note (서명 related text) must be preserved as-is.
- Every single `{{...}}` must be replaced — check ALL files in the package, not just the main content file.

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