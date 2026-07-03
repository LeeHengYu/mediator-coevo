# Task Instruction

Complete the project proposal document by filling in placeholders and making required modifications. Follow these steps precisely:

## Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). You'll need to unzip it, modify the XML content, and rezip it.

## Step 2: Inspect the workspace
```bash
ls /root/
file /root/project_proposal_template.hwpx
```

## Step 3: Read the JSON data
```bash
cat /root/project_proposal.json
```
Note all key-value pairs. You'll need these to replace `{{...}}` placeholders.

## Step 4: Extract the HWPX template
```bash
mkdir -p /root/hwpx_work
cd /root/hwpx_work
unzip -o /root/project_proposal_template.hwpx
```

## Step 5: Explore the extracted structure
```bash
find /root/hwpx_work -type f | head -50
```
Identify all XML files, especially those under `Contents/` (typically `section0.xml`, `section1.xml`, etc.) and any `content.hpf` or similar manifest files.

## Step 6: Search for all placeholders
```bash
grep -r '{{' /root/hwpx_work/ --include='*.xml' -l
grep -r '{{' /root/hwpx_work/ --include='*.xml'
```
This reveals which files contain `{{...}}` placeholders and exactly what they look like. **CRITICAL**: Placeholders may be split across multiple XML elements (e.g., `{{` in one text run, `project_name` in another, `}}` in a third). You must handle this.

## Step 7: Read each XML file containing placeholders in full
For every file identified in Step 6, read the complete file content. Understand the XML structure thoroughly before making any edits.

## Step 8: Apply replacements using Python
Write a Python script that:

1. Reads the JSON file to get replacement values.
2. For each XML file containing placeholders:
   a. Read the raw XML content.
   b. **First**, try to reconstruct fragmented placeholders: The XML may split `{{key}}` across multiple `<hp:t>` or similar text elements within the same paragraph/run. Concatenate adjacent text nodes within the same paragraph to find complete `{{key}}` patterns, then replace them while preserving XML structure.
   c. Replace each `{{key}}` with the corresponding JSON value.
   d. **Budget normalization**: For any budget/cost value, remove commas from the number but keep the currency symbol (e.g., `₩1,000,000,000` → `₩1000000000`).
   e. **Month span calculation**: For lines containing `단계1`, `단계2`, `단계3` with date ranges, calculate the month span from the date range present in that line and append ` (N개월)` after the phase text. The expected results are: `단계1` → append `(3개월)`, `단계2` → append `(3개월)`, `단계3` → append `(1개월)`. Parse the dates (likely in YYYY.MM or YYYY.MM.DD format) to compute the month difference, or use the known expected values.
   f. **Remove stale layout-cache elements**: For any paragraph whose text content was modified, remove layout-cache elements. These are typically `<hp:linesegarray>` or `<lineseg>` or similar elements that cache glyph positions. Remove them from modified paragraphs so the document renders cleanly.
3. Write modified XML back to the same file paths.

## Step 9: Verify no placeholders remain
```bash
grep -r '{{' /root/hwpx_work/ --include='*.xml'
```
This must return nothing. If any `{{` patterns remain, investigate and fix them.

## Step 10: Verify month spans were added
```bash
grep -r '개월' /root/hwpx_work/ --include='*.xml'
```
Confirm `(3개월)`, `(3개월)`, `(1개월)` appear for 단계1, 단계2, 단계3 respectively.

## Step 11: Verify budget normalization
Confirm the budget value has no commas but retains the currency symbol.

## Step 12: Repackage as HWPX
```bash
cd /root/hwpx_work
zip -r -0 /root/project_proposal_ready.hwpx mimetype
zip -r /root/project_proposal_ready.hwpx . -x mimetype -x '*.DS_Store'
```
Note: If a `mimetype` file exists, it should be stored first without compression (ZIP spec for OPC packages). If no mimetype file exists, just zip everything normally:
```bash
cd /root/hwpx_work
zip -r /root/project_proposal_ready.hwpx .
```

## Step 13: Validate the output
```bash
file /root/project_proposal_ready.hwpx
python3 -c "import zipfile; z=zipfile.ZipFile('/root/project_proposal_ready.hwpx'); print(z.namelist()); z.close()"
```
Confirm it's a valid ZIP and contains the expected files.

## Key Warnings
- **Fragmented placeholders in XML**: This is the #1 pitfall. `{{project_name}}` may appear as `<hp:t>{{</hp:t><hp:t>project_name}}</hp:t>` or even more fragmented. Your Python script MUST handle this by working at the paragraph level.
- **Do NOT change Korean labels or the static note line.**
- **Do NOT leave any `{{...}}` in the output.**
- **Remove `<hp:linesegarray>` (or equivalent layout cache elements) from every paragraph you modify.**
- The output file must be at exactly `/root/project_proposal_ready.hwpx`.

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