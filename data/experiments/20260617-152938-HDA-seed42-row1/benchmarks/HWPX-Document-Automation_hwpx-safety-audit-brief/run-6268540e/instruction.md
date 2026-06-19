# Task Instruction

Complete the following task to prepare a warehouse safety audit brief from a HWPX template.

## Goal
Fill the template `safety_audit_template.hwpx` with data from `audit_overview.json` and `corrective_actions.json`, then save the result to `/root/safety_audit_brief_final.hwpx`.

## Step-by-step Plan

### Step 1: Explore the workspace and locate all files
```bash
find /root -maxdepth 3 -type f | head -60
ls -la /root/
```
Identify where the template HWPX and JSON data files are located.

### Step 2: Read the JSON data files
```bash
cat <path_to>/audit_overview.json
cat <path_to>/corrective_actions.json
```
Note all field names, values, the risk tier value, the inspection date (in YYYY-MM-DD format), and the three corrective actions in their exact order.

### Step 3: Examine the HWPX template structure
HWPX files are ZIP archives. Unzip the template to inspect its contents:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to>/safety_audit_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
List all files inside the package. The main content is typically in XML files under `Contents/` (e.g., `Contents/section0.xml` or similar). There may also be header XML files.

### Step 4: Read and understand the XML content files
Read every XML file that contains text content, especially section XML files:
```bash
cat template_contents/Contents/section0.xml
```
(Adjust path based on what you find.) Look for:
- `{{...}}` placeholder patterns — note every single one
- Section titles and row labels (these must be preserved)
- The structure of the summary section and audit table
- The three corrective-action lines
- Any layout-cache elements (often `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, or `<hp:LineSeg>` elements inside paragraph tags)

Also check ALL other XML files in the package for any occurrences of `{{` placeholders or the risk tier / date values. Use:
```bash
grep -rl '{{' template_contents/
grep -rl 'lineseg\|LineSeg\|lineSegArray\|linesegarray' template_contents/ --ignore-case
```

### Step 5: Write a Python script to perform all substitutions
Create a Python script that:

1. **Copies the template HWPX** to the output path.
2. **Opens it as a ZIP** and reads all XML files.
3. **Loads both JSON files** to get the data values.
4. **Performs placeholder substitutions** in every XML file:
   - Replace each `{{placeholder}}` with the corresponding value from the JSON data.
   - For the inspection date: convert from `YYYY-MM-DD` to `YYYY.MM.DD` format. Replace ALL occurrences of the date (both placeholder substitutions and any already-present date strings).
   - For the risk tier: replace ALL occurrences throughout the document. After each risk tier value, append the severity note using this mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`. The note should be added as ` 즉시조치` (or appropriate) immediately after the risk tier text, like `High 즉시조치`.
   - Fill the three corrective-action lines in the same order they appear in `corrective_actions.json`.
5. **Removes layout-cache elements** from any paragraph (`<hp:p>` or similar) whose text content was modified. These are typically `<hp:linesegarray>` or `<hp:lineSegArray>` elements (and their children). Use XML parsing (lxml or xml.etree.ElementTree) to properly handle namespaces and remove these elements.
6. **Verifies no `{{...}}` placeholders remain** anywhere in the output.
7. **Writes the modified ZIP** to `/root/safety_audit_brief_final.hwpx`.

IMPORTANT implementation details:
- Use `zipfile` module to read/write the HWPX package.
- Parse XML with `lxml.etree` or `xml.etree.ElementTree` with proper namespace handling.
- When removing lineseg/layout-cache elements, search case-insensitively for tag names containing `lineseg` or `LineSeg` or `lineSegArray` across all namespace variants.
- Preserve all other XML structure, attributes, and non-modified content exactly.
- When writing the new ZIP, preserve the same directory structure and include ALL files from the original (not just XML files — include images, settings, mimetype, etc.).
- Do NOT use compression for the `mimetype` file if one exists (store it uncompressed, as per ODF/package conventions).

### Step 6: Run the script
```bash
python3 /tmp/hwpx_fill.py
```

### Step 7: Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/safety_audit_brief_final.hwpx

# Check no placeholders remain
unzip -p /root/safety_audit_brief_final.hwpx | grep -o '{{[^}]*}}' || echo 'No placeholders found - GOOD'

# Check the date format is YYYY.MM.DD (not YYYY-MM-DD)
unzip -p /root/safety_audit_brief_final.hwpx | grep -oP '\d{4}-\d{2}-\d{2}' && echo 'WARNING: Old date format found' || echo 'Date format OK'

# Check risk tier + severity note appears
unzip -p /root/safety_audit_brief_final.hwpx | grep -oP '(High|Medium|Low)\s+(즉시조치|계획보완|모니터링)' && echo 'Severity notes found - GOOD'

# Check no lineseg elements in modified paragraphs
# Unzip and inspect the section XML
mkdir -p /tmp/hwpx_verify
unzip -o /root/safety_audit_brief_final.hwpx -d /tmp/hwpx_verify
find /tmp/hwpx_verify -name '*.xml' -exec grep -li 'lineseg\|LineSeg' {} \;
```

If any lineseg elements remain, verify they are only in paragraphs that were NOT modified. If placeholders remain, fix and re-run.

### Step 8: Final check
Confirm the file exists at the correct path:
```bash
ls -la /root/safety_audit_brief_final.hwpx
file /root/safety_audit_brief_final.hwpx
```

## Critical Reminders
- Read ALL XML files in the HWPX package, not just the obvious ones. Placeholders or date/risk-tier text could appear in headers, footers, or other section files.
- The severity note must appear IMMEDIATELY AFTER EVERY occurrence of the risk tier text, not just the first one.
- The date must be reformatted EVERYWHERE it appears, not just in placeholders.
- Section titles and row labels must NOT be changed.
- The corrective actions must be in the EXACT order from the JSON file.
- Layout-cache removal: for EVERY paragraph you modify, remove lineseg-related child elements. Be thorough with namespace handling — the namespace prefix might vary.
- The output must be a proper ZIP file with `.hwpx` extension at exactly `/root/safety_audit_brief_final.hwpx`.

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