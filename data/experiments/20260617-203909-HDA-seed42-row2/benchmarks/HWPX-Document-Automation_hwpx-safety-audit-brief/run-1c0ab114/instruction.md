# Task Instruction

# Task: Prepare Warehouse Safety Audit Brief (HWPX)

## Overview
You need to fill a HWPX template with data from two JSON files and save the result. HWPX files are ZIP archives containing XML files (similar to DOCX/ODF). The main content is typically in XML files within the archive.

## Step-by-step Plan

### Step 1: Examine the workspace and data files
```bash
cd /root
ls -la
cat audit_overview.json
cat corrective_actions.json
```
Understand the JSON structure - note all field names, values, the risk tier value, and the inspection date format.

### Step 2: Examine the HWPX template structure
```bash
# HWPX is a ZIP archive
cp safety_audit_template.hwpx safety_audit_template_backup.hwpx
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip /root/safety_audit_template.hwpx
find . -type f | sort
```
List all files in the archive to understand the package structure.

### Step 3: Inspect all XML content files
Read every XML file in the extracted archive, especially:
- Files in `Contents/` directory (commonly `section0.xml` or similar)
- Any content XML files that contain the document text
- Look for `{{...}}` placeholder patterns

```bash
# Search for all placeholders
grep -r '{{' . --include='*.xml' -l
grep -r '{{' . --include='*.xml'
```

Also search for any layout-cache or char-shape-cache elements:
```bash
grep -rn 'linesegarray\|lineSegArray\|LineSeg\|layoutCache\|charShapeCache\|LINESEG\|hp:lineseg\|hp:lineSegArray' . --include='*.xml' | head -40
```

### Step 4: Understand the placeholder-to-data mapping
Map each `{{placeholder}}` in the template to the corresponding value from `audit_overview.json` or `corrective_actions.json`. Note:
- Overview fields go in the summary section
- Value cells go in the audit table
- Three corrective-action lines must be filled in the order from `corrective_actions.json`

### Step 5: Write a Python script to perform all replacements
Create a Python script that:

1. **Loads both JSON files** and extracts all needed values.
2. **Extracts the HWPX ZIP** to a temp directory.
3. **Reads each XML content file** and performs replacements:
   - Replace all `{{...}}` placeholders with corresponding JSON values
   - **Date format**: Convert any `YYYY-MM-DD` date to `YYYY.MM.DD` format (replace hyphens with dots). Do this EVERYWHERE in the document, not just in placeholders.
   - **Risk tier**: Update every occurrence of the risk tier value.
   - **Severity note**: Immediately after each occurrence of the risk tier text, append a severity note based on the mapping: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`. Use a space or appropriate separator between the tier and the note.
4. **Remove layout-cache elements** from any paragraph (`<hp:p>` or similar) whose text content was modified. These are typically `<hp:linesegarray>...</hp:linesegarray>` or `<hp:lineSegArray>...</hp:lineSegArray>` elements inside paragraphs. Remove them entirely so the word processor recalculates layout.
5. **Verify no `{{...}}` placeholders remain** in any XML file.
6. **Repackage the ZIP** as a valid HWPX file at `/root/safety_audit_brief_final.hwpx`.

**CRITICAL repackaging notes:**
- Use `zipfile` module in Python
- Preserve the original ZIP structure exactly (same directory paths, same file names)
- Use `ZIP_DEFLATED` compression
- If there's a `mimetype` file, it should typically be stored first and uncompressed (like ODF)
- Preserve all non-XML files (images, settings, etc.) byte-for-byte

### Step 6: Run the script
```bash
python3 /tmp/fill_template.py
```

### Step 7: Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/safety_audit_brief_final.hwpx

# Extract and verify no placeholders remain
mkdir -p /tmp/hwpx_verify
cd /tmp/hwpx_verify
unzip /root/safety_audit_brief_final.hwpx
grep -r '{{' . --include='*.xml'
# Should return nothing

# Verify date format is YYYY.MM.DD (no YYYY-MM-DD remaining)
grep -rP '\d{4}-\d{2}-\d{2}' . --include='*.xml'
# Should return nothing

# Verify severity notes are present
grep -r '즉시조치\|계획보완\|모니터링' . --include='*.xml'

# Verify section titles and row labels are preserved
# (Inspect the content XML to confirm structure is intact)

# Verify layout cache elements removed from modified paragraphs
grep -rn 'linesegarray\|lineSegArray\|LineSeg' . --include='*.xml'
```

## Key Cautions
- The HWPX XML namespace is typically `hp:` or similar Korean-specific namespace. Inspect actual tag names before writing replacement code.
- When removing layout-cache elements, use XML parsing (lxml or ElementTree) rather than regex, to avoid breaking XML structure.
- The corrective actions must appear in the SAME ORDER as in the JSON file.
- Keep ALL existing section titles and row labels unchanged.
- The severity note should be appended right after the risk tier text (e.g., if risk tier is "High", it becomes "High 즉시조치" or similar - check the context to determine appropriate formatting).
- Make sure to handle the case where risk tier and date might appear in multiple places throughout the document.

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