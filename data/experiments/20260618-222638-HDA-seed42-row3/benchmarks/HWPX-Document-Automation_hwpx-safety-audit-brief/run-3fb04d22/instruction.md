# Task Instruction

# Task: Prepare Warehouse Safety Audit Brief (HWPX)

## Goal
Fill the template `safety_audit_template.hwpx` with data from `audit_overview.json` and `corrective_actions.json`, then save the completed document to `/root/safety_audit_brief_final.hwpx`.

## Step-by-step Instructions

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
Note all field names, values, the risk tier, the inspection date (in YYYY-MM-DD format), and the three corrective actions (preserving their order).

### Step 3: Examine the HWPX template structure
HWPX files are ZIP archives. Unzip the template to inspect its contents:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to>/safety_audit_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip template.hwpx -d template_extracted
find template_extracted -type f
```
List all files in the package. The main content is typically in XML files under `Contents/` (e.g., `section0.xml` or similar). There may also be `content.hpf`, `header.xml`, `META-INF/` etc.

### Step 4: Inspect the XML content files for placeholders
Read each XML content file to find `{{...}}` placeholders:
```bash
grep -rn '{{' template_extracted/
```
Also read the full content of the main section XML file(s) to understand the document structure:
```bash
cat template_extracted/Contents/section0.xml
```
(Adjust path based on what you find.) Carefully note:
- All `{{placeholder}}` names and their locations
- The structure of the summary/overview section
- The structure of the audit table with value cells
- The three corrective-action lines
- Every occurrence of the risk tier placeholder
- Every occurrence of the inspection date placeholder
- Any `<hp:linesegarray>`, `<hp:lineSegArray>`, or `<hp:layoutcache>` or similar layout-cache elements within paragraphs

### Step 5: Create a Python script to perform all substitutions
Write a Python script that:

1. **Loads both JSON files** and extracts all needed values.

2. **Reads each XML file** in the extracted HWPX that contains `{{...}}` placeholders.

3. **Performs the following substitutions:**
   - Replace all overview field placeholders with values from `audit_overview.json`
   - Replace audit table value cell placeholders with appropriate values
   - Replace corrective-action line placeholders with the three actions from `corrective_actions.json` **in the same order they appear in that file**
   - Replace **every** occurrence of the risk tier placeholder with the actual risk tier value
   - Replace the inspection date: convert from `YYYY-MM-DD` to `YYYY.MM.DD` format **everywhere** it appears (both in placeholders and after substitution)
   - After each risk tier value, append a severity note using the mapping:
     - `High` → ` 즉시조치`
     - `Medium` → ` 계획보완`  
     - `Low` → ` 모니터링`
     
     So if risk tier is "High", every occurrence becomes "High 즉시조치". Make sure the note is appended with a space separator immediately after the risk tier text, in the same text run/element.

4. **Removes stale layout-cache elements** from any paragraph whose text was modified. Specifically:
   - Find `<hp:linesegarray>...</hp:linesegarray>` (or `<linesegarray>`, `<hp:lineSegArray>`, etc.) elements and remove them from modified paragraphs
   - Also remove any `<hp:layoutcache>` or similar caching elements from modified paragraphs
   - Use XML parsing (e.g., `xml.etree.ElementTree` with namespace awareness) to properly identify and remove these elements rather than regex if possible. But if the XML has complex namespaces, regex on the serialized XML is acceptable as long as it's precise.

5. **Verifies no `{{...}}` placeholders remain** in any XML file.

6. **Preserves all existing section titles and row labels** — only modify placeholder values, not structural text.

### Step 6: Repackage the HWPX
After modifying the XML files in place within the extracted directory, repackage as a valid HWPX (ZIP) file:
```python
import zipfile, os

def repackage_hwpx(source_dir, output_path):
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(source_dir):
            for f in files:
                full_path = os.path.join(root, f)
                arcname = os.path.relpath(full_path, source_dir)
                zf.write(full_path, arcname)
```
Save to `/root/safety_audit_brief_final.hwpx`.

**IMPORTANT for ZIP packaging:** If the original HWPX has a `mimetype` file, it should be stored first and uncompressed (ZIP_STORED), similar to ODF/EPUB conventions. Check the original ZIP structure:
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('/tmp/hwpx_work/template.hwpx'); [print(i.filename, i.compress_type) for i in z.infolist()]"
```
Replicate the same compression settings for each entry if possible.

### Step 7: Validate the output
```bash
# Verify it's a valid ZIP
unzip -t /root/safety_audit_brief_final.hwpx

# Verify no placeholders remain
unzip -p /root/safety_audit_brief_final.hwpx | grep -c '{{'
# Should output 0

# Verify the date format is YYYY.MM.DD (not YYYY-MM-DD)
unzip -p /root/safety_audit_brief_final.hwpx | grep -oP '\d{4}-\d{2}-\d{2}'
# Should find nothing (all converted to dot format)

unzip -p /root/safety_audit_brief_final.hwpx | grep -oP '\d{4}\.\d{2}\.\d{2}'
# Should find the inspection date in dot format

# Verify severity note appears
unzip -p /root/safety_audit_brief_final.hwpx | grep -E '즉시조치|계획보완|모니터링'
# Should find matches

# Verify no linesegarray in modified paragraphs (check overall)
unzip -p /root/safety_audit_brief_final.hwpx | grep -ci 'lineseg'
# Compare with original to confirm reduction
```

### Step 8: Final check
Read the main section XML from the final HWPX and visually confirm:
- Overview fields are filled
- Audit table values are filled
- Three corrective actions appear in correct order
- Risk tier has severity note appended
- Date is in YYYY.MM.DD format
- No `{{...}}` remains
- Section titles and row labels are preserved

## Critical Reminders
- **Namespace handling**: HWPX XML files use namespaces (commonly `hp:`, `hwpx:`, etc.). Be careful with namespace prefixes when parsing and modifying XML.
- **Text may be split across multiple `<hp:t>` or `<t>` elements** within a single run. A placeholder like `{{field}}` might span multiple text elements. Check for this and handle it (e.g., by concatenating text within a run before substitution, or by searching across adjacent elements).
- **Layout cache removal is critical**: If `<hp:linesegarray>` or similar elements are left in paragraphs where text length changed, the document will display with overlapping characters.
- **Order matters for corrective actions**: Use the exact order from the JSON array.
- **The severity note must appear immediately after the risk tier text**, separated by a space, in every location where the risk tier appears.

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