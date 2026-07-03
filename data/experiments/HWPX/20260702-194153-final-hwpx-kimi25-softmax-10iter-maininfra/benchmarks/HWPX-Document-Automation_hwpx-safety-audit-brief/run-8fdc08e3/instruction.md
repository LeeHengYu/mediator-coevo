# Task Instruction

Complete the following task to prepare a warehouse safety audit brief from a HWPX template.

## Context
HWPX files are ZIP archives containing XML files (similar to DOCX). The main content is typically in XML files within the archive. You need to fill a template with data from JSON files.

## Step-by-step Instructions

### Step 1: Explore the workspace
```bash
cd /root
ls -la
find . -maxdepth 2 -type f | head -50
```
Identify the template file `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

### Step 2: Read the JSON data files
```bash
cat audit_overview.json
cat corrective_actions.json
```
Note all field values, especially: risk tier, inspection date (in YYYY-MM-DD format), and the three corrective actions in order.

### Step 3: Examine the HWPX template structure
Since HWPX is a ZIP archive:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('safety_audit_template.hwpx', 'r') as z:
    for name in z.namelist():
        print(name)
"
```
Then read each XML content file to understand the structure and find all `{{...}}` placeholders:
```bash
python3 -c "
import zipfile
with zipfile.ZipFile('safety_audit_template.hwpx', 'r') as z:
    for name in z.namelist():
        try:
            content = z.read(name).decode('utf-8')
            if '{{' in content or name.endswith('.xml'):
                print(f'=== {name} ===')
                print(content)
                print()
        except:
            print(f'=== {name} === (binary, skipped)')
"
```
Carefully catalog ALL `{{...}}` placeholders and their locations across ALL files in the archive.

### Step 4: Write a Python script to produce the final HWPX
Create a comprehensive Python script that:

1. Reads both JSON files.
2. Opens the template HWPX as a ZIP.
3. For each file in the ZIP, processes the content:
   a. Replaces ALL `{{...}}` placeholders with corresponding values from the JSON data.
   b. Fills overview fields in the summary section.
   c. Fills value cells in the audit table.
   d. Fills the three corrective-action lines in the order they appear in `corrective_actions.json`.
   e. Replaces EVERY occurrence of the risk tier value.
   f. Reformats the inspection date from `YYYY-MM-DD` to `YYYY.MM.DD` everywhere it appears (both in placeholders and after substitution).
   g. After inserting the risk tier, adds a severity note immediately after it using the mapping: High -> 즉시조치, Medium -> 계획보완, Low -> 모니터링. The note should appear as " (즉시조치)" or similar, directly after the risk tier text.
   h. Removes layout-cache elements (such as `<linesegarray>`, `<lineSegArray>`, `<hp:linesegarray>`, or similar cache elements) from any paragraph whose text content was modified. Look for elements like `<TEXT ... charShapeId` run containers that have associated `<LINESEGARRAY>` or layout caching siblings. Inspect the actual XML tag names used in the template.
4. Verifies no `{{` or `}}` remain in any file content.
5. Writes the result to `/root/safety_audit_brief_final.hwpx` as a valid ZIP with the same compression settings.

IMPORTANT DETAILS:
- When working with XML in HWPX, placeholders like `{{field_name}}` may be split across multiple XML text runs/spans. Check if placeholders are contiguous in a single text node or split across elements. If split, you may need to concatenate text across runs, perform replacement, then redistribute.
- Use `lxml` or `xml.etree.ElementTree` for XML parsing to handle namespaces properly.
- Preserve all XML namespaces, attributes, and structure except for the specific replacements and layout-cache removal.
- The severity note should be appended right after the risk tier text (e.g., if risk tier is "High", it becomes "High (즉시조치)").
- Ensure the date format conversion happens EVERYWHERE - both in placeholder substitutions and in any hardcoded dates.

### Step 5: Run the script
```bash
python3 build_hwpx.py
```

### Step 6: Validate the output
```bash
# Verify it's a valid ZIP
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx', 'r') as z:
    for name in z.namelist():
        content = z.read(name)
        try:
            text = content.decode('utf-8')
            # Check no placeholders remain
            if '{{' in text or '}}' in text:
                print(f'WARNING: Placeholder found in {name}')
                import re
                for m in re.finditer(r'\{\{.*?\}\}', text):
                    print(f'  Found: {m.group()}')
            # Check date format
            import re
            old_dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
            if old_dates:
                print(f'WARNING: Old date format in {name}: {old_dates}')
        except:
            pass
    print('Validation complete - file is a valid ZIP')
"
```

Also verify the content looks correct by printing the main content XML of the output file to confirm all substitutions landed properly, severity notes are present, dates are in YYYY.MM.DD format, and no layout-cache elements remain on modified paragraphs.

### Critical Reminders
- Do NOT leave any `{{...}}` placeholders in the final document.
- The severity note must appear immediately after EVERY occurrence of the risk tier.
- ALL dates must be in YYYY.MM.DD format (dot-separated).
- Section titles and row labels must be preserved exactly.
- Corrective actions must appear in the same order as in the JSON file.
- Layout-cache elements must be removed from modified paragraphs only.
- The output must be at `/root/safety_audit_brief_final.hwpx`.

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