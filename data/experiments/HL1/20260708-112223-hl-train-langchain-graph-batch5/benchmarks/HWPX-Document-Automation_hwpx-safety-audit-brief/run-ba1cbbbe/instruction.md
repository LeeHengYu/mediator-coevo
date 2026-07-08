# Task Instruction

## Task: Prepare Warehouse Safety Audit Brief HWPX

### Goal
Fill the template `safety_audit_template.hwpx` using data from `audit_overview.json` and `corrective_actions.json`, then save the result to `/root/safety_audit_brief_final.hwpx`.

### Step-by-step Instructions

#### 1. Explore the workspace
```bash
find /root -maxdepth 3 -type f | head -60
```
Identify the template HWPX file and the two JSON data files. Note their exact paths.

#### 2. Inspect the JSON data files
```bash
cat <path>/audit_overview.json
cat <path>/corrective_actions.json
```
Record every field name and value. Pay special attention to:
- The **risk tier** value (e.g., `"High"`, `"Medium"`, or `"Low"`)
- The **inspection date** format (should be `YYYY-MM-DD`)
- The corrective action items and their order

#### 3. Unpack the HWPX template
HWPX is a ZIP-based package.
```bash
mkdir /tmp/hwpx_work
cp <path>/safety_audit_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
List all files. The XML content files are typically under `Contents/` (e.g., `section0.xml`, `section1.xml`).

#### 4. Inspect ALL XML section files
```bash
cat template_contents/Contents/section0.xml
cat template_contents/Contents/section1.xml
```
(Also check for any other section*.xml files.)

Identify:
- All `{{...}}` placeholder tokens and what they map to
- Where the risk tier appears (every occurrence)
- Where the inspection date appears (every occurrence)
- The summary/overview section fields
- The audit table value cells
- The three corrective-action lines
- Any `<hp:linesegarray>` or similar layout-cache elements inside `<hp:p>` tags

#### 5. Build and run a Python script to perform all replacements

Write a single Python script `/tmp/hwpx_work/fill_template.py` that:

**a) Loads both JSON files.**

**b) Defines the severity mapping:**
```python
severity_map = {"High": "즉시조치", "Medium": "계획보완", "Low": "모니터링"}
```

**c) For each section XML file, performs these replacements:**

1. **Replace all `{{placeholder}}` tokens** with corresponding values from `audit_overview.json` and `corrective_actions.json`. Map each placeholder to the correct JSON field.

2. **Risk tier with severity note (CRITICAL FORMAT):**
   - Every occurrence of the raw risk tier value (e.g., `High`) must become `High (즉시조치)` — that is: `{value} ({severity_map[value]})`.
   - Use **parentheses** around the Korean severity note, with a space before the opening parenthesis.
   - This includes occurrences that came from placeholder replacement AND any pre-existing occurrences.
   - Do NOT double-apply: if the text already contains `High (즉시조치)`, skip it.

3. **Rewrite inspection date format:**
   - Convert every `YYYY-MM-DD` date to `YYYY.MM.DD` (replace hyphens with dots).
   - Apply to all occurrences across the document.

4. **Corrective actions:** Fill the three corrective-action lines in the exact order they appear in `corrective_actions.json`.

5. **Remove stale layout-cache elements:** For any `<hp:p>` paragraph whose text content was modified, remove `<hp:linesegarray>...</hp:linesegarray>` elements (and any similar layout cache like `<hp:lineSegArray>`) from within that paragraph. Use XML parsing or careful regex to do this.

6. **Verify no `{{...}}` placeholders remain** in the output XML. Assert this.

**d) Repackage the HWPX:**
```python
import zipfile, os
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('template_contents'):
        for f in files:
            full = os.path.join(root, f)
            arcname = os.path.relpath(full, 'template_contents')
            zf.write(full, arcname)
```

Run the script:
```bash
cd /tmp/hwpx_work
python fill_template.py
```

#### 6. Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/safety_audit_brief_final.hwpx

# Extract and inspect the section XMLs
mkdir /tmp/verify
cd /tmp/verify
unzip /root/safety_audit_brief_final.hwpx -d verify_contents

# Check for remaining placeholders
grep -r '{{' verify_contents/ && echo 'FAIL: placeholders remain' || echo 'OK: no placeholders'

# Check severity format is correct (CRITICAL)
grep -r '즉시조치\|계획보완\|모니터링' verify_contents/Contents/
# Must show pattern like: High (즉시조치)  or Medium (계획보완) etc.

# Check date format
grep -rE '[0-9]{4}\.[0-9]{2}\.[0-9]{2}' verify_contents/Contents/
# Should find dates in YYYY.MM.DD format

# Check NO old date format remains
grep -rE '[0-9]{4}-[0-9]{2}-[0-9]{2}' verify_contents/Contents/ && echo 'FAIL: old date format remains' || echo 'OK: dates converted'

# Print full XML content for visual inspection
cat verify_contents/Contents/section0.xml
cat verify_contents/Contents/section1.xml
```

### Critical Reminders
- **Severity format MUST be `Value (Korean)` with parentheses** — e.g., `High (즉시조치)`. The verifier checks for this exact pattern. Previous failures were caused by omitting parentheses.
- The risk tier replacement must cover EVERY occurrence in ALL section XML files.
- Keep all existing section titles and row labels intact.
- The corrective actions must appear in the same order as in the JSON file.
- Remove layout cache (`linesegarray`/`lineSegArray`) from modified paragraphs to prevent overlapping characters.

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