# Task Instruction

## Task: Generate `/root/safety_audit_brief_final.hwpx` from template and JSON data

### Step 0 — Explore the workspace
```
ls /root/
find /root/ -name '*.hwpx' -o -name '*.json' | head -30
```
Identify the exact paths of:
- `safety_audit_template.hwpx`
- `audit_overview.json`
- `corrective_actions.json`

Also locate the test file (`/tests/test_outputs.py` or similar) and read it fully to understand every assertion the verifier makes.

### Step 1 — Understand the HWPX structure
A `.hwpx` file is a ZIP archive. Unzip the template to a temp directory:
```
mkdir -p /tmp/hwpx_work
cp <path>/safety_audit_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -o template.hwpx -d template_extracted
find template_extracted -type f
```
Read every XML file, especially `Contents/section0.xml` and `Contents/section1.xml` (or however sections are named). Print their full contents. Identify:
- All `{{...}}` placeholder tokens and their exact names
- Where placeholders appear in the XML element tree
- Which section contains the overview/summary fields
- Which section contains the audit table
- Which section contains corrective-action lines
- Every occurrence of the risk-tier placeholder
- Every occurrence of the inspection-date placeholder

### Step 2 — Read JSON data
```
cat <path>/audit_overview.json
cat <path>/corrective_actions.json
```
Note all field names and values. Pay special attention to:
- The risk tier value (e.g., `"High"`, `"Medium"`, or `"Low"`)
- The inspection date format (expected `YYYY-MM-DD`)
- The corrective actions list and their order

### Step 3 — Build the substitution script
Write a Python script `/tmp/hwpx_work/build.py` that:

1. **Copies** the template ZIP and extracts it.
2. **Reads** both JSON files.
3. **Parses** each section XML with `lxml.etree` (or `xml.etree.ElementTree`).
4. **Performs substitutions** on the text content of every element and tail text:
   - Replace each `{{placeholder}}` with the corresponding JSON value.
   - For the **risk tier** placeholder: substitute with `<RiskValue> (<SeverityNote>)` — e.g., if risk tier is `"High"`, the replacement text must be exactly `High (즉시조치)`. Use this mapping:
     - `High` → `즉시조치`
     - `Medium` → `계획보완`
     - `Low` → `모니터링`
   - **CRITICAL**: The severity note MUST include the parentheses as part of the string: `High (즉시조치)`, not `High 즉시조치` or `High（즉시조치）`. Use standard ASCII parentheses `(` and `)`.
   - For the **inspection date**: convert from `YYYY-MM-DD` to `YYYY.MM.DD` (replace hyphens with dots). Do this for EVERY occurrence across all XML files.
5. **Fills corrective-action lines** in the same order as `corrective_actions.json`.
6. **Removes stale layout-cache elements**: For any `<hp:linesegarray>` (or similar layout-cache elements like `<lineseg>`, `<lineSegArray>`) that are children/descendants of paragraphs whose text was modified, remove them entirely. Check the actual namespace and tag names in the XML first.
7. **Verifies** no `{{` or `}}` remains in any text node across all XML files.
8. **Re-packs** the HWPX: re-creates the ZIP archive with the same directory structure and compression, saving to `/root/safety_audit_brief_final.hwpx`.

### Step 4 — Important implementation details
- When walking XML elements, iterate over ALL elements and check both `.text` and `.tail` for placeholders.
- When replacing the risk tier, search for ALL occurrences across ALL section files, not just one.
- The date replacement should also be global across all XML files.
- When repacking the ZIP, use `zipfile.ZipFile` with `ZIP_DEFLATED`. Walk the extracted directory and add files with their relative paths matching the original structure. Preserve the `mimetype` file if present (store it uncompressed as first entry if that's how the original was structured).
- After building, verify the output by unzipping it and grep-ing for `{{` and for the expected strings like `High (즉시조치)` and the reformatted date.

### Step 5 — Run and validate
```
python3 /tmp/hwpx_work/build.py
```
Then verify:
```
# Check no placeholders remain
unzip -p /root/safety_audit_brief_final.hwpx Contents/section0.xml | grep -c '{{'
unzip -p /root/safety_audit_brief_final.hwpx Contents/section1.xml | grep -c '{{'

# Check severity note format
unzip -p /root/safety_audit_brief_final.hwpx Contents/section0.xml Contents/section1.xml | grep -o 'High (즉시조치)\|Medium (계획보완)\|Low (모니터링)'

# Check date format
unzip -p /root/safety_audit_brief_final.hwpx Contents/section0.xml Contents/section1.xml | grep -oP '\d{4}\.\d{2}\.\d{2}'
```

### Step 6 — Run the verifier tests
```
cd / && python -m pytest tests/test_outputs.py -v 2>&1
```
If any assertion fails, read the error carefully, inspect the relevant XML section content, fix the script, and re-run. Do NOT mark complete until all tests pass.

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