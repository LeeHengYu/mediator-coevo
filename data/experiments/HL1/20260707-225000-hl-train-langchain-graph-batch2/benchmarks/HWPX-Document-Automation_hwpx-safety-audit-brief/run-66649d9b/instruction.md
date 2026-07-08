# Task Instruction

Complete the following task to produce `/root/safety_audit_brief_final.hwpx` from the template and JSON data files.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
```
Identify the template file `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

### 2. Examine the JSON data files
```bash
cat /root/audit_overview.json
cat /root/corrective_actions.json
```
Note all field names, values, the risk tier value, and the inspection date format (YYYY-MM-DD).

### 3. Examine the HWPX template structure
```bash
cd /root && mkdir -p hwpx_work && cp safety_audit_template.hwpx hwpx_work/ && cd hwpx_work
python3 -c "import zipfile; z=zipfile.ZipFile('safety_audit_template.hwpx','r'); print(z.namelist())"
```
Then extract and examine the main content XML (likely `Contents/section0.xml`):
```bash
python3 -c "import zipfile; z=zipfile.ZipFile('safety_audit_template.hwpx','r'); print(z.read('Contents/section0.xml').decode('utf-8'))"
```
Identify all `{{...}}` placeholders, their exact names, and how they map to the JSON fields. Also note any other XML files that might contain placeholders.

### 4. Write and run the Python transformation script

Create a Python script `/root/build_hwpx.py` that does the following:

1. **Load JSON data** from `audit_overview.json` and `corrective_actions.json`.
2. **Copy the template HWPX** to the output path.
3. **Open the HWPX as a zip**, read `Contents/section0.xml` (and check any other XML files for placeholders).
4. **Replace all `{{...}}` placeholders** with the corresponding values from the JSON data:
   - Map overview fields to their placeholders in the summary section.
   - Map corrective action fields to their placeholders, maintaining the order from `corrective_actions.json`.
   - After replacement, verify no `{{` or `}}` remains in any XML content.
5. **Reformat the inspection date**: Replace every occurrence of the date in `YYYY-MM-DD` format with `YYYY.MM.DD` format (e.g., `2024-03-15` → `2024.03.15`). Do this globally across all text content.
6. **Add severity note after risk tier**: Wherever the risk tier value appears (e.g., "High", "Medium", or "Low"), append the Korean severity note immediately after it using this mapping:
   - `High` → ` 즉시조치`
   - `Medium` → ` 계획보완`  
   - `Low` → ` 모니터링`
   
   Be careful to only add the note once per occurrence and not to double-add if running multiple passes. The note should appear right after the risk tier text, separated by a space.
7. **Remove all `lineSegArray` elements** (layout cache) from the XML. Use regex: `re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', xml_content, flags=re.DOTALL)`. Also handle the `<lineSegArray>` variant without namespace prefix if present. This is CRITICAL — without this step, HWP viewers will display stale cached text.
8. **Repackage the HWPX**: Create a new zip file at `/root/safety_audit_brief_final.hwpx` containing all original files, with the modified XML replacing the original. Preserve the zip structure exactly (same file paths). Use `zipfile.ZIP_DEFLATED` compression.

### 5. Run the script
```bash
python3 /root/build_hwpx.py
```

### 6. Validate the output
```bash
# Check the file exists and is a valid zip
python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx','r'); print('Valid zip, files:', z.namelist())"

# Check no placeholders remain
python3 -c "
import zipfile
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx', 'r')
for name in z.namelist():
    if name.endswith('.xml'):
        content = z.read(name).decode('utf-8')
        if '{{' in content or '}}' in content:
            print(f'PLACEHOLDER FOUND in {name}')
        else:
            print(f'{name}: clean')
        if 'lineSegArray' in content:
            print(f'WARNING: lineSegArray still present in {name}')
"

# Check date format is correct (YYYY.MM.DD, not YYYY-MM-DD)
python3 -c "
import zipfile, re
z = zipfile.ZipFile('/root/safety_audit_brief_final.hwpx', 'r')
for name in z.namelist():
    if name.endswith('.xml'):
        content = z.read(name).decode('utf-8')
        dates_dash = re.findall(r'\d{4}-\d{2}-\d{2}', content)
        dates_dot = re.findall(r'\d{4}\.\d{2}\.\d{2}', content)
        if dates_dash:
            print(f'WARNING: YYYY-MM-DD dates still in {name}: {dates_dash}')
        if dates_dot:
            print(f'OK: YYYY.MM.DD dates in {name}: {dates_dot}')
"
```

### 7. Run the verifier if available
```bash
cd /root && ls test_output.py 2>/dev/null && python3 -m pytest test_output.py -v
```

### Important Notes
- When replacing placeholders, be aware that HWPX XML may split placeholder text across multiple `<hp:t>` runs (e.g., `{{audit` in one run and `_date}}` in another). If simple string replacement on the full XML doesn't catch all placeholders, you may need to first concatenate adjacent text runs or use a more sophisticated approach.
- Keep all existing section titles and row labels unchanged.
- The severity note mapping is case-sensitive to the risk tier value from the JSON.
- Ensure the corrective actions appear in the same order as in `corrective_actions.json`.

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