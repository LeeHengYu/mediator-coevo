# Task Instruction

## Task: Prepare Warehouse Safety Audit Brief (HWPX)

You must fill in a template HWPX document using data from two JSON files, then save the result.

### Step 0 – Explore the workspace
```bash
find /root -maxdepth 3 -type f | head -80
```
Identify the exact paths for:
- `safety_audit_template.hwpx`
- `audit_overview.json`
- `corrective_actions.json`
- Any test/verifier scripts (e.g. `test_output.py`)

### Step 1 – Read the JSON data files
```bash
cat <path>/audit_overview.json
cat <path>/corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The inspection date (will be in `YYYY-MM-DD` format – must become `YYYY.MM.DD` everywhere)
- The risk tier value (e.g. "High", "Medium", or "Low")
- The corrective actions list and their order

### Step 2 – Inspect the HWPX template structure
HWPX is a ZIP package. Unzip it and examine the XML contents:
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
python3 -c "
import zipfile, os
with zipfile.ZipFile('<path>/safety_audit_template.hwpx', 'r') as z:
    z.extractall('/tmp/hwpx_work/template')
    for name in z.namelist():
        print(name)
"
```
Then read every XML file that contains `<hp:t>` text runs, especially the main content XML (likely `Contents/section0.xml` or similar). Look for `{{...}}` placeholders.

**Critical**: Placeholders may be fragmented across multiple `<hp:t>` tags (e.g., `<hp:t>{{</hp:t><hp:t>field</hp:t><hp:t>}}</hp:t>`). You MUST handle this.

### Step 3 – Write a Python script to perform all replacements

Create `/tmp/hwpx_work/build.py` that does the following:

1. **Read both JSON files** into Python dicts/lists.

2. **Read the section XML** file(s) as raw text.

3. **Defragment placeholders**: Use a regex approach to reconstruct placeholder tokens that may be split across XML tags. Specifically:
   - Find all `<hp:run ...>...</hp:run>` elements.
   - Within each run, concatenate all `<hp:t>` text content to find the logical text.
   - If the logical text contains `{{...}}` patterns, rebuild the run so the placeholder text is in a single `<hp:t>` tag (preserve the first tag's attributes).
   - Alternatively, use a regex like `\{\{(?:[^}]|<[^>]*>)*?\}\}` to match placeholders spanning tags, then flatten them.

4. **Replace all `{{...}}` placeholders** with the corresponding values from the JSON data:
   - Overview/summary fields from `audit_overview.json`
   - Audit table value cells from `audit_overview.json`
   - Three corrective-action lines from `corrective_actions.json` **in order**

5. **Date format**: Convert every occurrence of the inspection date from `YYYY-MM-DD` to `YYYY.MM.DD` in the entire XML text.

6. **Risk tier + severity note**: For every occurrence of the risk tier string, append the Korean severity note:
   - High → " 즉시조치"
   - Medium → " 계획보완"
   - Low → " 모니터링"
   So e.g. "High" becomes "High 즉시조치". Make sure this applies everywhere the risk tier appears.

7. **Remove layout caches**: Strip all `<hp:lineSegArray>...</hp:lineSegArray>` elements from the XML. This prevents overlapping-character rendering issues. Use regex: `re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', xml_text, flags=re.DOTALL)`

8. **Verify no remaining placeholders**: Assert that `{{` does not appear anywhere in the final XML. If it does, print which placeholders remain and abort.

9. **Rebuild the HWPX ZIP**: Using Python's `zipfile` module, create `/root/safety_audit_brief_final.hwpx` containing all original files from the template, but with the modified XML file(s) replacing the originals. Preserve the directory structure. Use `ZIP_DEFLATED` compression.

### Step 4 – Execute the script
```bash
python3 /tmp/hwpx_work/build.py
```
Check for errors. If any `{{...}}` placeholders remain, debug by printing the surrounding XML context and fix.

### Step 5 – Validate the output
```bash
# Verify it's a valid ZIP
python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx','r'); print(z.namelist()); z.close()"

# Verify no remaining placeholders
python3 -c "
import zipfile
with zipfile.ZipFile('/root/safety_audit_brief_final.hwpx','r') as z:
    for name in z.namelist():
        if name.endswith('.xml'):
            data = z.read(name).decode('utf-8')
            if '{{' in data:
                print(f'REMAINING PLACEHOLDER in {name}')
            else:
                print(f'{name}: OK')
"

# Check date format is YYYY.MM.DD not YYYY-MM-DD
# Check risk tier has severity note appended
# Check no lineSegArray elements remain
```

### Step 6 – Run verifier if present
```bash
cd <task_directory>
python3 -m pytest test_output.py -v 2>&1 | head -100
```
If tests fail, read the failure output carefully, fix, and re-run.

### Important Reminders
- **Placeholder fragmentation is the #1 risk.** Always defragment before replacing.
- **Layout cache removal is mandatory.** Strip `<hp:lineSegArray>` elements.
- **Keep existing section titles and row labels** – only replace placeholder values.
- **Corrective actions must be in the same order** as in the JSON file.
- The severity note goes immediately after the risk tier text, separated by a space.
- The output path MUST be exactly `/root/safety_audit_brief_final.hwpx`.

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