# Task Instruction

You must produce `/root/safety_audit_brief_final.hwpx` from the template and two JSON data files.

## Step 0 – Reconnaissance

1. `ls /root/` to see all provided files.
2. `cat /root/audit_overview.json` and `cat /root/corrective_actions.json` – record every field name and value.
3. Inspect the template: `cp /root/safety_audit_template.hwpx /tmp/template.hwpx && cd /tmp && unzip -o template.hwpx -d template_contents && find template_contents -type f`
4. Read every XML file found (especially files matching `section*.xml` and `content.hpf` or similar). Record:
   - Every `{{...}}` placeholder and its exact text.
   - Every location where the risk tier value or date might appear.
   - The XML namespace declarations.
   - Any `<hp:linesegarray>`, `<hp:lineBreakList>`, or similar layout-cache elements inside `<hp:run>` or `<hp:p>` tags.

5. Read `test_output.py` (or `test_outputs.py`) in the task directory to discover the **exact assertions** the verifier makes. Pay special attention to:
   - The exact string format for the severity note (the previous failure showed the test expects `High (즉시조치)` with parentheses, not just `High 즉시조치`).
   - How the test loads the HWPX (unzips, reads section XMLs, concatenates text).
   - Which values it checks for.
   - Whether it checks that no `{{` remains.
   - Date format expectations.

## Step 1 – Determine the exact severity-note format

Based on the test assertions (from Step 0-5), determine the **exact** format the verifier expects for the risk-tier + severity note combination. The previous run failed because the format was wrong. The test expects `High (즉시조치)` (parenthesized). Confirm this by reading the test file. If the test expects a different format, use that format.

## Step 2 – Write a Python script to produce the final HWPX

Write and run a Python script `/tmp/build_hwpx.py` that does the following:

```python
import json, os, re, shutil, zipfile

# Load data
with open('/root/audit_overview.json') as f:
    overview = json.load(f)
with open('/root/corrective_actions.json') as f:
    actions = json.load(f)

# Copy template
shutil.copy('/root/safety_audit_template.hwpx', '/tmp/output.hwpx')

# Extract
extract_dir = '/tmp/hwpx_edit'
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir)
with zipfile.ZipFile('/tmp/output.hwpx', 'r') as z:
    z.extractall(extract_dir)

# Identify all XML files
xml_files = []
for root, dirs, files in os.walk(extract_dir):
    for fname in files:
        if fname.endswith('.xml'):
            xml_files.append(os.path.join(root, fname))

# Build replacement map from overview fields
# (This must be adapted based on actual placeholder names found in Step 0)
# ... fill in based on actual placeholders ...

# For each XML file:
for xf in xml_files:
    with open(xf, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Replace all {{...}} placeholders with corresponding data values
    # 2. Replace risk tier everywhere: e.g., replace the bare risk tier value
    #    with 'High (즉시조치)' (or appropriate tier+note)
    # 3. Rewrite date from YYYY-MM-DD to YYYY.MM.DD everywhere
    # 4. Fill corrective action lines in order
    # 5. Remove layout-cache elements from any paragraph whose text was modified:
    #    Remove <hp:linesegarray>...</hp:linesegarray> elements
    #    Remove <hp:lineBreakList>...</hp:lineBreakList> elements
    #    (Use regex to strip these from modified paragraphs, or from all paragraphs to be safe)
    
    with open(xf, 'w', encoding='utf-8') as f:
        f.write(content)

# Repackage as HWPX (ZIP)
output_path = '/root/safety_audit_brief_final.hwpx'
with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zout:
    for root, dirs, files in os.walk(extract_dir):
        for fname in files:
            full = os.path.join(root, fname)
            arcname = os.path.relpath(full, extract_dir)
            zout.write(full, arcname)

print('Done:', output_path)
```

**Critical details for the script:**

a) **Severity note format**: Use `RiskTier (KoreanNote)` with parentheses, e.g., `High (즉시조치)`, `Medium (계획보완)`, `Low (모니터링)`. Confirm against test assertions.

b) **Risk tier replacement**: Every occurrence of the raw risk tier string in the XML must be replaced with the tier + severity note. Be careful: if the placeholder is `{{risk_tier}}` or `{{risk_level}}`, replace it. Also do a second pass to catch any literal occurrences of just the tier name without the note.

c) **Date rewriting**: Convert every `YYYY-MM-DD` formatted date to `YYYY.MM.DD`. This includes both placeholder-injected dates and any that might already be literal in the template. Use a regex like `r'(\d{4})-(\d{2})-(\d{2})'` but be careful not to corrupt XML attribute dates that should stay as-is (check context).

d) **Corrective actions**: The JSON likely contains an array. Fill them in order into the three corrective-action placeholders.

e) **Layout cache removal**: For safety, remove ALL `<hp:linesegarray>` and `<hp:lineBreakList>` elements from any section XML file where text was modified. Use regex: `re.sub(r'<hp:linesegarray[^>]*>.*?</hp:linesegarray>', '', content, flags=re.DOTALL)` and similarly for lineBreakList. This prevents overlapping characters.

f) **No remaining placeholders**: After all replacements, verify no `{{` remains in any XML file. If any do, the script should print a warning and you must investigate.

## Step 3 – Validate

1. Unzip the output file and read the section XMLs to confirm:
   - No `{{` placeholders remain.
   - The risk tier + severity note appears in the expected format.
   - Dates are in `YYYY.MM.DD` format.
   - Corrective actions are present.
   - Layout cache elements are removed from modified sections.
2. Run the test suite: `cd /root && python -m pytest test_output.py -v` (or whatever the test file is named). If it fails, read the error, fix the script, and re-run.

## Step 4 – Iterate if needed

If the test fails, carefully read the assertion error, compare expected vs actual, adjust the build script, regenerate, and re-test. Do not stop until the test passes or you have exhausted all reasonable approaches.

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