# Task Instruction

You will prepare a warehouse safety audit brief by filling a HWPX template with data from two JSON files, then saving the result.

## Steps

### 1. Inspect the workspace
```bash
ls /root/
find /root/ -name '*.json' -o -name '*.hwpx' | head -30
```
Identify the exact paths of `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

### 2. Read the JSON data files
```bash
cat <path>/audit_overview.json
cat <path>/corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The inspection date (will need `YYYY-MM-DD` → `YYYY.MM.DD` conversion)
- The risk tier value (will need a severity note appended)
- The order of corrective actions (must be preserved exactly)

### 3. Unpack the HWPX template
HWPX is a ZIP archive containing XML files.
```bash
mkdir /tmp/hwpx_work
cp <path>/safety_audit_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f | sort
```
List all files to understand the package structure.

### 4. Examine the XML content files
Look at each XML file that contains document body content (typically under `Contents/` or similar). Search for `{{` placeholders:
```bash
grep -rn '{{' template_contents/
```
Also inspect the full XML of the main content file(s) to understand the section structure, table layout, paragraph tags, and `<hp:linesegarray>` elements.

### 5. Write a Python script to perform all modifications
Create `/tmp/hwpx_work/fill_template.py` that does the following:

a) **Load both JSON files** and extract all values.

b) **Read each XML content file** that contains placeholders.

c) **Replace all `{{...}}` placeholders** with corresponding JSON values:
   - Summary/overview fields from `audit_overview.json`
   - Audit table value cells from `audit_overview.json`
   - Three corrective-action lines from `corrective_actions.json` in their original order

d) **Date reformatting**: Find EVERY occurrence of the inspection date in `YYYY-MM-DD` format and replace with `YYYY.MM.DD` (replace hyphens with dots). This includes both the filled-in values AND any occurrences that were already in the template.

e) **Risk tier + severity note**: For the risk tier value (e.g., `High`, `Medium`, or `Low`), append the Korean severity note using this mapping:
   - `High` → `즉시조치`
   - `Medium` → `계획보완`  
   - `Low` → `모니터링`
   
   Format: `<RiskTier> (<Note>)` — e.g., `High (즉시조치)`. Update EVERY occurrence of the risk tier in the document.

f) **Remove stale layout cache**: For any `<hp:p>` paragraph element whose text content was modified, remove all `<hp:linesegarray>` child elements (and their contents) from that paragraph. This prevents overlapping character rendering. Use an XML parser (lxml or ElementTree with namespace handling) for this step if feasible, or use careful regex if the namespace handling is too complex.

g) **Verify no `{{...}}` placeholders remain** in any content file.

h) **Write modified XML back** to the unpacked directory.

### 6. Repack the HWPX
```bash
cd /tmp/hwpx_work/template_contents
zip -r /root/safety_audit_brief_final.hwpx . -x '*.DS_Store'
```
Use `zip` with no extra compression metadata that would break the package. Make sure to zip from inside the directory so paths are relative.

### 7. Validate the output
```bash
# Confirm it's a valid ZIP
unzip -t /root/safety_audit_brief_final.hwpx

# Confirm no placeholders remain
unzip -p /root/safety_audit_brief_final.hwpx | grep -c '{{'
# Should be 0

# Confirm date format is YYYY.MM.DD (no YYYY-MM-DD left in content)
unzip -p /root/safety_audit_brief_final.hwpx | grep -oP '\d{4}-\d{2}-\d{2}'
# Should return nothing (or only non-date content matches)

# Confirm risk tier has severity note
unzip -p /root/safety_audit_brief_final.hwpx | grep -oP '(High|Medium|Low)\s*\('
# Should show the tier with opening paren

# Confirm no linesegarray in modified paragraphs
# (Check that linesegarray count decreased or is appropriate)
```

### 8. Run the verifier if available
```bash
cd <task_directory>
python -m pytest test_output.py -v 2>&1 | head -80
```
If tests fail, read the assertion errors carefully, fix the specific issue, and re-run.

## Critical Reminders
- The severity note format matters: use exactly `<Tier> (<Note>)` with a space before the parenthesis.
- Corrective actions must appear in the SAME ORDER as in the JSON file.
- ALL occurrences of the date and risk tier must be updated, not just the first.
- Section titles and row labels must NOT be changed.
- The `<hp:linesegarray>` removal is essential — only remove from paragraphs you actually modified.
- The final file must be at exactly `/root/safety_audit_brief_final.hwpx`.

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