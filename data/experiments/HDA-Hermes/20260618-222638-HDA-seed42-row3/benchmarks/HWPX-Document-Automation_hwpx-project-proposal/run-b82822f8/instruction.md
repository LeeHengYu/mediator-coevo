# Task Instruction

You will complete a project-proposal HWPX document by filling in placeholders from a JSON file, appending month-span annotations to phase lines, normalizing the budget value, and cleaning up layout caches. Follow these steps precisely:

### 1. Inspect the workspace
```bash
ls /root/
cat /root/project_proposal.json
```
Understand the JSON keys and values. Note the budget field (it will have commas to remove but keep the currency symbol like ₩).

### 2. Unpack the HWPX template
```bash
mkdir -p /tmp/hwpx_work
cd /tmp/hwpx_work
unzip -o /root/project_proposal_template.hwpx -d template_contents
```

### 3. Examine the XML section files
```bash
find /tmp/hwpx_work/template_contents -name '*.xml' | sort
```
Then read every XML file that could contain `{{` placeholders:
```bash
grep -rl '{{' /tmp/hwpx_work/template_contents/
```
Read each matched file in full so you know the exact XML structure, tag names, and placeholder locations.

### 4. Write a Python script to perform all edits
Create `/tmp/hwpx_work/fill_template.py` that does the following:

a) **Load the JSON** from `/root/project_proposal.json`.

b) **For each XML file** found under `template_contents/` that contains `{{`:
   - Read its full text.
   - Replace every `{{key}}` placeholder with the corresponding JSON value.
   - **Budget normalization**: For the budget value, remove all commas but keep the leading currency symbol (e.g., `₩500,000,000` → `₩500000000`). Apply this normalization *before* substitution so the replaced text is already clean.
   - **Phase month-span annotation**: After placeholder substitution, for lines/paragraphs containing phase schedule info (단계1, 단계2, 단계3), calculate the month span from the date range present in that text. The date range format is like `2025.01 ~ 2025.03`. Calculate months as: parse start and end as (year, month), compute `(end_year*12 + end_month) - (start_year*12 + start_month) + 1`, then append ` (N개월)` to the text of that paragraph/run. Use regex to find patterns like `(\d{4})\.(\d{2})\s*~\s*(\d{4})\.(\d{2})` and compute the span.
   - **Remove layout-cache elements**: Remove all `<hp:lineSegArray>...</hp:lineSegArray>` elements (and any similar layout cache tags like `<hp:linesegarray>` case-insensitively) from any paragraph whose text content was modified. This prevents overlapping character rendering. Use regex to strip these: `re.sub(r'<hp:lineSegArray[^>]*>.*?</hp:lineSegArray>', '', text, flags=re.DOTALL)`. Also handle the case where the tag might be in a different namespace prefix.
   - Write the modified XML back.

c) **Verify no `{{` remains** in any XML file. Print a confirmation or error.

### 5. Run the script
```bash
python3 /tmp/hwpx_work/fill_template.py
```

### 6. Verify the output
```bash
# Check no placeholders remain
grep -r '{{' /tmp/hwpx_work/template_contents/ && echo 'FAIL: placeholders remain' || echo 'OK: no placeholders'

# Check month spans are present
grep -r '개월' /tmp/hwpx_work/template_contents/

# Check budget is normalized (no commas after currency symbol)
grep -r '₩' /tmp/hwpx_work/template_contents/ || grep -r 'budget\|예산' /tmp/hwpx_work/template_contents/
```

### 7. Repackage the HWPX
```bash
cd /tmp/hwpx_work/template_contents
zip -r /root/project_proposal_ready.hwpx . -x '.*'
```
The zip must be created from *inside* the template_contents directory so the archive root matches the original structure (mimetype, META-INF/, Contents/, etc.).

### 8. Final validation
```bash
# Confirm the output exists and is a valid zip
file /root/project_proposal_ready.hwpx
python3 -c "import zipfile; z=zipfile.ZipFile('/root/project_proposal_ready.hwpx'); print('Valid zip, entries:', len(z.namelist())); z.close()"
```

### 9. Run the verifier if available
```bash
cd /root && ls test_output.py 2>/dev/null && python3 -m pytest test_output.py -v
```

### Important notes
- Read each XML file's actual content before editing. Do not assume structure.
- The phase annotation `(N개월)` must be appended *after* the date range on the same paragraph/run text, separated by a space.
- Budget: strip commas only from the numeric portion, preserve the currency symbol exactly.
- Korean labels and static note lines must remain untouched.
- Layout cache removal applies only to paragraphs you modified, but removing them globally is also acceptable.
- If the verifier test fails, read the error, inspect the expected vs actual values, and fix accordingly.

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