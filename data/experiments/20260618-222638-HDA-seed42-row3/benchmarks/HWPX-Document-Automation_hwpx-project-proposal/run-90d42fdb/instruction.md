# Task Instruction

Execute the following steps in order to produce `/root/project_proposal_ready.hwpx`.

## 1. Inspect the workspace

```bash
ls /root/
cat /root/project_proposal.json
```

Identify the JSON keys and values. Note the budget field (it likely has commas — you must strip commas but keep the currency symbol like `₩`).

## 2. Explore the HWPX template

HWPX files are ZIP archives. Unzip the template to a working directory:

```bash
mkdir -p /tmp/hwpx_work
cp /root/project_proposal_template.hwpx /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip -o template.hwpx -d template_extracted
find template_extracted -type f -name '*.xml' | sort
```

Then read every XML file that could contain `{{` placeholders or phase lines:

```bash
grep -rl '{{' template_extracted/
grep -rl '단계' template_extracted/
```

Read the full content of each matching XML file (likely `Contents/section0.xml` or similar):

```bash
cat template_extracted/Contents/section0.xml
```

Also check for any other section files. Read the content carefully.

## 3. Understand the XML structure around placeholders and phases

Critical: Placeholders like `{{project_name}}` might be split across multiple `<hp:t>` elements (e.g., `<hp:t>{{</hp:t><hp:t>project_name</hp:t><hp:t>}}</hp:t>`). If so, you must handle this.

- Read the raw XML bytes around each `{{` to see if they are contiguous or split.
- Also look at the phase lines (`단계1`, `단계2`, `단계3`) to see the exact date format used (dashes like `2026-08 ~ 2026-10` vs dots like `2025.01 ~ 2025.03`). The previous failure showed the test expects dash-format dates.

## 4. Understand layout-cache elements

Look for `<hp:linesegarray>` or `<hp:lineSegArray>` or similar layout-cache elements inside `<hp:run>` or `<hp:p>` elements. Any paragraph you modify must have these layout-cache elements removed so the document renders cleanly. Identify the exact tag names by inspecting the XML.

## 5. Write a Python script to perform the transformation

Create `/tmp/hwpx_work/transform.py` that:

a) Loads the JSON from `/root/project_proposal.json`.

b) For each XML file containing `{{` placeholders or phase lines:
   - Read the raw XML content as a UTF-8 string.
   - **First**, fix any split placeholders: collapse adjacent `<hp:t>` tags within the same `<hp:run>` that together form a `{{...}}` pattern. You can do this by:
     1. Concatenating all `<hp:t>` content within each `<hp:run>` to check if it contains `{{...}}`.
     2. If a run's combined text contains a placeholder, merge all `<hp:t>` elements in that run into one.
   - **Then**, replace each `{{key}}` with the corresponding JSON value. For the budget key, strip commas but keep the currency symbol (e.g., `₩500,000,000` → `₩500000000`).
   - **Then**, for each phase line containing `단계1`, `단계2`, or `단계3` (or their descriptive text) with a date range, calculate the month span and append ` (N개월)` to the text of that `<hp:t>` element. The month span calculation:
     - Parse the two dates from the line (they may be in `YYYY-MM` or `YYYY.MM` format with ` ~ ` separator).
     - Calculate months = (end_year - start_year) * 12 + (end_month - start_month) + 1 (inclusive) — BUT check the expected values: if `2026-08 ~ 2026-10` should yield `3개월`, then it's inclusive counting (10 - 8 + 1 = 3). Verify with the task instruction: 단계1 → 3개월, 단계2 → 3개월, 단계3 → 1개월.
   - **Then**, remove layout-cache elements (`<hp:linesegarray>...</hp:linesegarray>` or equivalent) from any paragraph (`<hp:p>`) that was modified. Use regex or XML parsing. Be careful with namespaces.
   - Verify no `{{` remains in the output.

c) Use `lxml` or `xml.etree.ElementTree` for XML parsing if available. If using string manipulation, be extremely careful with namespace prefixes.

d) Write the modified XML back.

## 6. Repackage the HWPX

Repackage the extracted directory back into a ZIP with `.hwpx` extension:

```bash
cd /tmp/hwpx_work/template_extracted
zip -r /root/project_proposal_ready.hwpx . -x '*.DS_Store'
```

IMPORTANT: HWPX files must preserve the original ZIP structure. Use `stored` compression for `mimetype` if it exists (like ODF), or just use default compression for all files. Check if there's a `mimetype` file first.

## 7. Validate

```bash
# Verify it's a valid zip
unzip -t /root/project_proposal_ready.hwpx

# Check no placeholders remain
unzip -p /root/project_proposal_ready.hwpx | grep -c '{{'

# Check phase annotations exist
unzip -p /root/project_proposal_ready.hwpx | grep '개월'

# Check budget is normalized (no commas)
unzip -p /root/project_proposal_ready.hwpx | grep -oP '₩[0-9,]+'
```

The budget grep should show no commas. The phase grep should show 3 lines with `(3개월)`, `(3개월)`, `(1개월)` respectively.

## 8. Run the test

```bash
cd /root && python -m pytest test_output.py -v 2>&1 | head -80
```

If tests fail, read the assertion error carefully, inspect the actual XML content around the failing string, and fix. Pay special attention to:
- Whether the `(N개월)` is appended to the same `<hp:t>` element as the phase text and date range
- Whether date separators match (dash vs dot)
- Whether budget normalization is correct
- Whether all placeholders were replaced (including ones possibly split across tags)
- Whether layout cache elements were properly removed from modified paragraphs

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