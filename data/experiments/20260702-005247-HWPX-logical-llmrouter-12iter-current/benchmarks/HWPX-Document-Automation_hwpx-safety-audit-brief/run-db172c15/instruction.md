# Task Instruction

## Task: Prepare warehouse safety audit brief

You must fill a HWPX template with data from two JSON files and save the result.

### Step 0 — Orient
```bash
cd /root
find . -maxdepth 3 -type f | head -80
```
Locate:
- `safety_audit_template.hwpx`
- `audit_overview.json`
- `corrective_actions.json`
- Any test file (likely `test_output.py` or similar)

Read the test file completely to understand **exactly** what the verifier checks (field names, formatting, string patterns, XML structure expectations). This is critical — the test is the contract.

### Step 1 — Understand the HWPX structure
A `.hwpx` file is a ZIP archive (like DOCX). Unzip the template to a working directory:
```bash
mkdir -p /root/hwpx_work
cp safety_audit_template.hwpx /root/hwpx_work/template.hwpx
cd /root/hwpx_work
unzip -o template.hwpx -d template_contents
find template_contents -type f
```
Identify the XML file(s) containing the document body (likely `Contents/section0.xml` or similar). Read each XML file that contains `<hp:t>` tags or `{{` placeholders.

### Step 2 — Read the JSON data files
```bash
cat /root/audit_overview.json
cat /root/corrective_actions.json
```
Note every field name and value. Pay special attention to:
- The risk tier value (e.g., "High", "Medium", "Low")
- The inspection date format (YYYY-MM-DD) — you must convert to YYYY.MM.DD

### Step 3 — Read the test file thoroughly
```bash
cat /root/test_output.py  # or wherever it is
```
Extract the **exact** string patterns the test checks for. Previous failure showed the test expects:
- `'High (즉시조치)'` — note the space before `(` and the parentheses
- The severity mapping: High -> 즉시조치, Medium -> 계획보완, Low -> 모니터링
- Date in YYYY.MM.DD format everywhere
- No remaining `{{...}}` placeholders

### Step 4 — Write a Python script to perform all replacements
Write `/root/build_hwpx.py` that:

1. Copies the template HWPX to a working location and unzips it.
2. Reads `audit_overview.json` and `corrective_actions.json`.
3. For each XML file in the package that contains text content:
   a. Parses it with `lxml.etree` (preferred) or `xml.etree.ElementTree`.
   b. Finds all `<hp:t>` (or equivalent text) elements.
   c. For **each** text element, performs placeholder replacement:
      - Replace `{{placeholder}}` patterns with the corresponding JSON values.
      - For the risk tier placeholder: replace with `VALUE (SEVERITY_NOTE)` using the mapping. E.g., if risk is "High", the replacement is `High (즉시조치)`.
      - For date values: convert from `YYYY-MM-DD` to `YYYY.MM.DD` everywhere (both in placeholder replacements AND in any already-placed date text).
   d. For corrective actions: fill the three corrective-action lines **in the same order** they appear in `corrective_actions.json`.
   e. After replacing text in any `<hp:t>` element, remove any sibling or child layout-cache elements (look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:cache>`, or `<hp:parameterset>` with layout-related names) from the **parent paragraph** (`<hp:p>`) to prevent stale layout cache.
   f. Verify no `{{` remains in any text element. If any remain, print a warning with the element text.
4. Writes the modified XML back (preserving XML declaration and encoding).
5. Re-zips the directory into `/root/safety_audit_brief_final.hwpx` using the same archive structure.

**Critical details for the script:**
- Handle the case where a placeholder might be split across multiple `<hp:t>` tags within the same `<hp:run>` or `<hp:p>`. Strategy: concatenate all `<hp:t>` texts in a paragraph, check for placeholders, and if found, consolidate into a single `<hp:t>` (removing extras) then do replacement.
- Use namespace-aware XML parsing. Extract namespaces from the root element.
- When re-zipping, use `zipfile.ZipFile` with `ZIP_DEFLATED` and preserve the directory structure exactly.

### Step 5 — Execute and verify
```bash
python3 /root/build_hwpx.py
```
Then verify:
```bash
# Check it's a valid zip
python3 -c "import zipfile; z=zipfile.ZipFile('/root/safety_audit_brief_final.hwpx'); print(z.namelist()); z.close()"

# Check the XML content for expected strings
cd /root/hwpx_verify && mkdir -p hwpx_verify && cd hwpx_verify
unzip -o /root/safety_audit_brief_final.hwpx -d verify_contents
grep -r '{{' verify_contents/ || echo 'No placeholders remain - GOOD'
grep -r '즉시조치\|계획보완\|모니터링' verify_contents/
grep -rn 'High\|Medium\|Low' verify_contents/ | head -20
```

### Step 6 — Run the test suite
```bash
cd /root && python3 -m pytest test_output.py -v 2>&1 | tail -60
```
If any test fails, read the assertion error carefully, inspect the relevant XML section, fix the script, and re-run. Common issues:
- Namespace prefixes differ from expected
- Text split across tags not handled
- Layout cache not fully removed
- Date format not converted in all occurrences
- Severity note format mismatch (must be exactly `VALUE (NOTE)` with space and parentheses)

Iterate until all tests pass.

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