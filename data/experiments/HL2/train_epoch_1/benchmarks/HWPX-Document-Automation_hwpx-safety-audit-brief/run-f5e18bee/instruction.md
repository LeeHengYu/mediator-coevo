# Task Instruction

You must produce `/root/safety_audit_brief_final.hwpx` by filling in the template HWPX package with data from the two JSON files. Follow every step below precisely.

## Step 0 – Inspect all inputs

```bash
cd /root
ls -la
find . -name '*.json' -o -name '*.hwpx' | head -20
```

Identify the exact paths of `safety_audit_template.hwpx`, `audit_overview.json`, and `corrective_actions.json`.

## Step 1 – Understand the HWPX structure

An `.hwpx` file is a ZIP archive. Unzip the template into a working directory:

```bash
mkdir -p /root/hwpx_work
cp safety_audit_template.hwpx /root/hwpx_work/template.zip
cd /root/hwpx_work
unzip -o template.zip -d template_contents
find template_contents -type f | sort
```

Read every XML file inside (especially files like `section0.xml`, `section1.xml`, or `content.hpf`, `content.xml`, etc.). Print their full contents so you can see every `{{...}}` placeholder and every text node.

## Step 2 – Read the JSON data files

```bash
cat /root/audit_overview.json
cat /root/corrective_actions.json
```

Note every field name and value. Pay special attention to:
- The risk tier value (e.g. `High`, `Medium`, `Low`)
- The inspection date (in `YYYY-MM-DD` format)
- The corrective action items and their order

## Step 3 – Read the verifier test

```bash
find / -path '*/tests/test_output*' -o -path '*/tests/test_*.py' 2>/dev/null | head -10
cat /root/tests/test_outputs.py 2>/dev/null || find / -name 'test_output*' -exec cat {} \;
```

Read the test file completely. Understand exactly what strings, formats, and conditions the verifier checks. This is the ground truth contract.

## Step 4 – Write a Python script to perform the substitution

Create `/root/fill_template.py`. The script must:

1. Copy the template HWPX to a working directory and unzip it.
2. Load both JSON files.
3. For each XML file in the package that contains text content (especially section XML files):
   a. Parse it (use `lxml.etree` if available, otherwise `xml.etree.ElementTree`).
   b. Walk every text node (both `.text` and `.tail` of every element).
   c. Replace every `{{placeholder}}` with the corresponding value from the JSON data.
   d. **Date rewriting**: Every occurrence of the inspection date in `YYYY-MM-DD` format must be rewritten to `YYYY.MM.DD` (replace hyphens with dots).
   e. **Risk tier + severity note**: After substituting the risk tier value, append a severity note **in parentheses**. The mapping is:
      - `High` → `High (즉시조치)`
      - `Medium` → `Medium (계획보완)`
      - `Low` → `Low (모니터링)`
      
      CRITICAL: The format MUST be `Value (Note)` with a space before the opening parenthesis and the note inside parentheses. For example: `High (즉시조치)`. Do NOT omit the parentheses.
      
      Apply this to EVERY occurrence of the risk tier string in every text node, not just placeholders.
   f. **Corrective actions**: Fill the three corrective-action lines in the exact order they appear in `corrective_actions.json`.
   g. **Stale layout cache removal**: After modifying any paragraph's text, remove any child elements that look like layout caches. Specifically, remove elements whose tag (local name) contains `lineseg`, `LineSeg`, `LINESEG`, `lineBreak`, `LineBreak`, or similar layout-cache tags. Check the actual XML namespace and tag names present in the template to identify these. If the template uses `hp:linesegarray` or similar, remove those elements from any paragraph you modified.
   h. Verify no `{{` or `}}` remain in any text node.
4. Serialize each modified XML back to the file (preserve XML declaration and encoding if present).
5. Re-zip the entire directory structure back into a valid `.hwpx` (ZIP) file at `/root/safety_audit_brief_final.hwpx`, preserving the original directory structure exactly.

## Step 5 – Run the script

```bash
cd /root
python3 fill_template.py
```

If errors occur, debug and fix them.

## Step 6 – Validate the output

1. Verify the output is a valid ZIP:
```bash
unzip -t /root/safety_audit_brief_final.hwpx
```

2. Extract and inspect the XML sections to confirm:
```bash
mkdir -p /root/verify_output
cd /root/verify_output
unzip -o /root/safety_audit_brief_final.hwpx
find . -name '*.xml' -exec grep -l 'section' {} \;
```

3. Check critical strings are present (adjust filenames based on what you found):
```bash
# Check risk tier with parenthesized severity note
grep -r '(즉시조치)\|(계획보완)\|(모니터링)' . || echo 'SEVERITY NOTE MISSING'
# Check date format is YYYY.MM.DD not YYYY-MM-DD
grep -rn '[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}' . --include='*.xml' && echo 'OLD DATE FORMAT FOUND' || echo 'Date format OK'
# Check no placeholders remain
grep -rn '{{' . --include='*.xml' && echo 'PLACEHOLDERS REMAIN' || echo 'No placeholders'
```

4. Run the verifier test:
```bash
cd /root && python3 -m pytest tests/ -v 2>&1 | head -80
```

If the test fails, read the exact assertion error, fix the script, regenerate the output, and re-run the test. Iterate until all tests pass.

## Key Reminders
- The severity note format is `Value (Note)` WITH PARENTHESES. This was the exact failure in the previous run.
- Every `{{...}}` must be replaced; none may remain.
- The inspection date must use dots not hyphens everywhere.
- Corrective actions must be in the same order as the JSON array.
- Remove stale layout-cache elements from modified paragraphs.
- The final file must be a valid ZIP/HWPX package with the correct internal structure.

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