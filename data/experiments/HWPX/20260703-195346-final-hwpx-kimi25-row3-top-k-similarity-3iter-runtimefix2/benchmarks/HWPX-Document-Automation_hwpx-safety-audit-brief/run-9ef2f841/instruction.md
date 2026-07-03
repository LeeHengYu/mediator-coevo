# Task Instruction

Complete the warehouse safety audit brief HWPX document automation task. Follow these steps precisely:

## Step 1: Reconnaissance
1. List all files in the task directory: `ls -la /root/` and find the task workspace.
2. Find and read the test file (`test_output.py` or similar) to understand EXACT verification expectations — what values are checked, what format, what assertions.
3. Read `audit_overview.json` and `corrective_actions.json` to get all data values.
4. Examine the HWPX template `safety_audit_template.hwpx` — it's a ZIP file. List its contents with `unzip -l` and then extract it to a temp directory.
5. Read all section XML files (e.g., `Contents/section0.xml`, `Contents/section1.xml`, etc.) to find all `{{...}}` placeholders and understand the document structure.

## Step 2: Understand the Contract
From the test file, identify:
- Which specific string values must appear in the output XML sections
- Whether dates should be in `YYYY.MM.DD` format (dot-separated)
- The exact format for severity notes — based on prior success, the format is `RiskTier (Korean note)` e.g., `High (즉시조치)`
- Whether placeholders `{{...}}` must be completely absent
- Any structural XML checks

## Step 3: Build the Solution
Write a Python script that:
1. Copies the template HWPX to `/root/safety_audit_brief_final.hwpx`
2. Opens it as a ZIP archive
3. For each section XML file:
   a. Reads the XML content
   b. Replaces ALL `{{...}}` placeholders with corresponding values from the JSON files
   c. For the risk tier field: insert the value AND append the severity note in parentheses, e.g., if risk_tier is 'High', the replacement becomes 'High (즉시조치)'
   d. For ANY occurrence of the risk tier text already placed, ensure the severity note follows it
   e. Converts ALL dates from `YYYY-MM-DD` format to `YYYY.MM.DD` format (replace hyphens with dots in date patterns)
   f. Fills corrective action lines in the SAME ORDER as they appear in `corrective_actions.json`
   g. Removes ALL `<hp:linesegarray>...</hp:linesegarray>` elements (including multiline) from any paragraph whose text content was modified — this prevents layout cache staleness
   h. Ensures NO `{{` or `}}` placeholder markers remain
4. Writes the modified XML back into the ZIP
5. Saves the result to `/root/safety_audit_brief_final.hwpx`

## Step 4: Key Details
- The severity mapping is: `High -> 즉시조치`, `Medium -> 계획보완`, `Low -> 모니터링`
- The severity note format (based on prior successful run) is: `RiskTier (SeverityNote)` — with a space before the opening parenthesis
- Keep ALL existing section titles and row labels unchanged
- The HWPX must remain a valid ZIP package with correct structure
- Use `re.sub` for robust regex-based replacements
- For linesegarray removal, use regex: `re.sub(r'<hp:linesegarray>.*?</hp:linesegarray>', '', content, flags=re.DOTALL)`

## Step 5: Validate
1. After generating the output, verify it's a valid ZIP: `unzip -l /root/safety_audit_brief_final.hwpx`
2. Extract and inspect the section XMLs to confirm no `{{` placeholders remain
3. Confirm dates are in `YYYY.MM.DD` format
4. Confirm severity notes are present
5. Run the test suite: `cd /root && python -m pytest test_output.py -v` (or wherever the test file is located)
6. If tests fail, read the error output carefully, identify the exact mismatch, fix, and re-run.

## Critical Reminders
- READ THE TEST FILE FIRST before writing any replacement logic. The test defines the contract.
- Do not guess placeholder names — read them from the actual XML.
- Do not guess JSON field names — read them from the actual JSON files.
- Strip linesegarray from modified paragraphs to prevent rendering issues.
- Ensure the final file is at exactly `/root/safety_audit_brief_final.hwpx`.

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