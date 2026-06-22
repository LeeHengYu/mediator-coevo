# Task Instruction

## Task: Update renewal_playbook.hwpx

You need to revise the existing `renewal_playbook.hwpx` using data from `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

### Step 1: Explore the workspace and understand inputs

1. List all files in the task directory to find `renewal_playbook.hwpx`, `renewal_update.json`, `followups.csv`, and any test files.
2. Read `renewal_update.json` completely — note every field (customer name, current owner, renewal window, pricing band, escalation contact, pricing note, etc.).
3. Read `followups.csv` completely — note the columns and the `sequence` column that determines ordering.
4. Read any test/verifier files (e.g., `test_output.py`, `test_outputs.py`, or similar) to understand exactly what the verifier checks. This is critical — understand the exact assertions, expected strings, and structural checks.

### Step 2: Inspect the HWPX package structure

1. A `.hwpx` file is a ZIP archive. Copy `renewal_playbook.hwpx` to a working location and unzip it.
2. List all files in the extracted archive.
3. Read every XML file in the `Contents/` directory (especially `section0.xml`, `content.hpf`, `header.xml`, etc.) to understand the document structure.
4. Identify where the following appear in the XML:
   - Customer name
   - Current owner
   - Renewal window
   - Pricing band
   - Escalation contact
   - Pricing note
   - The three existing follow-up lines
   - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`

### Step 3: Plan the edits carefully

Before making any edits:
1. Map each field from `renewal_update.json` to its old value found in the XML.
2. Identify ALL occurrences of each old value across ALL XML files in the package.
3. For follow-up lines: identify the exact XML elements containing the three old follow-up items. These will be replaced (not appended) with the CSV items sorted by `sequence` column.
4. Confirm the appendix sentence location — it must NOT be modified.

### Step 4: Perform the edits

Using Python with `xml.etree.ElementTree` or `lxml` (or careful string operations if XML parsing is problematic with namespaces):

1. For each field in `renewal_update.json`, replace ALL occurrences of the old value with the new value in every XML file where it appears.
2. For follow-up lines:
   - Sort the CSV rows by the `sequence` column (ascending).
   - Replace the three existing follow-up paragraph elements with the new follow-up items in sequence order.
   - Make sure old follow-up text is completely removed (no duplicates).
3. **Critical: Layout cache cleanup** — For any paragraph (`<hp:p>`) whose text content you modified, remove any `<hp:linesegarray>` or `<hp:lineSegArray>` child elements (these are layout cache elements that cause overlapping characters if stale). Check the verifier to see if it checks for this specifically.
4. Do NOT modify the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.`.

### Step 5: Repackage the HWPX

1. Repackage the modified files back into a ZIP archive saved as `/root/renewal_playbook_updated.hwpx`.
2. Use `zipfile.ZIP_DEFLATED` compression.
3. Ensure the directory structure inside the ZIP matches the original exactly (same paths, same files).
4. Make sure `[Content_Types].xml` and `_rels/.rels` are preserved if they exist.

### Step 6: Validate

1. Verify `/root/renewal_playbook_updated.hwpx` exists and is a valid ZIP.
2. Extract it and verify:
   - All new field values from `renewal_update.json` appear in the XML.
   - No old field values remain in editable sections.
   - Follow-up lines match CSV items in sequence order.
   - Appendix sentence is unchanged.
   - No `<hp:linesegarray>` or similar layout cache elements remain in modified paragraphs.
3. Run the verifier test: `cd /root && python -m pytest tests/ -xvs` (or whatever test path the verifier uses). Read the output carefully.
4. If tests fail, read the error messages, identify what's wrong, fix it, and re-run.

### Important Notes
- From past failures (safety-audit-brief): ensure that specific expected strings appear exactly as the verifier expects them in the XML content. Check the test file to see exact expected strings.
- From past successes: the HWPX automation pattern of unzip → edit XML → rezip has worked reliably.
- Be meticulous about namespace handling in XML — HWPX files use namespaces like `urn:hancom:hwpml:...`.
- If the verifier checks for absence of layout cache elements, ensure thorough removal from ALL modified paragraphs, not just some.

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