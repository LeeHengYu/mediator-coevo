# Task Instruction

Complete the following task to update a renewal playbook HWPX document.

## Goal
Revise `renewal_playbook.hwpx` using data from `renewal_update.json` and `followups.csv`, saving the result to `/root/renewal_playbook_updated.hwpx`.

## Steps

### 1. Inspect input files
- `cat renewal_update.json` to see the new field values (customer name, current owner, renewal window, pricing band, escalation contact, pricing note).
- `cat followups.csv` to see the follow-up items and their `sequence` column.
- Note the exact old values and new values for each field.

### 2. Inspect the HWPX package
- Create a working directory: `mkdir -p /tmp/hwpx_work && cp renewal_playbook.hwpx /tmp/hwpx_work/ && cd /tmp/hwpx_work`
- Unzip: `unzip -o renewal_playbook.hwpx -d hwpx_contents`
- List contents: `find hwpx_contents -type f`
- Read the main section XML: `cat hwpx_contents/Contents/section0.xml` (and any other section files if they exist).
- Identify where the old field values appear in the XML text runs (`<hp:t>` elements).
- Identify the three existing follow-up lines.
- Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` exists and note its location.

### 3. Write a Python script to perform the edits
Create a Python script that:

a) Parses `renewal_update.json` to get the new values for: customer name, current owner, renewal window, pricing band, escalation contact, pricing note.

b) Parses `followups.csv` and sorts rows by the `sequence` column.

c) Reads the section XML file(s) as text (to preserve exact XML structure).

d) For each field in the JSON update:
   - Determine the old value by inspecting the original XML content.
   - Replace ALL occurrences of the old value with the new value throughout the editable sections.
   - Do NOT touch the appendix sentence.

e) For the follow-up lines:
   - Identify the three existing follow-up paragraph elements in the XML.
   - Replace their text content with the CSV items in sequence order.
   - If the CSV has a different number of items than 3, add or remove paragraph elements accordingly (but likely it's exactly 3).

f) For any paragraph whose `<hp:t>` text was modified, remove the `<hp:linesegarray>` element (and its children) from that paragraph. This is critical to prevent layout corruption. Use an XML parser (lxml or ElementTree) for this step to be precise.

g) Write the modified XML back.

h) Repackage the HWPX:
   - `cd hwpx_contents && zip -r /root/renewal_playbook_updated.hwpx . -x '*.DS_Store'`
   - Use `stored` compression for `mimetype` if it exists (like ODF packages), but HWPX typically doesn't require this. Just use normal zip.

### 4. Execute the script
Run the Python script and check for errors.

### 5. Validate the output
- Unzip `/root/renewal_playbook_updated.hwpx` to a temp directory and inspect the section XML.
- Confirm all old field values are replaced with new ones.
- Confirm follow-up lines match CSV items in sequence order.
- Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is unchanged.
- Confirm no `<hp:linesegarray>` elements remain in modified paragraphs.
- Confirm the file is a valid zip archive: `python3 -c "import zipfile; z=zipfile.ZipFile('/root/renewal_playbook_updated.hwpx'); z.testzip(); print('Valid zip')"`

### 6. Run the verifier
- Check if there's a test file: `ls tests/` or `ls *.py`
- If `test_output.py` or similar exists, run: `cd /root && python -m pytest test_output.py -v` (or the appropriate test path).

## Critical Reminders
- **Remove `<hp:linesegarray>`** from every paragraph you modify. This is the #1 cause of failures in HWPX tasks.
- **Do not modify the appendix sentence** `이 부록 문단은 그대로 유지해야 합니다.`.
- **Replace, don't duplicate** — remove old values entirely.
- **Preserve XML structure** — only change text content and remove layout cache elements.
- **Use proper namespace handling** when parsing XML — HWPX uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph` for `hp:`.
- The output path must be exactly `/root/renewal_playbook_updated.hwpx`.

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