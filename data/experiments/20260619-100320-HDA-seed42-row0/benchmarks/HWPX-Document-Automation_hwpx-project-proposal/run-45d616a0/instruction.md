# Task Instruction

Complete the project proposal document by filling in placeholders and making required modifications. Follow these steps precisely:

## Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). You will need to unzip it, modify the XML content, and rezip it.

## Step 2: Inspect the input files
1. `cat /root/project_proposal.json` — read all the JSON values you'll need to substitute.
2. `cd /root && cp project_proposal_template.hwpx template_backup.hwpx`
3. `mkdir -p /root/hwpx_work && cd /root/hwpx_work && unzip -o /root/project_proposal_template.hwpx` — extract the HWPX archive.
4. `find /root/hwpx_work -type f -name '*.xml' -o -name '*.rels'` — list all files in the archive.
5. Read every XML file found (especially files under `Contents/` such as `section0.xml`, `section1.xml`, etc.) to find all `{{...}}` placeholders. Use `grep -r '{{' /root/hwpx_work/` to locate them all.

## Step 3: Understand the placeholder mapping
From the JSON file, map each `{{placeholder_name}}` to its corresponding JSON value. Read the JSON carefully to get exact field names and values.

## Step 4: Perform substitutions in the XML files
For each XML file containing placeholders:
1. Read the file content.
2. Replace every `{{...}}` placeholder with the matching JSON value.
3. **Budget normalization**: For any budget/금액 value, remove commas from the number but keep the leading currency symbol (e.g., `₩1,500,000,000` becomes `₩1500000000`, or `$1,500,000` becomes `$1500000`). Check the JSON to see the exact format.
4. **Month span appending**: Find lines containing `단계1`, `단계2`, `단계3` phase descriptions. Each phase line should already contain a date range. Calculate the month span from that date range and append it in parentheses:
   - `단계1` → append `(3개월)`
   - `단계2` → append `(3개월)`  
   - `단계3` → append `(1개월)`
   These values are specified in the requirements. Append the parenthesized month span after the phase text content in the appropriate XML text element. Make sure the appended text is within the same `<hp:t>` or equivalent text run element, or add a new text run in the same paragraph — whichever keeps the XML valid.
5. **Stale layout-cache removal**: For any paragraph (`<hp:p>`) whose text content you modify, remove any `<hp:linesegarray>` or `<hp:lineSegArray>` elements (these are layout cache elements that cause overlapping characters when stale). Search for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:LineSeg>`, or similar layout caching elements within modified paragraphs and delete them.
6. Keep all Korean labels and static note lines unchanged (only modify placeholder text and phase lines as specified).

## Step 5: Verify no placeholders remain
Run `grep -r '{{' /root/hwpx_work/` to confirm zero matches. If any remain, fix them.

## Step 6: Repackage the HWPX file
The HWPX/ZIP must be packaged correctly:
```
cd /root/hwpx_work
zip -r -0 /root/project_proposal_ready.hwpx mimetype
zip -r /root/project_proposal_ready.hwpx . -x mimetype -x ./mimetype
```
Note: `mimetype` must be the first entry and stored without compression (`-0`). If there is no `mimetype` file in the archive, just do a normal `zip -r /root/project_proposal_ready.hwpx .` from within the work directory.

## Step 7: Validate the output
1. `file /root/project_proposal_ready.hwpx` — should show ZIP archive.
2. `unzip -l /root/project_proposal_ready.hwpx` — list contents to verify structure matches the original template.
3. Create a temp directory, unzip the result there, and `grep -r '{{' /tmp/verify/` to confirm no placeholders remain.
4. Verify the budget value has no commas but retains the currency symbol.
5. Verify phase lines contain the appended month spans `(3개월)`, `(3개월)`, `(1개월)`.
6. Verify that modified paragraphs do not contain `linesegarray` or `lineSegArray` elements.

## Important Notes
- Be very careful with XML encoding — preserve all existing XML structure, namespaces, and attributes.
- Only modify text content and remove layout cache elements in modified paragraphs.
- Use Python if sed/awk becomes unwieldy for XML manipulation — `import xml.etree.ElementTree` or direct string operations on the XML files.
- The final file MUST be at `/root/project_proposal_ready.hwpx`.

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