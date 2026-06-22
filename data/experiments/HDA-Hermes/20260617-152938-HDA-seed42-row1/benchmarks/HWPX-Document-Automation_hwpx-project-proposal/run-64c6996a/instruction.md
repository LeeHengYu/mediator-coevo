# Task Instruction

Complete the project proposal document by filling in placeholders and making required modifications. Follow these steps precisely:

## Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). You will need to:
- Unzip the template
- Edit the XML content files
- Repackage as a valid ZIP/HWPX

## Step 2: Inspect the files
1. `cat /root/project_proposal.json` to see all the replacement values.
2. `cd /root && mkdir -p hwpx_work && cp project_proposal_template.hwpx hwpx_work/ && cd hwpx_work`
3. `unzip -o project_proposal_template.hwpx -d template_extracted`
4. `find template_extracted -type f` to see the file structure.
5. Read every XML file in the extracted directory, especially files under `Contents/` (like `section0.xml`, `content.hpf`, etc.). Use `cat` on each XML file to understand the structure. Look for `{{...}}` placeholders.

## Step 3: Read and understand the JSON data
Parse the JSON file to identify all key-value pairs. Note the budget value — it likely has commas (e.g., `₩50,000,000`). You must remove commas but keep the currency symbol (e.g., `₩50000000`).

## Step 4: Perform replacements in the XML content
For each XML file containing `{{...}}` placeholders:
1. Replace every `{{placeholder}}` with the corresponding value from the JSON.
2. For the budget field, normalize by removing commas from the numeric part while preserving the leading currency symbol (e.g., `₩` or `$`).
3. After each phase line (단계1, 단계2, 단계3), append a parenthesized month span calculated from the date range already present in that line:
   - Look at the start and end dates in each phase line. Calculate the number of months between them.
   - For example, if 단계1 spans 3 months, append ` (3개월)` to that line's text.
   - The expected values based on the task description: 단계1 -> `(3개월)`, 단계2 -> `(3개월)`, 단계3 -> `(1개월)`.
   - Make sure the month span text is appended to the same text run or paragraph element as the phase text, not as a separate paragraph.

## Step 5: Remove stale layout-cache elements
For any paragraph (`<hp:p>` or similar) whose text content you modified:
- Look for layout-cache elements such as `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, `<hp:lineSeg>`, or similar cached layout data within or associated with modified paragraphs.
- Remove these stale layout-cache elements entirely from modified paragraphs so the document renders cleanly without overlapping characters.
- Do NOT remove layout-cache from paragraphs you did not modify.

## Step 6: Verify no placeholders remain
After all replacements, grep all XML files for `{{` to confirm zero remaining placeholders:
```
grep -r '{{' template_extracted/
```
If any remain, fix them.

## Step 7: Keep all Korean labels and static note lines unchanged
Do not modify any Korean label text (like field labels) or static note lines. Only modify placeholder values and phase lines as specified.

## Step 8: Repackage as HWPX
Repackage the modified files back into a valid HWPX (ZIP) archive:
```
cd template_extracted
zip -r -0 /root/project_proposal_ready.hwpx mimetype
zip -r /root/project_proposal_ready.hwpx * -x mimetype
cd /root
```
Note: The `mimetype` file (if present) should be stored first without compression (`-0` flag) for OPC/ZIP package compliance. If there's no `mimetype` file, just zip everything normally.

## Step 9: Final validation
1. Verify the output file exists: `ls -la /root/project_proposal_ready.hwpx`
2. Verify it's a valid ZIP: `unzip -t /root/project_proposal_ready.hwpx`
3. Extract to a temp dir and grep for any remaining `{{`: `mkdir -p /tmp/verify && unzip -o /root/project_proposal_ready.hwpx -d /tmp/verify && grep -r '{{' /tmp/verify/; echo 'Grep exit code:' $?`
4. Spot-check a few replaced values in the XML to confirm correctness.
5. Verify the budget value has no commas but retains the currency symbol.
6. Verify each 단계 line has the correct month span appended.
7. Verify that modified paragraphs do not contain layout-cache elements.

## Important Notes
- Be very careful with XML encoding. Preserve all XML structure, namespaces, and attributes.
- Use Python for complex string operations if sed/awk becomes unwieldy with XML content.
- When editing XML, prefer Python's `xml.etree.ElementTree` or direct string replacement with careful handling, but be aware that ElementTree may alter namespace prefixes. If using ElementTree, register namespaces first. If the XML is complex, consider using careful string replacement with Python's `re` module instead.
- The HWPX XML namespace is typically something like `http://www.hancom.co.kr/hwpml/...` — inspect the actual files to determine exact namespaces before parsing.

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