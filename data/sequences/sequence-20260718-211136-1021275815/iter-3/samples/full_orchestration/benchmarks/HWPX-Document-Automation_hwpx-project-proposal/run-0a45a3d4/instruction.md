# Task Instruction

Complete the project proposal document by filling in placeholders and making required modifications. Follow these steps precisely:

## Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). You will need to:
- Unzip the template
- Edit the XML content files
- Repackage as a ZIP with `.hwpx` extension

## Step 2: Inspect the files
1. `cd /root && ls` to see available files.
2. Read `project_proposal.json` to understand all the values to substitute.
3. Unzip the template: `mkdir -p /tmp/hwpx_work && cp project_proposal_template.hwpx /tmp/hwpx_work/template.zip && cd /tmp/hwpx_work && unzip template.zip -d template_extracted`
4. Explore the extracted directory structure: `find template_extracted -type f | head -50`
5. Identify the content XML files (likely under `Contents/` directory, e.g., `section0.xml`, `section1.xml`, etc.) and read each one carefully.

## Step 3: Identify all placeholders
Search all XML files for `{{` patterns: `grep -rn '{{' template_extracted/`
Document every placeholder found and map each to the corresponding JSON key.

## Step 4: Understand the modifications needed
1. **Placeholder replacement**: Replace every `{{...}}` with the matching JSON value.
2. **Month span appending**: For each phase line (단계1, 단계2, 단계3), calculate the month span from the date range already present in that line and append it in parentheses, e.g., `(3개월)`. The date ranges are in the line text itself — parse the start and end dates to compute the number of months.
3. **Budget normalization**: Remove commas from the budget numeric value but keep the leading currency symbol (e.g., `₩1,000,000` becomes `₩1000000`, or if the symbol is `$` keep `$` etc.). Check the JSON to see the exact format.
4. **Layout cache cleanup**: After modifying paragraph text in the XML, remove any `<hp:linesegarray>` or `<hp:lineSegArray>` elements (and their children) from the same paragraph (`<hp:p>`) that was modified. These are layout cache elements that cause overlapping characters if stale. Search for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineSeg>`, or similar layout caching elements within modified paragraphs.

## Step 5: Edit the XML files
Use Python for reliable XML manipulation. Write a Python script that:
1. Reads `project_proposal.json`.
2. For each content XML file (section*.xml), parses it.
3. Finds all text nodes containing `{{...}}` patterns and replaces them with JSON values.
4. For budget values, strips commas from the numeric portion while preserving the currency symbol.
5. For phase lines (단계1, 단계2, 단계3), after placeholder substitution, appends the month span. To calculate months: parse the two dates (likely in YYYY.MM.DD or YYYY-MM-DD format) and compute the difference in months. Append ` (N개월)` to the text of that line/run.
6. Removes `linesegarray` / `lineSegArray` elements from any paragraph whose text content was modified.
7. Writes the modified XML back, preserving the XML declaration and encoding.

IMPORTANT: When editing XML, be careful with namespaces. Read the actual namespace prefixes used in the files. Common HWPX namespaces include `http://www.hancom.co.kr/hwpml/2011/paragraph` for `hp:` prefix. Use the actual namespaces found in the files.

## Step 6: Verify no placeholders remain
After editing, run: `grep -rn '{{' template_extracted/`
This must return NO results.

## Step 7: Repackage as HWPX
The HWPX must be a valid ZIP. Repackage from within the extracted directory:
```
cd /tmp/hwpx_work/template_extracted
zip -r /root/project_proposal_ready.hwpx . -x '*.DS_Store'
```
Make sure the ZIP is created from inside the root of the extracted content (not wrapping an extra directory level).

## Step 8: Final validation
1. Verify the output exists: `ls -la /root/project_proposal_ready.hwpx`
2. Verify it's a valid ZIP: `unzip -t /root/project_proposal_ready.hwpx`
3. Verify no placeholders remain: `unzip -p /root/project_proposal_ready.hwpx | grep -c '{{'` should be 0.
4. Verify the budget value has no commas but retains the currency symbol.
5. Verify each phase line has the appended month span like `(3개월)`.
6. Verify Korean labels and static note lines are unchanged.
7. Verify no `linesegarray`/`lineSegArray` elements exist in modified paragraphs.

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