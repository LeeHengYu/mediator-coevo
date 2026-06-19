# Task Instruction

Complete the project proposal document by filling in placeholders and making required modifications. Follow these steps precisely:

## Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). You'll need to unzip it, modify the XML content, and rezip it.

## Step 2: Inspect the input files
1. `cat /root/project_proposal.json` — read all the JSON values you'll need to substitute.
2. `cd /root && mkdir -p hwpx_work && cp project_proposal_template.hwpx hwpx_work/ && cd hwpx_work`
3. `unzip project_proposal_template.hwpx -d template_extracted` — extract the HWPX package.
4. `find template_extracted -type f -name '*.xml' -o -name '*.rels'` — list all files in the package.
5. For each XML file found, `cat` it and look for `{{...}}` placeholder patterns. Also look for any content files (typically under `Contents/` directory, e.g., `section0.xml` or similar).
6. Carefully note the full structure of every XML file — you will need to preserve this structure exactly.

## Step 3: Understand the data and placeholders
From the JSON file, identify every key-value pair. Search all XML files for every `{{...}}` pattern using `grep -r '{{' template_extracted/`. Map each placeholder to its corresponding JSON key.

## Step 4: Plan the modifications
For each XML file containing placeholders:
- Replace every `{{...}}` placeholder with the matching JSON value.
- **Budget normalization**: If a budget value contains commas (e.g., `₩1,000,000,000`), remove the commas but keep the currency symbol (e.g., `₩1000000000`).
- **Phase month spans**: For lines containing phase information (단계1, 단계2, 단계3), calculate the month span from the date range present in that line and append it in parentheses. The expected results are: `단계1` line gets `(3개월)`, `단계2` line gets `(3개월)`, `단계3` line gets `(1개월)`. Append the parenthesized month span after the existing text content of that phase's paragraph/text run.
- **Preserve all Korean labels and static note lines unchanged.**
- **Remove stale layout-cache elements**: In any paragraph (`<hp:p>`) whose text content you modify, remove any `<hp:linesegarray>` or `<hp:lineSegArray>` or similar layout-cache child elements (these cache glyph positions and will cause overlapping characters if stale). Look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineSeg>`, or similar layout/cache elements within modified paragraphs. Remove them entirely from modified paragraphs only.

## Step 5: Perform the edits
Use Python for reliable XML manipulation. Write a Python script that:
1. Parses each XML file containing placeholders using `xml.etree.ElementTree` (be careful with namespaces — register them first so they're preserved in output).
2. Alternatively, if namespace handling is complex, use careful string replacement with `re` module on the raw XML text — but be very precise.
3. Actually, given HWPX XML complexity, prefer a hybrid approach: read the XML as text, do the replacements with regex/string operations, then write back. This avoids namespace mangling.
4. For each file:
   a. Read the file content as UTF-8 text.
   b. Replace all `{{placeholder}}` patterns with corresponding JSON values (with budget comma removal).
   c. For phase lines (단계1, 단계2, 단계3), find the text runs containing them and append the month span.
   d. For any `<hp:p>` element that was modified, remove `<hp:linesegarray>...</hp:linesegarray>` (case-insensitive tag matching within the HWPX namespace) blocks.
   e. Write the modified content back.

## Step 6: Verify no placeholders remain
Run `grep -r '{{' template_extracted/` — this must return no results.

## Step 7: Repackage the HWPX file
The HWPX/ZIP must be repacked correctly:
```bash
cd template_extracted
zip -r /root/project_proposal_ready.hwpx . -x '.*'
cd /root
```
Use `mimetype` file first if it exists (store it uncompressed as first entry, like ODF/OOXML conventions): check if there's a `mimetype` file; if so, add it first with `zip -0`.

## Step 8: Validate the output
1. `file /root/project_proposal_ready.hwpx` — should show ZIP archive.
2. `unzip -l /root/project_proposal_ready.hwpx` — should list all expected files.
3. Extract to a temp dir and `grep -r '{{' temp_dir/` — must find nothing.
4. Verify the phase lines contain the month spans: grep for `3개월` and `1개월`.
5. Verify the budget value has no commas but retains currency symbol.
6. Verify Korean labels and note lines are intact.

## Critical details
- Do NOT change any XML structure, attributes, or elements beyond what's needed for the replacements.
- The `{{...}}` placeholders may appear inside `<hp:t>` text elements or similar — replace the text content only.
- When appending month spans to phase lines, add them as part of the same text content (e.g., if `<hp:t>` contains `단계1: 2025.01 ~ 2025.03`, change it to `단계1: 2025.01 ~ 2025.03 (3개월)`).
- Month calculation: count inclusive months from start to end date. For example, 2025.01 ~ 2025.03 = 3 months, 2025.04 ~ 2025.06 = 3 months, 2025.07 ~ 2025.07 = 1 month.
- Ensure UTF-8 encoding is preserved throughout.
- The output path must be exactly `/root/project_proposal_ready.hwpx`.

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