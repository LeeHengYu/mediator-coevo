# Task Instruction

Complete the project proposal document by filling in placeholders and making required modifications. Follow these steps precisely:

## Step 1: Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). You will need to unzip it, modify the XML content, and repackage it.

## Step 2: Extract and inspect
```bash
mkdir -p /root/hwpx_work
cp /root/project_proposal_template.hwpx /root/hwpx_work/template.zip
cd /root/hwpx_work
unzip template.zip -d template_extracted
```

Inspect the directory structure:
```bash
find template_extracted -type f
```

Read the JSON data file:
```bash
cat /root/project_proposal.json
```

## Step 3: Identify all XML files containing `{{` placeholders
```bash
grep -rl '{{' template_extracted/
```

For each file found, display its full contents so you can see every placeholder and the surrounding XML structure.

## Step 4: Understand the placeholder-to-JSON mapping
Read the JSON file carefully. Each `{{key}}` placeholder in the XML files should be replaced with the corresponding value from the JSON. Map every placeholder to its JSON value.

## Step 5: Apply replacements with these specific rules

### 5a: Budget normalization
For the budget value, remove commas from the number but keep the leading currency symbol (e.g., `₩1,500,000,000` becomes `₩1500000000`, or `$1,234,567` becomes `$1234567`). Apply this normalized value when replacing the budget placeholder.

### 5b: Month span annotations for phase lines
After replacing placeholders, find each phase line (단계1, 단계2, 단계3). Each phase line contains a date range. Calculate the month span from the date range:
- Count the number of months between the start and end dates (inclusive of both endpoint months, i.e., the difference in months + 1, or as specified by the dates).
- For example, if a phase runs from 2025.01 to 2025.03, that is 3 months.
- Append ` (N개월)` after the phase text content in the same text run or as a new text run in the same paragraph. The format is parenthesized: `(3개월)`.

To calculate months: parse the start year.month and end year.month. Months = (end_year - start_year) * 12 + (end_month - start_month) + 1.

### 5c: Preserve Korean labels and static note lines
Do not modify any Korean label text or static note lines. Only modify placeholder text and add month span annotations.

### 5d: Remove stale layout-cache elements
In HWPX XML, paragraphs may contain layout-cache elements (look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, or similar caching/layout elements within paragraph tags). For any paragraph whose text content you modify, remove these layout-cache child elements entirely. This prevents overlapping characters when the document is opened. Search for elements that look like layout caches — they typically have names containing `lineseg`, `LineSeg`, `lineSegArray`, or similar.

## Step 6: Verify no placeholders remain
```bash
grep -r '{{' template_extracted/
```
This must return nothing. If any `{{...}}` patterns remain, fix them.

## Step 7: Verify month spans were added
Grep for `개월` in the extracted files to confirm the month span annotations are present for all three phases.

## Step 8: Repackage as HWPX
The HWPX file must be a valid ZIP with the same structure. Repackage:
```bash
cd /root/hwpx_work/template_extracted
zip -r /root/project_proposal_ready.hwpx . -x '*.DS_Store'
```

## Step 9: Final validation
```bash
# Verify it's a valid zip
unzip -t /root/project_proposal_ready.hwpx

# Verify no placeholders remain
unzip -p /root/project_proposal_ready.hwpx | grep -c '{{'
# Should output 0

# Verify month spans exist
unzip -p /root/project_proposal_ready.hwpx | grep '개월'
# Should show 3 lines with (3개월), (3개월), (1개월)

# Verify budget has no commas (check the normalized budget value appears without commas)
unzip -p /root/project_proposal_ready.hwpx | grep -o '₩[0-9,]*' || true
# The currency value should have no commas
```

Ensure the output file exists at exactly `/root/project_proposal_ready.hwpx`.

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