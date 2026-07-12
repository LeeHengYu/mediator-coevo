# Task Instruction

Complete the project proposal document by filling in placeholders and making specific modifications, then save the result as a valid .hwpx package.

## Step 1: Understand the .hwpx format
A .hwpx file is a ZIP archive containing XML files (similar to .docx). Inspect its structure first.

```bash
cd /root
ls -la
file project_proposal_template.hwpx
mkdir -p hwpx_work
cp project_proposal_template.hwpx hwpx_work/template.zip
cd hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```

## Step 2: Read the JSON data
```bash
cat /root/project_proposal.json
```
Note all key-value pairs. You'll need to map each key to a `{{key}}` placeholder in the XML files.

## Step 3: Inspect all XML content files
Examine every XML file inside the extracted archive, especially files under `Contents/` (commonly `section0.xml`, `section1.xml`, etc.) or similar paths. Look for:
- All `{{...}}` placeholder patterns
- Phase lines containing `단계1`, `단계2`, `단계3` with date ranges
- Budget values
- Any layout-cache elements (often `<hp:linesegarray>`, `<hp:lineseg>`, or similar cached layout data within paragraph elements)

```bash
grep -r '{{' template_contents/
grep -r '단계' template_contents/
grep -r 'lineseg\|lineSegArray\|LineSeg\|LINESEG' template_contents/ | head -20
```

## Step 4: Perform all replacements using a Python script
Write a Python script that:

1. Reads `project_proposal.json`
2. Extracts the template zip.
3. For each XML file in the archive:
   a. Replaces every `{{placeholder}}` with the corresponding JSON value.
   b. For the budget field: removes commas from the numeric part while keeping the leading currency symbol (e.g., `₩1,000,000,000` → `₩1000000000`, or `$1,234,567` → `$1234567`). The budget key in JSON might be something like `budget` or `총예산` — check the actual key name.
   c. After each line containing `단계1` (Phase 1), `단계2` (Phase 2), or `단계3` (Phase 3), appends a parenthesized month span. Calculate the month span from the date range already present in that line. For example, if a line says `2025.01 ~ 2025.03`, that's 3 months → append ` (3개월)`. The expected values per the task are: 단계1 → (3개월), 단계2 → (3개월), 단계3 → (1개월).
   d. Removes stale layout-cache elements from any paragraph whose text content was modified. These are typically `<hp:linesegarray>...</hp:linesegarray>` or `<hp:lineSegArray>...</hp:lineSegArray>` elements nested inside `<hp:p>` paragraph elements. Use XML parsing (lxml or ElementTree with namespace handling) to identify modified paragraphs and strip these elements.
   e. Ensures no `{{...}}` patterns remain anywhere.
4. Repackages everything back into a valid ZIP file saved as `/root/project_proposal_ready.hwpx`.

IMPORTANT implementation notes:
- **Namespace handling**: .hwpx XML files use namespaces extensively. When parsing, preserve all namespaces. Use `lxml.etree` if available, or `xml.etree.ElementTree` with proper namespace registration.
- **Placeholders may be split across XML elements**: A single `{{placeholder}}` might be split across multiple `<hp:t>` or similar text run elements due to formatting. You need to handle this — either by concatenating adjacent text runs, doing the replacement, and splitting back, or by working on the serialized XML string level carefully.
- **Strategy for placeholder replacement**: A robust approach is to first try XML-level replacement within text elements. If placeholders span multiple elements, fall back to string-level replacement on the raw XML, then re-parse to validate and strip layout caches.
- **Budget normalization**: The JSON value for budget likely contains commas. Remove commas from the numeric portion only, preserving the currency symbol.
- **Month span appending**: Find the text content of lines with 단계N and date ranges, calculate or hardcode the month spans (단계1→3개월, 단계2→3개월, 단계3→1개월), and append ` (N개월)` to the appropriate text element.
- **Layout cache removal**: For every `<hp:p>` (or equivalent paragraph element) where you modified any child text, find and remove `lineSegArray` or `linesegarray` child elements (check the actual element name in the XML namespace).
- **ZIP repackaging**: Preserve the original ZIP structure exactly. Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression. Preserve directory structure and file ordering.

## Step 5: Validate the output
```bash
# Check it's a valid zip
file /root/project_proposal_ready.hwpx

# Extract and verify no placeholders remain
mkdir -p /root/hwpx_verify
cp /root/project_proposal_ready.hwpx /root/hwpx_verify/verify.zip
cd /root/hwpx_verify
unzip verify.zip -d verify_contents
grep -r '{{' verify_contents/
# Should return nothing

# Verify phase month spans are present
grep -r '개월' verify_contents/
# Should show (3개월), (3개월), (1개월)

# Verify budget has no commas in numeric part
grep -r 'budget_value_pattern' verify_contents/  # adjust pattern based on actual budget value

# Verify Korean labels and static note are preserved
grep -r '참고\|비고\|NOTE' verify_contents/  # adjust based on actual static note content

# Verify layout caches removed from modified paragraphs
grep -ri 'lineseg' verify_contents/
# Check that modified paragraphs don't have these elements
```

## Key constraints to remember:
- ALL `{{...}}` placeholders must be replaced — check across ALL files in the archive, not just section XML files
- Budget: remove commas, keep currency symbol
- Phase lines: append month span in parentheses
- Modified paragraphs: strip layout cache elements
- Korean labels and static note: preserve unchanged
- Output must be a valid .hwpx (ZIP) package at `/root/project_proposal_ready.hwpx`

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