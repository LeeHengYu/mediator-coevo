# Task Instruction

Complete the project proposal document by following these steps precisely:

## 1. Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files (similar to OOXML). You will need to:
- Unzip the template
- Edit the XML content files
- Repackage as a valid ZIP with `.hwpx` extension

## 2. Inspect the template and data
```bash
mkdir -p /tmp/hwpx_work
cp /root/project_proposal_template.hwpx /tmp/hwpx_work/
cd /tmp/hwpx_work
unzip -o project_proposal_template.hwpx -d template_extracted
find template_extracted -type f | sort
```
Also read the JSON data:
```bash
cat /root/project_proposal.json
```

## 3. Find all content XML files containing placeholders
Search for `{{` across all extracted files:
```bash
grep -rl '{{' template_extracted/
```
Then inspect each matching file carefully to understand the XML structure, placeholder locations, and any layout-cache elements.

## 4. Perform replacements with a Python script
Write a Python script `/tmp/hwpx_work/process.py` that:

a. **Loads the JSON data** from `/root/project_proposal.json`.

b. **For each XML file in the extracted template** that contains `{{...}}` placeholders:
   - Parse the XML preserving namespaces, declarations, and structure.
   - Replace every `{{placeholder}}` with the corresponding value from the JSON. Map placeholder names to JSON keys carefully (inspect both to establish the mapping).
   - **Budget normalization**: For the budget value, remove commas but keep the leading currency symbol (e.g., `₩1,500,000,000` becomes `₩1500000000`).
   - **Month span appending**: For lines containing phase information (단계1, 단계2, 단계3), parse the date range already in that line to calculate the month span, then append ` (N개월)` after the phase text. Specifically:
     - Look at the date ranges in the phase lines (e.g., `2025.01 ~ 2025.03` means 3 months).
     - Calculate months as: (end_year - start_year) * 12 + (end_month - start_month + 1) if both endpoints are inclusive, or determine from the actual dates.
     - Append the parenthesized month count like `(3개월)` to the text content of the paragraph/run containing that phase.
   - **Remove stale layout-cache elements**: In any paragraph (`<hp:p>` or similar) whose text content was modified, remove any `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, or similar layout-cache child elements. These are pre-computed glyph positioning data that becomes stale after text changes. Search for elements with names like `linesegarray`, `LineSeg`, `lineseg`, `lineSegArray`, `charShapeArray` cache entries, etc. Inspect the actual XML to identify the exact element names used.
   - **Preserve all Korean labels and static note lines unchanged.**
   - **Ensure no `{{...}}` text remains** in any output file.

c. **Write modified XML files** back to the extracted directory, preserving XML declarations and encoding.

d. **Repackage** the modified extracted directory into `/root/project_proposal_ready.hwpx` as a ZIP file. Use the same directory structure. The ZIP should be created from within the extracted directory so paths are relative (no leading directory name unless the original had one). Match the original archive structure exactly.

## 5. Validate the result
After creating the output:
```bash
# Verify it's a valid ZIP
unzip -t /root/project_proposal_ready.hwpx

# Verify no placeholders remain
unzip -p /root/project_proposal_ready.hwpx | grep -c '{{'
# Should output 0

# Check that month spans were added
unzip -p /root/project_proposal_ready.hwpx | grep '개월'

# Check budget has no commas
unzip -p /root/project_proposal_ready.hwpx | grep -oP '₩[\d,]+' 
# Should show no commas in the number

# Verify Korean labels are present
unzip -p /root/project_proposal_ready.hwpx | grep -c '프로젝트\|과제\|예산\|단계'
```

## Critical details to watch for:
- **XML namespace handling**: Use `lxml` with `etree` or `xml.etree.ElementTree` but be careful with namespace prefixes. If using ElementTree, register namespaces before parsing to avoid ns0/ns1 prefix pollution.
- **Layout cache elements**: After inspecting the XML structure, identify the exact tag names for layout cache elements (likely within `<hp:p>` elements). Remove them from any modified paragraph. Common names in HWPX: `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar.
- **ZIP packaging**: When repackaging, ensure `mimetype` file (if present) is stored first and uncompressed (common in OPC packages). Walk the extracted directory and add files maintaining the original relative paths.
- **Text content in HWPX XML**: Text may be in `<hp:t>` elements within `<hp:run>` elements within `<hp:p>` elements. Inspect carefully.
- **Month calculation**: Be precise. If a range is `2025.01 ~ 2025.03`, that's January through March = 3 months. If `2025.04 ~ 2025.06`, that's 3 months. If `2025.07 ~ 2025.07`, that's 1 month.

The final output MUST be at `/root/project_proposal_ready.hwpx`.

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