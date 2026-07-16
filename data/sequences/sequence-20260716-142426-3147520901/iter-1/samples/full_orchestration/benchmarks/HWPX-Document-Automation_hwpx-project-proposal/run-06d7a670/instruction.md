# Task Instruction

Complete the project proposal document by following these steps exactly:

## 1. Inspect the workspace
```bash
find /root -maxdepth 2 -type f | head -40
ls -la /root/
```
Identify where `project_proposal_template.hwpx` and `project_proposal.json` are located.

## 2. Understand the HWPX format
A `.hwpx` file is a ZIP archive containing XML files. Unzip the template to inspect its structure:
```bash
mkdir -p /tmp/hwpx_work
cp <path_to_template> /tmp/hwpx_work/template.hwpx
cd /tmp/hwpx_work
unzip template.hwpx -d template_contents
find template_contents -type f
```

## 3. Read the JSON data
```bash
cat <path_to_project_proposal.json>
```
Note every key-value pair. Pay special attention to:
- The budget value: it likely has commas (e.g., `₩1,500,000,000`). You must remove the commas but keep the currency symbol (e.g., `₩1500000000`).
- Phase date ranges that will be used to compute month spans.

## 4. Find all placeholders
Search every XML file inside the unpacked HWPX for `{{` patterns:
```bash
grep -rn '{{' template_contents/
```
Also search for any existing layout-cache elements (commonly `<hp:linesegarray>`, `<hp:lineSegArray>`, or similar `lineSeg` elements inside paragraph tags). Note their locations.

## 5. Examine the XML structure carefully
For each XML file containing placeholders, read the full file contents. Pay attention to:
- How text runs are structured (text may be split across multiple `<hp:t>` or `<t>` elements within a single paragraph)
- The exact element names for layout caches (e.g., `<hp:linesegarray>` or `<hp:lineSegArray>` blocks)
- Korean labels and static note lines that must be preserved exactly

## 6. Write a Python script to perform all modifications
Create `/tmp/hwpx_work/process.py` that:

a) Reads the JSON file and builds a placeholder-to-value mapping.

b) For the budget value, strips commas but preserves the leading currency symbol.

c) For each XML file in the unpacked HWPX:
   - Parses it as XML (use `lxml.etree` if available, otherwise `xml.etree.ElementTree`).
   - Finds all text nodes and replaces every `{{placeholder}}` with the corresponding JSON value.
   - **Important**: Text may be split across multiple child elements within a run. You must handle cases where `{{` appears in one `<hp:t>` element and `}}` appears in another. Consider concatenating sibling text elements, performing replacement, then writing back. Alternatively, work on the raw XML string with regex if the structure is too fragmented.
   - For phase lines (단계1, 단계2, 단계3): after filling placeholders, find lines containing date ranges like `2025.01 ~ 2025.03` and compute the month span. The month span = (end_year*12 + end_month) - (start_year*12 + start_month). Append ` (N개월)` after the date range text on the same line/paragraph.
   - Remove stale layout-cache elements: In any paragraph element (`<hp:p>` or similar) whose text content was modified, find and remove child elements that represent layout caches. These are typically `<hp:linesegarray>` (or `<hp:lineSegArray>`) elements. Remove them entirely so the document opens cleanly.
   - Preserve all Korean labels and static note lines unchanged.

d) Writes modified XML files back, preserving the XML declaration and encoding.

e) Repacks everything into a valid HWPX (ZIP) file at `/root/project_proposal_ready.hwpx`, preserving the original directory structure inside the ZIP. Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression.

## 7. Run the script
```bash
cd /tmp/hwpx_work
python3 process.py
```

## 8. Validate the output
```bash
# Verify it's a valid ZIP
unzip -t /root/project_proposal_ready.hwpx

# Unpack and check no placeholders remain
mkdir -p /tmp/hwpx_verify
unzip /root/project_proposal_ready.hwpx -d /tmp/hwpx_verify
grep -rn '{{' /tmp/hwpx_verify/
# Should return nothing

# Verify month spans are present
grep -rn '개월' /tmp/hwpx_verify/
# Should show (3개월), (3개월), (1개월)

# Verify budget has no commas
grep -rn '₩' /tmp/hwpx_verify/ | head -5
# Currency symbol should be present, no commas in the number

# Verify no linesegarray in modified paragraphs
# (Check that layout caches were removed from edited paragraphs)
```

## Key Warnings
- **Split text runs**: HWPX XML commonly splits what looks like one string across multiple `<hp:t>` elements within `<hp:run>` elements within a paragraph. A placeholder like `{{project_name}}` might be split as `{{projec` in one element and `t_name}}` in another. You MUST handle this. A robust approach: for each paragraph, collect all text content, do replacements on the concatenated string, then redistribute or place the full text in the first text element and clear the rest.
- **Namespace handling**: HWPX XML files use namespaces. When parsing, register or handle namespaces properly so elements aren't lost or renamed.
- **Month calculation**: Parse dates like `2025.01` as year=2025, month=1. The span between `2025.01 ~ 2025.03` is 3 months (inclusive: March minus January + 1, OR exclusive end: just end_month - start_month... check which gives the expected results: 단계1→3개월, 단계2→3개월, 단계3→1개월). Verify against the expected outputs.
- **File must be at exactly `/root/project_proposal_ready.hwpx`**.
- **Do not modify Korean labels or static note lines.**

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