# Task Instruction

## Task: Complete the HWPX Project Proposal Document

### Goal
Fill in the template `project_proposal_template.hwpx` using values from `project_proposal.json`, then save the result to `/root/project_proposal_ready.hwpx`.

### Step-by-step Plan

#### 1. Explore the workspace and understand the inputs
```bash
cd /root
ls -la
find . -name 'project_proposal_template.hwpx' -o -name 'project_proposal.json' 2>/dev/null
```
Read the JSON file:
```bash
cat project_proposal.json
```

#### 2. Understand the HWPX structure
A `.hwpx` file is a ZIP archive containing XML files. Unzip the template to inspect its contents:
```bash
mkdir -p /tmp/hwpx_work
cp project_proposal_template.hwpx /tmp/hwpx_work/template.zip
cd /tmp/hwpx_work
unzip template.zip -d template_contents
find template_contents -type f
```
Identify which XML files contain the document body (likely under `Contents/` — look for files like `section0.xml` or `content.hpf` etc.).

#### 3. Inspect all XML files for `{{...}}` placeholders
```bash
grep -r '{{' template_contents/
```
This tells you exactly which files need editing and what placeholders exist. List every placeholder found.

#### 4. Read the JSON and map placeholders to values
Parse `project_proposal.json` and create a mapping from each `{{placeholder}}` to its replacement value. Pay attention to:
- **Budget normalization**: Remove commas from the budget number but keep the leading currency symbol (e.g., `₩1,500,000` → `₩1500000`).
- All other values are inserted as-is from the JSON.

#### 5. Write a Python script to perform all replacements
Create a Python script that:
1. Copies the template HWPX to the output path.
2. Opens it as a ZIP, reads each XML file.
3. For every XML file that contains `{{...}}` placeholders, performs text replacement using the JSON values (with budget normalization applied).
4. **Appends month spans to phase lines**: For lines containing `단계1`, `단계2`, `단계3`, calculate the month span from the date range already present in that line's text, then append ` (N개월)` to the text. The expected values per the instructions are: `단계1` → `(3개월)`, `단계2` → `(3개월)`, `단계3` → `(1개월)`.
5. **Removes stale layout-cache elements** from any modified paragraph. In HWPX XML, these are typically `<hp:linesegarray>` or `<lineseg>` elements (or similar caching elements within `<hp:p>` paragraph nodes). For any `<hp:p>` (or equivalent paragraph element) whose text content was modified, find and remove all child elements that represent layout caches (e.g., `<hp:linesegarray>`, `<linesegarray>`, elements with tag containing `lineseg` or `lineSegArray`). Inspect the actual XML namespace and tag names before writing removal code.
6. Verifies no `{{` remains in any XML content.
7. Writes the modified XML files back into a new ZIP archive saved as `/root/project_proposal_ready.hwpx`, preserving the exact directory structure and all unmodified files.

#### 6. Validate the output
```bash
# Check it's a valid ZIP
unzip -t /root/project_proposal_ready.hwpx

# Check no placeholders remain
mkdir -p /tmp/hwpx_verify
unzip /root/project_proposal_ready.hwpx -d /tmp/hwpx_verify
grep -r '{{' /tmp/hwpx_verify/ && echo 'FAIL: placeholders remain' || echo 'PASS: no placeholders'

# Check month spans are present
grep -r '개월' /tmp/hwpx_verify/

# Check budget has no commas but has currency symbol
grep -r '₩' /tmp/hwpx_verify/ || grep -r '원' /tmp/hwpx_verify/

# Check Korean labels are preserved
grep -r '단계' /tmp/hwpx_verify/
```

#### 7. Run the verifier if available
```bash
cd /root
ls test_output.py verify.py 2>/dev/null
# If test_output.py exists:
python -m pytest test_output.py -v
```

### Critical Details
- **Namespace awareness**: When parsing HWPX XML, use `lxml.etree` or `xml.etree.ElementTree` with proper namespace handling. Inspect actual namespace URIs from the XML files before writing XPath queries.
- **Layout cache removal**: This is essential. Any paragraph you modify must have its `linesegarray` (or equivalent layout cache) child elements removed so the document renders correctly. Inspect the XML structure to find the exact element names.
- **ZIP structure preservation**: When repackaging, preserve the `mimetype` file (if present) and the exact directory layout. Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression. If the original has a `mimetype` entry stored without compression, replicate that.
- **Encoding**: Write XML as UTF-8.
- **No `{{...}}` may remain**: Double-check every XML file in the output archive.

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