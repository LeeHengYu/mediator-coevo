# Task Instruction

Complete the project proposal document by filling in placeholders and making specific modifications, then save the result as a valid .hwpx package.

## Background
A `.hwpx` file is a ZIP-based Korean word processor document (Hancom/Hangul). Inside the ZIP archive, the document content is typically in XML files (often under `Contents/` directory, e.g., `Contents/section0.xml`). You need to extract the archive, modify the XML content, and repackage it.

## Step-by-step Instructions

### 1. Examine the input files
- Read `/root/project_proposal.json` to understand all available field values.
- Extract/list the contents of `/root/project_proposal_template.hwpx` (it's a ZIP file: `unzip -l project_proposal_template.hwpx`).
- Extract the hwpx to a working directory: `mkdir -p /root/hwpx_work && cd /root/hwpx_work && unzip -o /root/project_proposal_template.hwpx`
- Find and read ALL XML files inside, especially any under `Contents/` (e.g., `section0.xml`, `section1.xml`, etc.). Also check for content in other XML files. Search exhaustively for `{{` across all extracted files: `grep -r '{{' /root/hwpx_work/`

### 2. Read and understand the JSON data
- Parse the JSON file to identify all key-value pairs that correspond to `{{...}}` placeholders.

### 3. Perform replacements in the XML content
For every XML file containing `{{...}}` placeholders:

#### a. Replace all `{{...}}` placeholders
- Replace each `{{placeholder_name}}` with the corresponding value from the JSON file.
- **Budget normalization**: For any budget/cost value, remove commas from the number but keep the leading currency symbol (e.g., `₩1,500,000,000` becomes `₩1500000000`).

#### b. Append month spans to phase lines
- Find lines containing phase descriptions (단계1, 단계2, 단계3). Each phase line should already contain a date range.
- Calculate the month span from the date range in each line:
  - 단계1: look at its date range and compute months → append ` (3개월)`
  - 단계2: look at its date range and compute months → append ` (3개월)`  
  - 단계3: look at its date range and compute months → append ` (1개월)`
- Append the parenthesized month span text to the same text run or paragraph element where the phase info appears. Make sure it's appended after the existing text content of that element/run.

#### c. Remove stale layout-cache elements
- For any paragraph (`<hp:p>` or similar) whose text content you modified, look for layout-cache child elements (often named something like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<lineseg>`, `<hp:LineSeg>`, or similar layout/cache elements). Remove these elements entirely from modified paragraphs so the document renders cleanly without overlapping characters.
- Search for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineSeg>` or any element that appears to be a layout cache within paragraph elements. Remove them from paragraphs you edited.

#### d. Preserve everything else
- Keep all Korean labels and static note lines unchanged.
- Keep XML structure, namespaces, and attributes intact for unmodified elements.
- Do not alter formatting, styles, or other document metadata unless necessary for the above changes.

### 4. Verify no placeholders remain
- After all replacements, run: `grep -r '{{' /root/hwpx_work/`
- This must return NO results. If any `{{...}}` remain, investigate and fix.

### 5. Repackage the .hwpx file
- The .hwpx must be repackaged as a valid ZIP with the same structure:
  ```
  cd /root/hwpx_work
  rm -f /root/project_proposal_ready.hwpx
  zip -r -X /root/project_proposal_ready.hwpx . -x '.*'
  ```
- Important: The ZIP must preserve the original directory structure exactly. Use `mimetype` file first if it exists (some ODF-like formats require it to be first and uncompressed): check if there's a `mimetype` file; if so, add it first with `zip -0 -X /root/project_proposal_ready.hwpx mimetype` then add the rest with `zip -r -X /root/project_proposal_ready.hwpx . -x mimetype -x '.*'`.

### 6. Final validation
- Verify the output exists: `ls -la /root/project_proposal_ready.hwpx`
- Verify it's a valid ZIP: `unzip -t /root/project_proposal_ready.hwpx`
- Extract and verify no `{{` placeholders remain: `unzip -p /root/project_proposal_ready.hwpx | grep -c '{{'` should be 0.
- Spot-check that budget values have no commas but retain currency symbols.
- Spot-check that phase lines contain the month span annotations.

## Key Details to Watch For
- The XML inside hwpx files uses Korean text and may have text split across multiple `<hp:t>` or similar text run elements within a single paragraph. A placeholder like `{{name}}` might be split as `{{` in one run and `name}}` in another. If so, you need to handle this by joining text runs or doing multi-element replacement.
- Use Python for XML manipulation if sed/awk becomes unwieldy. Python's `lxml` or `xml.etree.ElementTree` with namespace handling would work well.
- When removing layout cache elements, be thorough but surgical — only remove from paragraphs you actually modified.

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