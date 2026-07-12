# Task Instruction

Complete the inventory status report by replacing placeholders in a .hwpx template with JSON data values.

## Steps

### 1. Inspect the input files
- Read `/root/inventory_data.json` to understand all available key-value pairs.
- Extract `/root/inventory_report_template.hwpx` to a temporary directory (e.g., `/tmp/hwpx_work/`) using `unzip`.
- List all files in the extracted archive to understand the package structure.
- Identify which XML files contain document content (typically `Contents/section*.xml` or similar).
- Read and examine each content XML file to find all `{{...}}` placeholders.

### 2. Understand placeholder distribution
- Placeholders like `{{key_name}}` may be split across multiple `<hp:t>` (or similar text run) elements within a single paragraph. For example, `{{` might be in one `<hp:t>` tag, `key` in another, and `}}` in yet another.
- You MUST handle this fragmentation. The safest approach: for each `<hp:p>` paragraph element, concatenate ALL text content from its `<hp:t>` descendants, perform placeholder replacements on the concatenated string, then place the result in a single `<hp:t>` element (removing the extra ones).

### 3. Write a Python script to perform the transformation
Create a Python script that:

a) **Loads the JSON data** from `/root/inventory_data.json`.

b) **Parses each content XML file** using `xml.etree.ElementTree` (with proper namespace handling).

c) **For each paragraph (`<hp:p>`):**
   - Collect all `<hp:t>` elements (searching recursively within the paragraph's run elements).
   - Concatenate their text content.
   - If the concatenated text contains any `{{...}}` pattern:
     - Replace ALL `{{key}}` occurrences with the corresponding value from the JSON data (convert numbers to strings as needed).
     - Put the fully replaced text into the FIRST `<hp:t>` element.
     - Clear or remove the text from all subsequent `<hp:t>` elements in that paragraph (or remove the extra run elements if safe, but at minimum clear their text).
   - **Remove any `<hp:linesegarray>` element** (layout cache) from paragraphs where text was modified. This is critical to prevent overlapping characters when the document is opened.

d) **Preserve everything else unchanged:** Korean labels, static note lines, empty paragraphs, and all non-content XML files.

e) **Write the modified XML back** to the same file paths in the extracted directory, preserving XML declarations and encoding.

### 4. Repack the .hwpx file
Repack the modified files into `/root/inventory_report_ready.hwpx` as a ZIP archive with these critical requirements:
- The `mimetype` file MUST be the **first entry** in the ZIP archive.
- The `mimetype` file MUST be stored **uncompressed** (compression method = ZIP_STORED, no extra field).
- All other files should be compressed normally (ZIP_DEFLATED).
- Use Python's `zipfile` module for precise control over entry order and compression.

### 5. Validate the output
- Extract the newly created `/root/inventory_report_ready.hwpx` to a separate temp directory.
- Read all content XML files and verify:
  - **No `{{` or `}}` patterns remain anywhere** in any text content.
  - Korean labels and static note text are preserved.
  - Empty paragraphs still exist in the document structure.
  - The XML is well-formed (parseable).
- Verify the ZIP structure: `mimetype` is the first entry and is uncompressed.
- Print a summary of all replacements made.

### Important Notes
- Be careful with XML namespaces. The HWPX format uses namespaces like `http://www.hancom.co.kr/hwpml/2016/HwpMl` for `hp:` prefix. Register all namespaces before parsing to avoid namespace prefix changes in output.
- When writing XML back, use `xml.etree.ElementTree.register_namespace()` for all namespaces found in the original files to preserve the original namespace prefixes.
- Numbers from JSON should be converted to their string representation (e.g., integer 150 becomes "150", float values keep their decimal representation as-is).
- If a placeholder key from the template doesn't exist in the JSON, report it as an error but continue processing other placeholders.

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