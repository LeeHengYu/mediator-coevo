# Task Instruction

Complete the following task to prepare an event announcement HWPX document.

## Goal
Replace all `{{...}}` placeholders in `event_announcement_template.hwpx` with values from `event_data.json` and save the result to `/root/event_announcement_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `event_data.json` to understand all available key-value pairs.
- A `.hwpx` file is a ZIP archive containing XML files. List the contents of `event_announcement_template.hwpx` using `unzip -l` or Python's `zipfile` module.
- Extract the archive to a temporary directory (e.g., `/tmp/hwpx_work/`).
- Search all extracted files (especially XML files under `Contents/`) for `{{` to find every placeholder. Print each match with filename and line content so you know exactly what needs replacing.

### 2. Understand the HWPX XML structure
- The main document content is typically in files like `Contents/section0.xml` (or similar).
- Placeholders like `{{event_name}}` may be split across multiple XML text runs/elements. Check carefully whether placeholders are contiguous within single text nodes or split across multiple `<hp:t>` (or similar) elements.
- If placeholders are split across elements, you must handle reassembly: join adjacent text runs, perform the replacement, then place the result back.
- Identify layout-cache elements: look for elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hp:lineseg>`, or `<hp:LineSeg>` or any element that appears to cache character positions/layout within paragraphs. These are typically children of paragraph (`<hp:p>`) elements.

### 3. Perform replacements using Python
Write a Python script that:

a) Reads `event_data.json` into a dictionary.

b) Extracts the `.hwpx` ZIP to a temp directory.

c) For each file in the extracted archive, reads the file content. For XML/text files:
   - First, check if any `{{` placeholder exists in the raw text of the file.
   - If placeholders are found, parse the XML properly (use `lxml` or `xml.etree.ElementTree`). 
   - **Critical**: Placeholders may be split across multiple adjacent `<hp:t>` (or `<t>`) text elements within a single paragraph run. To handle this:
     - For each paragraph element, concatenate all text content, check for `{{...}}` patterns, and if found, perform replacements on the concatenated text, then redistribute or place the full replaced text into the first text node and clear the others.
   - Alternatively, a simpler approach: read the entire XML file as a raw string, perform regex-based replacement of `{{key}}` with the corresponding JSON value for all keys. This works if placeholders are NOT split across elements. Check first whether this simpler approach catches all placeholders.
   - For each paragraph element that was modified, remove any layout-cache child elements (e.g., `<hp:linesegarray>`, `<hp:lineSegArray>`, `<linesegarray>`, or similar). These elements cache glyph/character positions and become stale after text changes.

d) After processing, verify no `{{` remains in any file by scanning all files.

e) Repackage the modified directory back into a ZIP file saved as `/root/event_announcement_ready.hwpx`. Use `zipfile.ZipFile` with `ZIP_DEFLATED` compression. Preserve the original directory structure exactly.

### 4. Validate the output
- Run `unzip -l /root/event_announcement_ready.hwpx` to confirm it's a valid ZIP.
- Search the entire archive for any remaining `{{` patterns: extract and grep, or use Python to scan all entries. There must be zero matches.
- Confirm Korean labels and static note lines are unchanged by spot-checking the content XML.
- Confirm layout-cache elements were removed from modified paragraphs.

### Important Notes
- Do NOT change any Korean text labels or the static note line — only replace `{{...}}` placeholders.
- Ensure every key from `event_data.json` that has a corresponding `{{key}}` placeholder is replaced.
- If a placeholder key in the template doesn't exist in the JSON, report it but do not leave `{{...}}` in the output.
- The final file MUST be at exactly `/root/event_announcement_ready.hwpx`.
- When repackaging, make sure the `mimetype` file (if present) is stored first and uncompressed, as is convention for OPC/ZIP-based document formats. Check the original ZIP for this pattern and replicate it.

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