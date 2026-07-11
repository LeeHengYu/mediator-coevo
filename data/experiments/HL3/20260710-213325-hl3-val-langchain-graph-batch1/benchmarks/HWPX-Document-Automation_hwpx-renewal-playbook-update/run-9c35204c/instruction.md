# Task Instruction

Complete the following task step by step.

## Objective
Revise the existing renewal playbook `renewal_playbook.hwpx` using `renewal_update.json` and `followups.csv`, then save the updated file to `/root/renewal_playbook_updated.hwpx`.

## Steps

### Step 1: Inspect the workspace
- List files in the current working directory and `/root/` to locate `renewal_playbook.hwpx`, `renewal_update.json`, and `followups.csv`.
- Read `renewal_update.json` completely to understand the field mappings (customer name, current owner, renewal window, pricing band, escalation contact, pricing note — both old and new values).
- Read `followups.csv` completely to understand the follow-up items and their `sequence` column for ordering.

### Step 2: Extract and explore the HWPX package
- Copy `renewal_playbook.hwpx` to a working directory (e.g., `/tmp/hwpx_work/`).
- Unzip it there (HWPX is a ZIP archive).
- List all extracted files to understand the package structure.
- Identify the main content XML files (typically under `Contents/` — look for files like `section0.xml`, `content.hpf`, etc.).
- Read and display the full content of each XML file that contains document text (especially section XML files). Also read `META-INF/` and any manifest files.

### Step 3: Understand the document content
- Identify all editable text sections in the XML where the old values from `renewal_update.json` appear.
- Identify the three existing follow-up lines that need to be replaced.
- Confirm the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` exists and note its location.

### Step 4: Write a Python script to perform all updates
Create a Python script that does the following:

1. **Extract** the HWPX file to a temp directory.
2. **Read** `renewal_update.json` to get old→new value mappings.
3. **Read** `followups.csv` and sort/order rows by the `sequence` column.
4. **Process each XML content file** (section files):
   a. Parse the XML content.
   b. **Replace field values**: For each field in the JSON (customer name, current owner, renewal window, pricing band, escalation contact, pricing note), find every occurrence of the old value in text nodes and replace with the new value.
   c. **Replace follow-up lines**: Find the three existing follow-up text lines and replace them with the CSV items in sequence order. Remove old follow-up paragraphs and insert new ones in their place — do NOT leave duplicates. Be careful to match the exact structure: if follow-ups are in separate `<hp:p>` paragraph elements, replace the text content of those paragraphs; if there are exactly 3 old ones and a different number of new ones, adjust paragraph count accordingly.
   d. **Verify** the appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` is untouched.
   e. **Remove stale layout cache**: For every `<hp:p>` paragraph whose text content was modified, remove any `<hp:lineSegArray>` elements (and their children) within that paragraph. Use namespace-aware matching — the namespace prefix might vary, so match on local name `lineSegArray` or use the full namespace URI.
   f. Write the modified XML back.
5. **Repackage** all files back into a ZIP with `.hwpx` extension at `/root/renewal_playbook_updated.hwpx`. Use `zipfile.ZIP_DEFLATED` compression. Preserve the original directory structure exactly. Make sure `mimetype` file (if present) is stored first and uncompressed (ZIP_STORED), similar to ODF/EPUB convention — but check the original archive's structure first and replicate it.

### Step 5: Run the script and validate
- Execute the Python script.
- Verify the output file exists at `/root/renewal_playbook_updated.hwpx`.
- Extract the output file to a new temp directory and inspect the XML content to confirm:
  - All old field values have been replaced with new values (search for old values — none should remain in editable sections).
  - The follow-up lines match the CSV items in correct sequence order.
  - The appendix sentence is preserved exactly.
  - No `<hp:lineSegArray>` elements remain in modified paragraphs.
  - The file is a valid ZIP archive.

### Important Details
- **Do NOT add new values alongside old values** — old values must be fully replaced/removed.
- **Namespace handling**: HWPX XML uses namespaces like `http://www.hancom.co.kr/hwpml/2011/paragraph` etc. When searching for `lineSegArray`, handle both prefixed (`hp:lineSegArray`) and namespace-URI-based matching.
- **Text node handling**: Values may span across text runs (`<hp:t>` elements) within a paragraph. If a value is split across multiple runs, you may need to handle concatenation or do string replacement at the serialized XML text level (regex on the XML string) rather than purely DOM-based replacement. Inspect the actual XML structure first to determine the best approach.
- **Encoding**: Preserve UTF-8 encoding for Korean text.
- **If the JSON has keys like `old_*` and `new_*`**, map them correctly. Read the actual JSON structure before coding.
- **Follow-up replacement**: Carefully identify what the existing follow-up lines look like in the XML. They might contain numbering or specific formatting. Match and replace the content appropriately.

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