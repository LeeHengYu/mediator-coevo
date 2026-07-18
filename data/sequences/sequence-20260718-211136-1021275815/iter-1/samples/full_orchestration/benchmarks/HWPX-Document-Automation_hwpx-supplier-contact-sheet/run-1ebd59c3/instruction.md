# Task Instruction

Complete the following task step by step:

## Goal
Update the HWPX supplier contact sheet `supplier_contact_template.hwpx` using values from `supplier_contact.json`, saving the result to `/root/supplier_contact_ready.hwpx`.

## Steps

### 1. Inspect the input files
- Read `supplier_contact.json` to understand all key-value pairs available for substitution.
- Examine the HWPX file structure: HWPX files are ZIP archives. List the contents of `supplier_contact_template.hwpx` using Python's `zipfile` module.
- Identify which XML files inside the archive contain `{{...}}` placeholders by reading each file and searching for the pattern `{{`. Pay special attention to files in the `Contents/` directory (e.g., `section0.xml` or similar content XML files).

### 2. Understand the placeholder mapping
- For each `{{placeholder_name}}` found in the XML content, verify there is a corresponding key in the JSON file.
- Note the exact placeholder syntax (e.g., `{{회사명}}`, `{{담당자}}`, etc.).

### 3. Write a Python script to perform the transformation
Create a Python script that:

a. **Reads the JSON file** and loads all key-value pairs.

b. **Opens the HWPX ZIP archive** and reads all entries.

c. **For each file in the archive:**
   - If it's an XML/text file that contains `{{` placeholders:
     - Replace every `{{key}}` with the corresponding value from the JSON.
     - **Critical: Remove stale layout cache elements.** After replacing placeholders in any paragraph, parse the XML and remove all `linesegarray` elements (and their children) from paragraphs whose text content was modified. These layout cache elements cause overlapping characters when the document is opened. Use an XML parser (like `lxml.etree` or `xml.etree.ElementTree`) for this step — do NOT rely on regex for XML structural changes.
     - Be careful with XML namespaces. Inspect the actual namespace URIs used in the document and handle them properly when searching for and removing `linesegarray` elements.
   - If it's a binary file or doesn't contain placeholders, copy it unchanged.

d. **Write the result** to `/root/supplier_contact_ready.hwpx` as a valid ZIP archive, preserving the original compression method and directory structure.

### 4. Detailed approach for layout cache removal
- Parse the XML content of modified files.
- Identify the namespace for paragraph and linesegarray elements (look at the root element's namespace declarations).
- For each paragraph element that contained a `{{...}}` placeholder (i.e., whose text was modified):
  - Find and remove any child `linesegarray` element (including all its descendants).
- Serialize the modified XML back, preserving the XML declaration and encoding.

### 5. Validate the output
- Open `/root/supplier_contact_ready.hwpx` as a ZIP and verify it's valid.
- Read all XML content files from the output and confirm:
  - No `{{` or `}}` patterns remain anywhere in any file.
  - Korean field labels are still present (e.g., search for a few known Korean labels from the template).
  - The static note line is unchanged.
  - All JSON values appear in the output.
  - No `linesegarray` elements exist in paragraphs that were modified.
- Print a summary of validation results.

### 6. Run the verifier
- If there is a test file (e.g., `test_output.py` or similar), run it with `pytest` to confirm the output passes all checks.
- Look for any test or verification script in the task directory and execute it.

## Important constraints
- Do NOT remove or weaken any tests or verification scripts.
- Preserve all Korean text labels already in the document.
- Ensure the static note line is untouched.
- The output must be a valid HWPX (ZIP) package.
- No `{{...}}` placeholders may remain in any file within the archive.

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