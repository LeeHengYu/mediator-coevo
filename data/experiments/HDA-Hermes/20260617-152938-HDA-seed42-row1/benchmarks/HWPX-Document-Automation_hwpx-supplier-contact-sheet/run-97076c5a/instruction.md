# Task Instruction

Complete the following task to produce `/root/supplier_contact_ready.hwpx` from the template and JSON data.

## Overview
Replace all `{{...}}` placeholders in `supplier_contact_template.hwpx` using values from `supplier_contact.json`, then save the result as `/root/supplier_contact_ready.hwpx`.

## Step-by-step Instructions

### 1. Inspect the workspace
- List files in the current directory to locate `supplier_contact_template.hwpx` and `supplier_contact.json`.
- Read and display the full contents of `supplier_contact.json` to understand all available key-value pairs.

### 2. Examine the HWPX template structure
- HWPX files are ZIP archives. Use Python's `zipfile` module to list all entries in `supplier_contact_template.hwpx`.
- Print the full list of archive members.
- Read and print the contents of every XML file inside the archive (especially files under `Contents/` such as `content.hpf`, `header.xml`, `section0.xml`, etc.). Focus on identifying where `{{...}}` placeholders appear.
- Also check `META-INF/` and any other text-based files for placeholders.
- Print each file's raw text content so you can see the exact XML structure, tag names, attributes, and placeholder locations.

### 3. Identify all placeholders
- Use regex `\{\{[^}]+\}\}` to scan every text-based file in the archive and list every unique placeholder found, along with which file(s) it appears in.
- Cross-reference these with the JSON keys to ensure every placeholder has a corresponding value.

### 4. Write a Python script to perform the replacement
Create a Python script that:

a. Opens `supplier_contact.json` and loads the key-value mapping.

b. Opens `supplier_contact_template.hwpx` as a ZIP archive.

c. For each file in the archive:
   - If it is a text/XML file, read it as UTF-8 text.
   - Replace every `{{KEY}}` with the corresponding value from the JSON (use exact string replacement for each key found in JSON).
   - **Critical: Remove stale layout cache elements from any paragraph whose text was modified.** In HWPX XML, layout caches are typically elements like `<hp:linesegarray>`, `<hp:lineSegArray>`, `<hc:lineseg>`, `<hp:parameterset>` with layout-related names, or elements in namespaces related to caching/layout. Specifically:
     - Parse modified XML files with an XML parser (lxml or ElementTree) that preserves namespaces.
     - For any paragraph element (`<hp:p>`, `<hp:paragraph>`, or similar) that contained a placeholder that was replaced, remove child elements that represent layout caches. These are commonly `<hp:linesegarray>` or similar elements. Inspect the actual XML structure to identify the correct element names.
     - If using string-based replacement initially, then parse with XML afterwards to strip layout cache elements from modified paragraphs.
   - If it is a binary file, copy it as-is.

d. Write all files (modified and unmodified) into a new ZIP archive at `/root/supplier_contact_ready.hwpx`, preserving the original directory structure and compression settings.

### 5. Validate the output
- Open `/root/supplier_contact_ready.hwpx` with `zipfile` and verify it is a valid ZIP.
- List all members and confirm they match the original template's members.
- Scan all text/XML files in the output for any remaining `{{...}}` patterns. Print results. **There must be zero remaining placeholders.**
- Print the full text content of the main content XML file(s) (e.g., `section0.xml`) from the output to visually confirm replacements were made correctly and Korean labels are preserved.
- Confirm the static note line is unchanged by comparing it between template and output.
- Confirm that layout cache elements have been removed from modified paragraphs.

### Important Notes
- Do NOT assume the XML structure. Read and inspect actual file contents first, then adapt your parsing/replacement logic to match what you find.
- Preserve all non-placeholder content exactly, including Korean text, formatting tags, and document structure.
- The placeholder keys in the JSON may not have the `{{}}` wrapper — you need to wrap them when matching (i.e., if JSON has key `company_name`, match `{{company_name}}` in the XML).
- Be careful with XML namespaces. Use namespace-aware parsing.
- If placeholders span across multiple XML elements (e.g., `{{` in one text run and `}}` in another), handle this case by examining the actual XML. If text runs are split, you may need to merge them or do replacement at the serialized text level carefully.
- Ensure the output ZIP uses the same compression method as the original for each entry.

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