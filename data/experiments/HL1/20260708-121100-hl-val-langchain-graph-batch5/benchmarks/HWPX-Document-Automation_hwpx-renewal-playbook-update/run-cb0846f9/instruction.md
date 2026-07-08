# Task Instruction

Execute the following steps to produce `/root/renewal_playbook_updated.hwpx`:

1. **Inspect source files.** Read and print:
   - The JSON file `renewal_update.json` (field names and new values).
   - The CSV file `followups.csv` (columns and rows; note the `sequence` column for ordering).
   - List the contents of the HWPX ZIP archive `renewal_playbook.hwpx` to confirm the internal structure (expect `Contents/section0.xml` among others).

2. **Extract and inspect `Contents/section0.xml`.** Read the XML from the ZIP. Print it (or a large portion) so you can see:
   - Where the customer name, current owner, renewal window, pricing band, escalation contact, and pricing note appear.
   - Where the three follow-up lines are.
   - The appendix sentence `이 부록 문단은 그대로 유지해야 합니다.` (must be preserved verbatim).

3. **Write a Python script** that does all of the following in one execution:

   a. **Parse inputs:**
      - Load `renewal_update.json` as a dict.
      - Load `followups.csv` with the `csv` module; sort rows by the `sequence` column (ascending integer).

   b. **Register XML namespaces** before parsing the XML so they are preserved on serialization. Use `xml.etree.ElementTree.register_namespace` for every namespace declared in the XML (inspect the root element's attributes for `xmlns:` prefixes — typically `hp`, `hc`, `hpf`, `hp10`, etc., plus the default namespace). This prevents ElementTree from rewriting prefixes.

   c. **Parse `Contents/section0.xml`** with `xml.etree.ElementTree`.

   d. **Build a replacement map** from the JSON fields. The JSON likely contains keys like `customer_name`, `current_owner`, `renewal_window`, `pricing_band`, `escalation_contact`, `pricing_note` with old→new or just new values. Identify the old values by scanning the XML text content, then map old→new for each field. If the JSON provides both old and new values, use them directly; otherwise, extract old values from the XML by matching field semantics.

   e. **Replace field values everywhere in editable paragraphs.** Walk all text-bearing elements (e.g., `.//hp:t` or any element whose `.text` contains the old values). For each old value found, replace it with the corresponding new value using `str.replace()`. Do NOT touch the appendix sentence.

   f. **Replace follow-up lines.** Identify the three existing follow-up paragraphs (they likely share a recognizable pattern or are consecutive). Remove them and insert new paragraphs (one per CSV row, in `sequence` order) using the same XML structure/tags as the originals. Copy the paragraph element structure from one of the originals as a template, then set the text content to each CSV follow-up item.

   g. **Remove `<hp:lineSegArray>` elements** from every paragraph whose text content was modified. This invalidates the layout cache so the viewer re-renders text without overlapping characters. Use `parent.remove(child)` pattern — iterate with `.//{namespace}lineSegArray` and remove each from its parent.

   h. **Verify the appendix sentence** `이 부록 문단은 그대로 유지해야 합니다.` still exists verbatim in the modified XML tree.

   i. **Write the updated HWPX.** Open the original `renewal_playbook.hwpx` as a ZIP for reading and create `/root/renewal_playbook_updated.hwpx` as a new ZIP for writing. Copy every entry from the original except `Contents/section0.xml`. For `Contents/section0.xml`, write the serialized modified XML (use `xml.etree.ElementTree.tostring` with `xml_declaration=True, encoding='utf-8'`). Preserve compression type.

4. **Validate the output:**
   - Open `/root/renewal_playbook_updated.hwpx` as a ZIP and list its contents (should match the original).
   - Extract and print the updated `Contents/section0.xml` to confirm:
     - All six fields are updated to new values.
     - Old values do not appear.
     - Follow-up lines match CSV rows in sequence order.
     - Appendix sentence is intact.
     - No `<hp:lineSegArray>` elements remain on modified paragraphs.

Key constraints:
- Do NOT add duplicate lines; remove old values before inserting new ones.
- Do NOT alter the appendix sentence.
- The output must be a valid ZIP (HWPX package) with the same internal structure.
- Strip `<hp:lineSegArray>` from any paragraph you modify to avoid layout cache issues.

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