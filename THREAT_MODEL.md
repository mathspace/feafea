# feafea threat model

## Overview

Feafea is an in-process Python feature-flag compiler and evaluator. Dictionary configuration is validated against a JSON schema, filters are parsed and compiled into Python functions, and evaluation combines a compiled configuration with caller attributes. Compiled configurations can be serialized and loaded through dill. The evaluator validates attribute types, checks compatibility using an assertion that is removed by python -O or PYTHONOPTIMIZE, returns detailed decisions, records Prometheus metrics, and optionally retains detailed evaluations until popped. It does not provide a hosted flag service, user authentication endpoint, or persistence transport.

| Component | Source |
| --- | --- |
| Dictionary validation and compilation | src/feafea/__init__.py:748 |
| Serialized configuration loading | src/feafea/__init__.py:739 |
| Attribute validation and evaluation | src/feafea/__init__.py:993 |
| Recording and retrieval | src/feafea/__init__.py:1052 |

| Deployment or workflow | Resource or capability | Configuration and precedence | Safe effective value or location | Readers, writers, or recipients | Enforcing control | Evidence or unknowns |
| --- | --- | --- | --- | --- | --- | --- |
| Dictionary compile | Generated Python functions | from_dict validates JSON schema then parses filters and compiles generated lines | Callable rules held in CompiledConfig memory | Trusted configuration publisher and host interpreter | JSON schema, attribute comparison type guards and cycle checks; undefined filter/flag/rule references become False, and the flag-reference type-validation block performs no checks; filter parsing is not a safe untrusted-input boundary | src/feafea/__init__.py:755; src/feafea/__init__.py:486; src/feafea/__init__.py:562; src/feafea/__init__.py:596; src/feafea/__init__.py:890 |
| Binary load | Python deserialization | Caller passes bytes directly to dill.loads | Loaded object in host interpreter; type assertion occurs afterward | Byte supplier and host process | Caller-owned provenance/integrity; post-load assertion is not a sandbox | src/feafea/__init__.py:740 |
| Evaluation | Caller attributes and metrics | Validated dict merged after default __now; metric labels from decision | Attributes and decision in memory; Prometheus metric state | Library caller and any separately configured metrics exporter | Attribute type validation and evaluation lock where recorded | src/feafea/__init__.py:1024 |
| Optional recording | Detailed evaluation retention | Evaluator(record_evaluations=True), then pop_evaluations | In-process list including evaluation attributes until drained | Host process and caller consuming popped records | Opt-in storage; caller controls retention/export | src/feafea/__init__.py:987 |

## Threat Model, Trust Boundaries, and Assumptions

Protect host interpreter authority, trustworthy flag decisions, caller-attribute confidentiality, and predictable resource consumption. Configuration authors and evaluation callers have distinct roles: a user attribute should influence only configured decisions, while compiled byte loaders consume executable serialization. A checksum comparison is compatibility checking, not origin authentication. A caller must establish provenance before using from_bytes. Flag evaluation is not a substitute for application authorization; allowing end users to choose identity or entitlement attributes can defeat a consuming application’s own policy without breaching the library. No tenant model or remote attacker reachability is inferred.

This model uses the repository’s own source and generic host/caller obligations. No private deployment facts, observed exploitation, or inferred tenant relationships are included. Source review establishes the operations below; it does not establish every dependency’s implementation or the permissions of an actual installation. The operator must distinguish a deliberately granted capability from a lower-trust input gaining a new one.

The origin, distribution channel, signatures, retention policy and exporters are responsibilities of each caller. from_bytes type assertion cannot authenticate bytes before dill executes; compatibility checksum checks likewise do not establish trusted provenance. Both the post-load type assertion and compatibility assertion disappear under optimized Python, so they are not reliable enforcement across supported interpreter modes (src/feafea/__init__.py:741; src/feafea/__init__.py:1024).

## Attack Surface, Mitigations, and Attacker Stories

The set-literal parser escape below is established by static source analysis; other stories remain investigation hypotheses. No application code or exploit input was executed, and deployment reachability is unverified. Priority reflects plausible capability gain; each prerequisite must hold before assigning a deployment-specific severity.

| Priority | Scenario and capability gain | Prerequisites | Impact | Existing controls | Mitigation | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | An attacker supplies compiled bytes to a consumer that treats from_bytes as a safe data parser. | Less-trusted bytes cross into host with greater interpreter authority. | Arbitrary code execution permitted by executable deserialization. | Type assertion follows dill.loads; no pre-load authenticity control is provided. | Accept serialized bytes only from authenticated trusted producers; dictionary filters also require trusted authors until executable set-literal parsing is removed. | src/feafea/__init__.py:739 |
| P1 | An attacker-controlled dictionary filter escapes set-literal parsing into Python execution. | Lower-trust filter text is compiled in a more privileged host. | Host code execution: the SET regex treats a backslash-escaped quote as a terminator while Python eval treats it as part of a string, permitting an executable suffix. This parser path is statically established, before generated functions run. | JSON schema and the regex do not preserve Python string-literal boundaries; later validation cannot undo execution in eval. | Reject untrusted filters until eval is replaced by non-executable literal parsing and explicit allowed-type checks. | src/feafea/__init__.py:110; src/feafea/__init__.py:180; src/feafea/__init__.py:418; src/feafea/__init__.py:812; src/feafea/__init__.py:844 |
| P2 | A consumer accepts end-user entitlement attributes as authoritative input, or trusts caller-supplied __now in evaluation records. | Application delegates security-sensitive choices to attacker attributes, or consumes returned/retained timestamps as authoritative records. | Entitlement inputs can change decisions; __now specifically forges evaluation timestamps, not filter decisions: attribute grammar forbids underscore-leading names. | Attribute types are checked; identity and timestamp provenance remain caller responsibilities. | Construct authoritative decision attributes server-side and prevent callers overriding internal timestamp fields. | src/feafea/__init__.py:116; src/feafea/__init__.py:1026; src/feafea/__init__.py:1039 |
| P2 | Large evaluation attributes, retained evaluations or large configurations exhaust resources; retained attributes may also reach unintended export readers. | Unbounded lower-trust attribute counts, strings or sets; alternatively unbounded configuration or recording without draining. Disclosure separately requires sensitive attributes and an unauthorized recipient. | CPU/memory exhaustion can occur with recording disabled and a small trusted configuration: validation traverses sets and insplit can hash every element and its full string value. | Attribute type checks impose no size limits; recording is opt-in and pop is synchronized. | Bound attribute count, string length, set cardinality, request rate, configuration and recording volume; filter attributes before external export. | src/feafea/__init__.py:553; src/feafea/__init__.py:997; src/feafea/__init__.py:1038 |

## Severity Calibration (Critical, High, Medium, Low)

Critical requires privileged deserialization with a demonstrated broad infrastructure consequence; the library alone supplies no such deployment.

High fits untrusted compiled-byte loading or a proven compiler escape in a privileged host.

Medium fits bounded denial of service or sensitive evaluation retention with real unauthorized readers.

Low fits rejected malformed attributes or a bad feature choice without access-control consequences. exec of validated generated functions is not by itself a demonstrated injection.

Confidence in the source-described data flow is separate from confidence in exploitability. A finding requires a concrete lower-trust entry, an effective control failure, and a consequential new capability. Host compromise assumed at the outset, deliberate operator authority, and self-only errors do not supply that missing evidence.

Repository: github.com/mathspace/feafea

Version: 2322cfe515339ee25e4f60a026903639c7d2711d
