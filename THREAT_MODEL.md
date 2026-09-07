# feafea threat model

## Overview

Feafea is an in-process Python feature-flag compiler and evaluator. Dictionary configuration is validated against a JSON schema, filters are parsed and compiled into Python functions, and evaluation combines a compiled configuration with caller attributes. Compiled configurations can be serialized and loaded through dill. The evaluator validates attribute types, checks compatibility, returns detailed decisions, records Prometheus metrics, and optionally retains detailed evaluations until popped. It does not provide a hosted flag service, user authentication endpoint, or persistence transport.

| Component | Source |
| --- | --- |
| Dictionary validation and compilation | src/feafea/__init__.py:748 |
| Serialized configuration loading | src/feafea/__init__.py:739 |
| Attribute validation and evaluation | src/feafea/__init__.py:993 |
| Recording and retrieval | src/feafea/__init__.py:1052 |

| Deployment or workflow | Resource or capability | Configuration and precedence | Safe effective value or location | Readers, writers, or recipients | Enforcing control | Evidence or unknowns |
| --- | --- | --- | --- | --- | --- | --- |
| Dictionary compile | Generated Python functions | from_dict validates JSON schema then parses filters and compiles generated lines | Callable rules held in CompiledConfig memory | Trusted configuration publisher and host interpreter | Schema, parser, reference/type/cycle checks; code generation must preserve literal boundaries | src/feafea/__init__.py:755 |
| Binary load | Python deserialization | Caller passes bytes directly to dill.loads | Loaded object in host interpreter; type assertion occurs afterward | Byte supplier and host process | Caller-owned provenance/integrity; post-load assertion is not a sandbox | src/feafea/__init__.py:740 |
| Evaluation | Caller attributes and metrics | Validated dict merged after default __now; metric labels from decision | Attributes and decision in memory; Prometheus metric state | Library caller and any separately configured metrics exporter | Attribute type validation and evaluation lock where recorded | src/feafea/__init__.py:1024 |
| Optional recording | Detailed evaluation retention | Evaluator(record_evaluations=True), then pop_evaluations | In-process list including evaluation attributes until drained | Host process and caller consuming popped records | Opt-in storage; caller controls retention/export | src/feafea/__init__.py:987 |

## Threat Model, Trust Boundaries, and Assumptions

Protect host interpreter authority, trustworthy flag decisions, caller-attribute confidentiality, and predictable resource consumption. Configuration authors and evaluation callers have distinct roles: a user attribute should influence only configured decisions, while compiled byte loaders consume executable serialization. A checksum comparison is compatibility checking, not origin authentication. A caller must establish provenance before using from_bytes. Flag evaluation is not a substitute for application authorization; allowing end users to choose identity or entitlement attributes can defeat a consuming application’s own policy without breaching the library. No tenant model or remote attacker reachability is inferred.

This model uses the repository’s own source and generic host/caller obligations. No private deployment facts, observed exploitation, or inferred tenant relationships are included. Source review establishes the operations below; it does not establish every dependency’s implementation or the permissions of an actual installation. The operator must distinguish a deliberately granted capability from a lower-trust input gaining a new one.

The origin, distribution channel, signatures, retention policy and exporters are responsibilities of each caller. from_bytes type assertion cannot authenticate bytes before dill executes; compatibility checksum checks likewise do not establish trusted provenance.

## Attack Surface, Mitigations, and Attacker Stories

These are prioritized hypotheses for investigation, not validated findings. Priority reflects plausible capability gain; each prerequisite must hold before assigning a deployment-specific severity.

| Priority | Scenario and capability gain | Prerequisites | Impact | Existing controls | Mitigation | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | An attacker supplies compiled bytes to a consumer that treats from_bytes as a safe data parser. | Less-trusted bytes cross into host with greater interpreter authority. | Arbitrary code execution permitted by executable deserialization. | Type assertion follows dill.loads; no pre-load authenticity control is provided. | Accept serialized bytes only from authenticated trusted producers; use constrained dictionary input for untrusted configuration. | src/feafea/__init__.py:739 |
| P1 conditional | An attacker-controlled configuration string breaks out of generated literal structure. | An actual parser/code-generation escape must exist; not established merely by exec calls. | Code execution in the compiler host. | JSON schema and dedicated filter parsing, reference and cycle checks. | Validate every generated literal uses safe representation and test any candidate against the real grammar. | src/feafea/__init__.py:853 |
| P2 | A consumer accepts end-user entitlement or __now attributes as authoritative input. | Application delegates security-sensitive choices to attacker-supplied attributes. | Incorrect flag decision with downstream privilege consequences. | Attribute types are checked; identity and time provenance are caller responsibilities. | Construct authoritative attributes server-side and reserve internal names. | src/feafea/__init__.py:1026 |
| P2 | Retained evaluations or very large configurations exhaust memory or expose sensitive attributes through exports. | Recording enabled without draining, or unbounded lower-trust configuration; sensitive caller attributes. | Host availability loss or conditional data disclosure. | Recording opt-in; pop is synchronized and metrics label selected decision fields. | Bound configuration and recording volumes; filter attributes before any external export. | src/feafea/__init__.py:1038 |

## Severity Calibration (Critical, High, Medium, Low)

Critical requires privileged deserialization with a demonstrated broad infrastructure consequence; the library alone supplies no such deployment.

High fits untrusted compiled-byte loading or a proven compiler escape in a privileged host.

Medium fits bounded denial of service or sensitive evaluation retention with real unauthorized readers.

Low fits rejected malformed attributes or a bad feature choice without access-control consequences. exec of validated generated functions is not by itself a demonstrated injection.

Confidence in the source-described data flow is separate from confidence in exploitability. A finding requires a concrete lower-trust entry, an effective control failure, and a consequential new capability. Host compromise assumed at the outset, deliberate operator authority, and self-only errors do not supply that missing evidence.

Repository: github.com/mathspace/feafea

Version: 2322cfe515339ee25e4f60a026903639c7d2711d
