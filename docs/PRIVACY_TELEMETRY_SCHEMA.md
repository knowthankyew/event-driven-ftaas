# Shared Privacy & Telemetry Configuration Schema
> Standardized configuration schema across the `knowthankyew` application portfolio (`bill-of-rights-bot`, `careCheck`, `gradcast`, `lease-audit`, `mailStripper`, `ml` / FTaaS, `paystub-check`, `warranty-watch`).

---

## 1. Design Principles

Every application in the portfolio conforms to the **Consumer-Safe, Zero-Fork Enterprise Overlay** standard:
1. **Consumer Default (Safe & Burnable):** Runs with zero network egress (`deny`), in-memory telemetry (`memory_only`), zero cloud retention, and complete memory wipe upon triggering "Burn Local Data".
2. **Enterprise Overlay (Opt-In Observability):** An enterprise operator can attach standard OpenTelemetry collectors (`otlp`) and configure audit retention without modifying source code or maintaining a separate fork.
3. **Strict Allowlist Invariant:** To prevent short clause leaks or sensitive snippets from slipping through under unanticipated key names (e.g., `matchedText`, `snippet`, `excerpt`, `clause_text`), attribute sanitizers operate on a **strict allowlist**, not a denylist. Only explicitly registered metadata keys (`rule_id`, `rule_ids`, `clause_count`, `flagged_count`, `duration_ms`, `job_id`, `status`, `jurisdiction`, `device`, `adapter_size_bytes`, `dataset_hash`) are permitted. All unrecognized or arbitrary keys are automatically redacted as `[REDACTED_NOT_IN_ALLOWLIST]`.

---

## 2. Configuration Schema & Environment Mapping

| Configuration Key | Type | Default Value | Environment Variable | Description |
| :--- | :--- | :--- | :--- | :--- |
| `privacy.burn_enabled` | `boolean` | `true` | `KTY_PRIVACY_BURN_ENABLED` / `VITE_PRIVACY_BURN_ENABLED` | When `true`, burn purges all session buffers, memory, and telemetry. When `false`, UI resets but telemetry exporter pipeline remains attached. |
| `privacy.network_egress` | `string` | `"deny"` | `KTY_PRIVACY_NETWORK_EGRESS` | Network egress policy (`deny` \| `allow_otlp` \| `allow_all`). Enforced via Content Security Policy (CSP) in web apps and network clients in backend services. |
| `telemetry.mode` | `string` | `"memory_only"` | `KTY_TELEMETRY_MODE` / `VITE_TELEMETRY_MODE` | Active telemetry strategy (`memory_only` \| `otlp` \| `disabled`). Automatically escalates from `memory_only` to `otlp` when an endpoint is configured. |
| `telemetry.otlp_endpoint` | `string?` | `null` | `OTEL_EXPORTER_OTLP_ENDPOINT` / `VITE_OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP gRPC/HTTP endpoint URL (e.g. `http://otel-collector:4318/v1/traces`). |
| `telemetry.allow_raw_payloads` | `boolean` | `false` | `KTY_TELEMETRY_ALLOW_RAW_PAYLOADS` | Security override permitting raw document bodies in telemetry. **Default `false` must never be bypassed in consumer builds.** |
| `audit.durable` | `boolean` | `false` | `KTY_AUDIT_DURABLE` | When `false`, audit events live solely in memory buffer and wipe upon burn. When `true`, enterprise persists to durable audit log. |

---

## 3. Configuration Formats

### YAML Format (`kty-config.yaml`)

```yaml
version: "1.0"
privacy:
  burn_enabled: true
  network_egress: "deny"

telemetry:
  mode: "memory_only" # "disabled" | "memory_only" | "otlp"
  otlp_endpoint: null
  allow_raw_payloads: false

audit:
  durable: false
```

### JSON Format (`kty-config.json`)

```json
{
  "$schema": "https://knowthankyew.org/schemas/privacy-telemetry-v1.json",
  "version": "1.0",
  "privacy": {
    "burn_enabled": true,
    "network_egress": "deny"
  },
  "telemetry": {
    "mode": "memory_only",
    "otlp_endpoint": null,
    "allow_raw_payloads": false
  },
  "audit": {
    "durable": false
  }
}
```

---

## 4. TypeScript Interface Definition

```typescript
export type TelemetryMode = 'disabled' | 'memory_only' | 'otlp';
export type EgressPolicy = 'deny' | 'allow_otlp' | 'allow_all';

export interface AppPrivacyConfig {
  privacy: {
    burnEnabled: boolean;
    networkEgress: EgressPolicy;
  };
  telemetry: {
    mode: TelemetryMode;
    otlpEndpoint?: string | null;
    allowRawPayloads: boolean;
  };
  audit: {
    durable: boolean;
  };
}

export const DEFAULT_PRIVACY_CONFIG: AppPrivacyConfig = {
  privacy: {
    burnEnabled: true,
    networkEgress: 'deny',
  },
  telemetry: {
    mode: 'memory_only',
    otlpEndpoint: null,
    allowRawPayloads: false,
  },
  audit: {
    durable: false,
  },
};
```

---

## 5. .NET Options Class Definition

```csharp
namespace KnowThankYew.Common.Configuration;

public sealed class PrivacyTelemetryOptions
{
    public const string SectionName = "PrivacyTelemetry";

    public PrivacyOptions Privacy { get; set; } = new();
    public TelemetryOptions Telemetry { get; set; } = new();
    public AuditOptions Audit { get; set; } = new();
}

public sealed class PrivacyOptions
{
    public bool BurnEnabled { get; set; } = true;
    public string NetworkEgress { get; set; } = "deny";
}

public sealed class TelemetryOptions
{
    public string Mode { get; set; } = "memory_only"; // "disabled" | "memory_only" | "otlp"
    public string? OtlpEndpoint { get; set; }
    public bool AllowRawPayloads { get; set; } = false;
}

public sealed class AuditOptions
{
    public bool Durable { get; set; } = false;
}
```

---

## 6. JSON Schema Specification

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "KnowThankYewPrivacyTelemetryConfig",
  "type": "object",
  "properties": {
    "version": { "type": "string", "enum": ["1.0"] },
    "privacy": {
      "type": "object",
      "properties": {
        "burn_enabled": { "type": "boolean", "default": true },
        "network_egress": { "type": "string", "enum": ["deny", "allow_otlp", "allow_all"], "default": "deny" }
      },
      "required": ["burn_enabled", "network_egress"]
    },
    "telemetry": {
      "type": "object",
      "properties": {
        "mode": { "type": "string", "enum": ["disabled", "memory_only", "otlp"], "default": "memory_only" },
        "otlp_endpoint": { "type": ["string", "null"], "format": "uri", "default": null },
        "allow_raw_payloads": { "type": "boolean", "default": false }
      },
      "required": ["mode", "allow_raw_payloads"]
    },
    "audit": {
      "type": "object",
      "properties": {
        "durable": { "type": "boolean", "default": false }
      },
      "required": ["durable"]
    }
  },
  "required": ["version", "privacy", "telemetry", "audit"]
}
```

---

## 7. Single Source of Truth for UI Privacy Claims & Honest Indicators

To prevent false privacy claims sitting adjacent to honest telemetry indicators:

1. **Unified Privacy Claims Function:**
   All client-side applications must derive user-facing claims from a single helper (`getPrivacyClaims(report)`).
   - **Consumer Mode (`isLocalOnlyHonest === true`):** Asserts `100% Client-Side Local Execution • Zero Network Transmission`.
   - **Enterprise Mode (`isLocalOnlyHonest === false`):** Must dynamically adjust all UI copy (dropzones, disclaimers, headers, footers) to reflect operational metadata export while emphasizing that document text is redacted via allowlist.
2. **Prominent Enterprise Differentiation:**
   When built with an external exporter (`otlp`), the UI must render an unmistakable persistent top banner (e.g. `enterprise-persistent-banner`) and an `[ENTERPRISE]` title badge so that non-default builds cannot be mistaken for default consumer builds at a glance.
3. **`NOTICE.md` Compliance:**
   Each repository must maintain a root `NOTICE.md` detailing statutory disclaimers, consumer privacy air-gap invariants, enterprise observability disclosures, and third-party software attribution aligned with the CycloneDX SBOM.

