//! Immutable, secret-free declaration identity for crash-resume decisions.
//!
//! A matching contract detects local declaration/config drift. It is neither an
//! authenticated provider identity nor proof that a declaration is idempotent.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

pub(crate) const TOOL_RESUME_CONTRACT_VERSION: u8 = 1;
const MAX_TOOL_NAME: usize = 256;
const MAX_DESCRIPTOR_DEPTH: usize = 32;
const MAX_DESCRIPTOR_NODES: usize = 16_384;
const MAX_DESCRIPTOR_BYTES: usize = 1_048_576;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ToolResumeContract {
    pub version: u8,
    pub name: String,
    pub idempotent: bool,
    pub fingerprint: String,
}

impl ToolResumeContract {
    pub(crate) fn new(name: &str, idempotent: bool, descriptor: &Value) -> Option<Self> {
        if name.is_empty() || name.len() > MAX_TOOL_NAME {
            return None;
        }
        Some(Self {
            version: TOOL_RESUME_CONTRACT_VERSION,
            name: name.to_string(),
            idempotent,
            fingerprint: sha256_hex(&canonical_json(descriptor)?),
        })
    }

    pub(crate) fn is_valid(&self) -> bool {
        self.version == TOOL_RESUME_CONTRACT_VERSION
            && !self.name.is_empty()
            && self.name.len() <= MAX_TOOL_NAME
            && self.fingerprint.len() == 64
            && self
                .fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    }
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(crate) fn json_sha256(value: &Value) -> Option<String> {
    Some(sha256_hex(&canonical_json(value)?))
}

fn canonical_json(value: &Value) -> Option<Vec<u8>> {
    // Stream escaping/serialization into the bounded buffer. Serializing to a
    // temporary String first would allocate the oversized value before rejecting it.
    fn append_json(out: &mut Vec<u8>, value: &impl Serialize) -> Option<()> {
        struct LimitedOutput<'a>(&'a mut Vec<u8>);
        impl std::io::Write for LimitedOutput<'_> {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                if bytes.len() > MAX_DESCRIPTOR_BYTES.saturating_sub(self.0.len()) {
                    return Err(std::io::Error::other("resume descriptor limit"));
                }
                self.0.extend_from_slice(bytes);
                Ok(bytes.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        serde_json::to_writer(LimitedOutput(out), value).ok()
    }
    fn append(out: &mut Vec<u8>, bytes: &[u8]) -> Option<()> {
        let next = out.len().checked_add(bytes.len())?;
        if next > MAX_DESCRIPTOR_BYTES {
            return None;
        }
        out.extend_from_slice(bytes);
        Some(())
    }

    fn write(value: &Value, depth: usize, nodes: &mut usize, out: &mut Vec<u8>) -> Option<()> {
        if depth > MAX_DESCRIPTOR_DEPTH || *nodes >= MAX_DESCRIPTOR_NODES {
            return None;
        }
        *nodes += 1;
        match value {
            Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                append_json(out, value)?;
            }
            Value::Array(values) => {
                if values.len() > MAX_DESCRIPTOR_NODES.saturating_sub(*nodes) {
                    return None;
                }
                append(out, b"[")?;
                for (index, value) in values.iter().enumerate() {
                    if index > 0 {
                        append(out, b",")?;
                    }
                    write(value, depth.checked_add(1)?, nodes, out)?;
                }
                append(out, b"]")?;
            }
            Value::Object(values) => {
                if values.len() > MAX_DESCRIPTOR_NODES.saturating_sub(*nodes) {
                    return None;
                }
                append(out, b"{")?;
                let mut keys = values.keys().collect::<Vec<_>>();
                keys.sort_unstable();
                for (index, key) in keys.into_iter().enumerate() {
                    if index > 0 {
                        append(out, b",")?;
                    }
                    append_json(out, key)?;
                    append(out, b":")?;
                    write(values.get(key)?, depth.checked_add(1)?, nodes, out)?;
                }
                append(out, b"}")?;
            }
        }
        Some(())
    }

    let mut out = Vec::new();
    let mut nodes = 0;
    write(value, 0, &mut nodes, &mut out)?;
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::ToolResumeContract;
    use serde_json::{Value, json};

    #[test]
    fn fingerprint_is_deterministic_across_object_key_order() {
        let first = ToolResumeContract::new("tool", true, &json!({"b": 2, "a": 1})).unwrap();
        let second = ToolResumeContract::new("tool", true, &json!({"a": 1, "b": 2})).unwrap();
        assert_eq!(first, second);
        assert!(first.is_valid());
    }

    #[test]
    fn validation_rejects_unknown_version_and_invalid_digest() {
        let mut contract = ToolResumeContract::new("tool", true, &json!({})).unwrap();
        contract.version = 2;
        assert!(!contract.is_valid());
        contract.version = 1;
        contract.fingerprint = "A".repeat(64);
        assert!(!contract.is_valid());
    }

    #[test]
    fn descriptor_limits_fail_closed_before_unbounded_serialization() {
        let mut deep = json!(null);
        for _ in 0..=32 {
            deep = json!([deep]);
        }
        assert!(ToolResumeContract::new("tool", true, &deep).is_none());

        let many_nodes = Value::Array((0..16_385).map(|_| json!(0)).collect());
        assert!(ToolResumeContract::new("tool", true, &many_nodes).is_none());

        let oversized = json!({"value": "x".repeat(1_048_576)});
        assert!(ToolResumeContract::new("tool", true, &oversized).is_none());
    }
}
