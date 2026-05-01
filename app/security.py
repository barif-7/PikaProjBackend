"""
Security utilities for the Pika backend.

Covers two distinct concerns:

1. SSRF protection for user-supplied Ollama endpoint URLs
   Users can configure a custom Ollama host via PUT /provider-connections/ollama.
   Without validation the backend would act as an SSRF proxy — letting an
   attacker reach internal services, GCP metadata endpoints, or Cloud Run
   service accounts.

2. HMAC-signed OAuth state parameter
   The `state` param in the Google OAuth flow carries the mobile deep-link
   callback URL.  Without signing, an attacker can modify the base64-encoded
   state to redirect the auth code to an attacker-controlled URL.  We sign
   the state with HMAC-SHA256 so any tampering is detected before the
   mobile_callback is used.
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
from typing import Optional
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------

# IP ranges that must never be reachable through a user-supplied endpoint.
_BLOCKED_IP_RANGES: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = [
    ipaddress.ip_network("127.0.0.0/8"),      # IPv4 loopback
    ipaddress.ip_network("10.0.0.0/8"),        # RFC 1918 private
    ipaddress.ip_network("172.16.0.0/12"),     # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"),    # RFC 1918 private
    ipaddress.ip_network("169.254.0.0/16"),    # link-local / GCP+AWS metadata
    ipaddress.ip_network("100.64.0.0/10"),     # CGNAT shared address space
    ipaddress.ip_network("0.0.0.0/8"),         # "this" network
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),          # IPv6 ULA (private)
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]

# Well-known dangerous hostnames that should be blocked even before DNS.
_BLOCKED_HOSTNAMES: frozenset[str] = frozenset(
    {
        "localhost",
        "metadata.google.internal",   # GCP metadata server hostname
        "metadata.internal",
        "169.254.169.254",            # GCP/AWS/Azure metadata IP (also caught by range check)
        "fd00:ec2::254",              # AWS metadata IPv6
    }
)

# Maximum URL length we'll accept from a client.
_MAX_URL_LENGTH = 2048


class SSRFError(Exception):
    """Raised when a user-supplied URL targets a forbidden destination."""


def validate_ollama_endpoint(url: str, allowlist: Optional[list[str]] = None) -> str:
    """
    Validate a user-supplied Ollama endpoint URL for SSRF safety.

    Checks applied (in order):
    1. URL length cap
    2. Scheme must be http or https
    3. Hostname must be present and not in the blocked hostname set
    4. If the hostname is a raw IP address, it must not fall in a blocked range
    5. If ``allowlist`` is provided, the URL must start with one of the allowed
       prefixes — this is the strongest control and should be used in production

    Returns the normalised URL (trailing slash stripped) on success.
    Raises ``SSRFError`` on any violation.

    Note: This check cannot fully defend against DNS rebinding attacks
    (where a hostname resolves to a safe IP at validation time but a private
    IP at request time).  For production, combine with VPC egress controls or
    an explicit allowlist.
    """
    if len(url) > _MAX_URL_LENGTH:
        raise SSRFError(f"Endpoint URL exceeds maximum length of {_MAX_URL_LENGTH} characters.")

    try:
        parsed = urlparse(url)
    except Exception as exc:
        raise SSRFError("Malformed endpoint URL.") from exc

    if parsed.scheme not in ("http", "https"):
        raise SSRFError("Endpoint URL must use the http or https scheme.")

    hostname = (parsed.hostname or "").lower().strip()
    if not hostname:
        raise SSRFError("Endpoint URL must include a hostname.")

    # Block by hostname string before attempting IP parsing.
    if hostname in _BLOCKED_HOSTNAMES:
        raise SSRFError(f"Endpoint hostname '{hostname}' is not permitted.")

    # If the hostname is a raw IP address, check against blocked ranges.
    # We separate parsing from range-checking so that SSRFError (which no
    # longer inherits from ValueError) is never accidentally swallowed by the
    # except clause.
    try:
        ip: Optional[ipaddress.IPv4Address | ipaddress.IPv6Address] = ipaddress.ip_address(hostname)
    except ValueError:
        # Not an IP address — a regular DNS hostname.  We cannot rule out DNS
        # rebinding here, but we've blocked the known dangerous names above.
        ip = None

    if ip is not None:
        for blocked in _BLOCKED_IP_RANGES:
            try:
                in_range = ip in blocked
            except TypeError:
                # ip and blocked are different IP address versions — skip.
                continue
            if in_range:
                raise SSRFError(
                    f"Endpoint IP '{hostname}' falls within a blocked address range "
                    f"({blocked}).  Only publicly routable addresses are permitted."
                )

    # Allowlist enforcement — strongest control.  If configured, the URL must
    # start with one of the allowed prefixes (scheme+host, optionally +path).
    if allowlist:
        normalised = url.rstrip("/")
        if not any(_url_matches_allowed_prefix(normalised, prefix) for prefix in allowlist):
            raise SSRFError(
                "Endpoint URL is not in the configured allowlist.  "
                "Contact your administrator to add a new Ollama endpoint."
            )

    return url.rstrip("/")


def parse_ollama_allowlist(raw: str) -> list[str]:
    """
    Parse the ``OLLAMA_ENDPOINT_ALLOWLIST`` env var value.

    Expected format: comma-separated URL prefixes, e.g.:
        https://ollama.example.com,https://ollama2.example.com
    """
    return [entry.strip().rstrip("/") for entry in raw.split(",") if entry.strip()]


def _url_matches_allowed_prefix(url: str, allowed_prefix: str) -> bool:
    """
    Compare a candidate URL against an allowlist entry safely.

    A raw ``startswith`` check is insufficient because an attacker could use a
    host like ``https://ollama.example.com.evil.tld``.  This matcher compares
    scheme + hostname + port, then enforces that the candidate path is either
    identical to the allowlisted path or nested beneath it.
    """
    candidate = urlparse(url.rstrip("/"))
    allowed = urlparse(allowed_prefix.rstrip("/"))

    if candidate.scheme.lower() != allowed.scheme.lower():
        return False
    if (candidate.hostname or "").lower() != (allowed.hostname or "").lower():
        return False
    if candidate.port != allowed.port:
        return False

    allowed_path = allowed.path.rstrip("/")
    candidate_path = candidate.path.rstrip("/")
    if not allowed_path:
        return True
    if candidate_path == allowed_path:
        return True
    return candidate_path.startswith(f"{allowed_path}/")


# ---------------------------------------------------------------------------
# HMAC-signed OAuth state
# ---------------------------------------------------------------------------

_STATE_SEPARATOR = "."

# Minimum secret entropy in bytes.  Warn (but do not crash) if the configured
# secret is shorter — an attacker with knowledge of a weak secret could forge
# state parameters.
_MIN_SECRET_BYTES = 16


def sign_oauth_state(payload_b64: str, secret: str) -> str:
    """
    Append an HMAC-SHA256 signature to a base64-encoded state payload.

    Returns ``"{payload_b64}.{hex_signature}"``.

    The payload must already be base64url-encoded (no padding).  This function
    only appends the MAC; callers are responsible for encoding.
    """
    mac = _compute_hmac(payload_b64, secret)
    return f"{payload_b64}{_STATE_SEPARATOR}{mac}"


def verify_and_extract_state(signed_state: str, secret: str) -> str:
    """
    Verify the HMAC signature and return the raw base64 payload.

    Raises ``ValueError`` if the state is malformed or the signature does not
    match — callers should treat this as an invalid/tampered request.
    """
    parts = signed_state.rsplit(_STATE_SEPARATOR, 1)
    if len(parts) != 2:
        raise ValueError("OAuth state is malformed (missing signature separator).")

    payload_b64, provided_mac = parts
    if not payload_b64 or not provided_mac:
        raise ValueError("OAuth state is malformed (empty component).")

    expected_mac = _compute_hmac(payload_b64, secret)
    if not hmac.compare_digest(expected_mac, provided_mac):
        raise ValueError("OAuth state signature is invalid.  The request may have been tampered with.")

    return payload_b64


def generate_fallback_secret() -> str:
    """
    Generate a random per-process secret for use when OAUTH_STATE_SECRET is
    not configured.

    Per-process means the secret changes on every container restart, which
    invalidates any in-flight OAuth flows on restart.  This is acceptable for
    development but NOT suitable for production — set OAUTH_STATE_SECRET.
    """
    return secrets.token_hex(32)


def _compute_hmac(message: str, secret: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
