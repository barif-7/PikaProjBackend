from __future__ import annotations

import base64
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Union
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

from .config import Settings
from .security import sign_oauth_state, verify_and_extract_state


GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
_STORE_LOCK = Lock()


class AuthConfigurationError(RuntimeError):
    pass


class AuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthStore:
    root_dir: Path
    session_ttl_seconds: float

    def __post_init__(self) -> None:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._users_path.touch(exist_ok=True)
        self._sessions_path.touch(exist_ok=True)
        self._provider_connections_path.touch(exist_ok=True)

    @property
    def _users_path(self) -> Path:
        return self.root_dir / "users.json"

    @property
    def _sessions_path(self) -> Path:
        return self.root_dir / "sessions.json"

    @property
    def _provider_connections_path(self) -> Path:
        return self.root_dir / "provider-connections.json"

    def create_google_session(
        self,
        *,
        google_sub: str,
        email: str,
        display_name: str,
        photo_url: Optional[str] = None,
    ) -> dict[str, Any]:
        user_id = f"google:{google_sub}"
        created_at = _utc_now_iso()
        expires_at = _utc_after_seconds_iso(self.session_ttl_seconds)
        session_token = secrets.token_urlsafe(32)

        with _STORE_LOCK:
            users = _load_json_map(self._users_path)
            sessions = _load_json_map(self._sessions_path)

            existing_user = users.get(user_id, {})
            users[user_id] = {
                "user_id": user_id,
                "provider": "google",
                "google_sub": google_sub,
                "email": email,
                "display_name": display_name,
                "photo_url": photo_url,
                "created_at": existing_user.get("created_at", created_at),
                "updated_at": created_at,
            }
            sessions[session_token] = {
                "session_token": session_token,
                "user_id": user_id,
                "created_at": created_at,
                "expires_at": expires_at,
            }

            _save_json_map(self._users_path, users)
            _save_json_map(self._sessions_path, sessions)

        return self.session_response(session_token)

    def session_response(self, session_token: str) -> dict[str, Any]:
        with _STORE_LOCK:
            users = _load_json_map(self._users_path)
            sessions = _load_json_map(self._sessions_path)
            session = sessions.get(session_token)
            if not session:
                raise AuthError("Unknown session.")
            if _is_expired(session.get("expires_at")):
                sessions.pop(session_token, None)
                _save_json_map(self._sessions_path, sessions)
                raise AuthError("Session expired.")

            session["expires_at"] = _utc_after_seconds_iso(self.session_ttl_seconds)
            sessions[session_token] = session
            _save_json_map(self._sessions_path, sessions)

            user = users.get(session["user_id"])
            if not user:
                raise AuthError("Session user is missing.")

            return {
                "sessionToken": session_token,
                "expiresAt": session.get("expires_at"),
                "user": {
                    "userId": user["user_id"],
                    "email": user["email"],
                    "displayName": user["display_name"],
                    "photoURL": user.get("photo_url"),
                },
            }

    def refresh_session(self, session_token: str) -> dict[str, Any]:
        return self.session_response(session_token)

    def revoke_session(self, session_token: str) -> None:
        with _STORE_LOCK:
            sessions = _load_json_map(self._sessions_path)
            if session_token in sessions:
                sessions.pop(session_token, None)
                _save_json_map(self._sessions_path, sessions)

    def save_ollama_connection(
        self,
        *,
        user_id: str,
        endpoint_url: str,
        model: Optional[str],
        api_token: Optional[str],
        label: Optional[str],
    ) -> dict[str, Any]:
        with _STORE_LOCK:
            connections = _load_json_map(self._provider_connections_path)
            user_connections = connections.get(user_id, {})
            user_connections["ollama"] = {
                "endpoint_url": endpoint_url,
                "model": model,
                "api_token": api_token,
                "label": label,
                "updated_at": _utc_now_iso(),
            }
            connections[user_id] = user_connections
            _save_json_map(self._provider_connections_path, connections)

        return self.ollama_connection_response(user_id)

    def ollama_connection_response(self, user_id: str) -> dict[str, Any]:
        ollama = self.ollama_connection(user_id)
        if not ollama:
            raise AuthError("No Ollama connection configured for this user.")

        return {
            "endpointURL": ollama["endpoint_url"],
            "model": ollama.get("model"),
            "hasAPIToken": bool(ollama.get("api_token")),
            "label": ollama.get("label"),
            "updatedAt": ollama.get("updated_at"),
        }

    def ollama_connection(self, user_id: str) -> Optional[dict[str, Any]]:
        with _STORE_LOCK:
            connections = _load_json_map(self._provider_connections_path)
            ollama = connections.get(user_id, {}).get("ollama")
        return ollama

    def delete_user(self, user_id: str) -> None:
        """
        Delete all auth data for a user: user record, all sessions, and provider connections.

        Idempotent — succeeds even if the user does not exist.
        """
        with _STORE_LOCK:
            users = _load_json_map(self._users_path)
            sessions = _load_json_map(self._sessions_path)
            connections = _load_json_map(self._provider_connections_path)

            users.pop(user_id, None)
            # Remove every session that belongs to this user.
            sessions = {
                token: session
                for token, session in sessions.items()
                if session.get("user_id") != user_id
            }
            connections.pop(user_id, None)

            _save_json_map(self._users_path, users)
            _save_json_map(self._sessions_path, sessions)
            _save_json_map(self._provider_connections_path, connections)


def make_auth_store(settings: Settings) -> "AuthStore | Any":
    """
    Return the appropriate auth store based on PERSISTENCE_BACKEND.

    - "json"      → AuthStore (local JSON files, default, suitable for dev/single-instance)
    - "firestore" → FirestoreAuthStore (Firestore-backed, required for horizontal scaling)
    """
    if settings.persistence_backend == "firestore":
        # Local import to avoid a module-load-time circular dependency.
        from .store_firestore import make_firestore_auth_store
        return make_firestore_auth_store(settings)
    return AuthStore(
        root_dir=Path(settings.auth_data_dir),
        session_ttl_seconds=settings.auth_session_ttl_seconds,
    )


def build_google_authorize_url(settings: Settings, mobile_callback: str) -> str:
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret or not settings.google_oauth_callback_url:
        raise AuthConfigurationError("Google OAuth is not configured on the backend.")

    callback_parsed = urlparse(mobile_callback)
    callback_scheme = callback_parsed.scheme.lower()
    expected_scheme = settings.auth_mobile_callback_scheme.lower()
    if not expected_scheme:
        raise AuthConfigurationError("AUTH_MOBILE_CALLBACK_SCHEME is not configured.")
    if callback_scheme != expected_scheme:
        raise AuthConfigurationError(
            f"The mobile callback scheme '{callback_scheme}' is not allowed.  "
            f"Expected '{expected_scheme}'."
        )
    # Reject obviously dangerous callback schemes even if they somehow match.
    if callback_scheme in ("http", "https", "javascript", "data", "vbscript"):
        raise AuthConfigurationError(
            "The mobile callback must use a custom app URL scheme, not a web URL."
        )

    state_payload = {"nonce": secrets.token_urlsafe(12), "mobile_callback": mobile_callback}
    payload_b64 = _encode_state(state_payload)
    # Sign the state so that the mobile_callback cannot be tampered with in transit.
    signed_state = sign_oauth_state(payload_b64, settings.oauth_state_secret)

    query = urlencode(
        {
            "client_id": settings.google_oauth_client_id,
            "redirect_uri": settings.google_oauth_callback_url,
            "response_type": "code",
            "scope": "openid email profile",
            "access_type": "offline",
            "prompt": "select_account",
            "state": signed_state,
        }
    )
    return f"{GOOGLE_AUTHORIZE_URL}?{query}"


async def exchange_google_code_for_user(settings: Settings, code: str) -> dict[str, Any]:
    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret or not settings.google_oauth_callback_url:
        raise AuthConfigurationError("Google OAuth is not configured on the backend.")

    async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
        token_response = await client.post(
            GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "redirect_uri": settings.google_oauth_callback_url,
                "grant_type": "authorization_code",
            },
        )
        token_response.raise_for_status()
        token_payload = token_response.json()
        access_token = token_payload.get("access_token")
        if not access_token:
            raise AuthError("Google did not return an access token.")

        userinfo_response = await client.get(
            GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        userinfo_response.raise_for_status()
        userinfo = userinfo_response.json()

    google_sub = str(userinfo.get("sub") or "").strip()
    email = str(userinfo.get("email") or "").strip()
    display_name = str(userinfo.get("name") or email or "Google User").strip()
    if not google_sub or not email:
        raise AuthError("Google user info was incomplete.")

    return {
        "google_sub": google_sub,
        "email": email,
        "display_name": display_name,
        "photo_url": userinfo.get("picture"),
    }


def decode_google_state(encoded_state: str, secret: Optional[str] = None) -> dict[str, Any]:
    """
    Verify the HMAC signature and decode the OAuth state payload.

    If ``secret`` is provided the signature is verified and a tampered state
    raises ``AuthError``.  If ``secret`` is None the state is decoded without
    verification (legacy behaviour — only used in tests that do not set a
    secret).
    """
    try:
        if secret:
            payload_b64 = verify_and_extract_state(encoded_state, secret)
        else:
            # Legacy path — no signing.  Accept bare base64 payload.
            payload_b64 = encoded_state
        return json.loads(_urlsafe_b64decode(payload_b64).decode("utf-8"))
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError("Invalid OAuth state.") from exc


def append_query_params(url: str, params: dict[str, str]) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({key: value for key, value in params.items() if value != ""})
    return urlunparse(parsed._replace(query=urlencode(query)))


def extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise AuthError("Missing Authorization header.")

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthError("Expected a Bearer token.")

    return token.strip()


def _load_json_map(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_json_map(path: Path, payload: dict[str, Any]) -> None:
    # Atomic write: serialise to a sibling temp file and os.replace into place
    # so we never leave a truncated/half-written JSON file on disk if the
    # process is killed mid-write.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.replace(path)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_after_seconds_iso(seconds: float) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _is_expired(iso_timestamp: Optional[str]) -> bool:
    if not iso_timestamp:
        return False

    try:
        return datetime.fromisoformat(iso_timestamp) <= datetime.now(timezone.utc)
    except ValueError:
        return False


def _encode_state(payload: dict[str, Any]) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")


def _urlsafe_b64decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)
