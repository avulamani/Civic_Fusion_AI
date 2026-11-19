"""Simple role-based authentication for CivicFusion AI.

Two roles:
  - citizen  : default (no login required). Can submit complaints and
                track the status of their own ticket by ID.
  - officer  : requires credentials. Full access to the Municipal
                Command Dashboard, status updater, and analytics.

In production, replace OFFICER_CREDENTIALS with a database lookup
that stores hashed passwords (e.g., bcrypt).  For this coursework
demo, plain-text comparison is intentional and acceptable.
"""

# ---------------------------------------------------------------------------
# Officer account table  (username → password)
# Add more accounts here as needed.
# ---------------------------------------------------------------------------
OFFICER_CREDENTIALS: dict[str, str] = {
    "officer": "civic@2024",
    "admin":   "municipal#1",
}


def verify_credentials(username: str, password: str) -> bool:
    """Return True if username/password matches a known officer account."""
    stored = OFFICER_CREDENTIALS.get(username.strip().lower())
    return stored is not None and stored == password


def is_officer(session_state) -> bool:
    """Return True when the current session has officer-level access."""
    return session_state.get("role") == "officer"


def login(session_state, username: str, password: str) -> bool:
    """Attempt login.  Sets role in session_state; returns True on success."""
    if verify_credentials(username, password):
        session_state["role"] = "officer"
        session_state["logged_in_as"] = username.strip().lower()
        return True
    return False


def logout(session_state) -> None:
    """Clear authentication state."""
    session_state.pop("role", None)
    session_state.pop("logged_in_as", None)

