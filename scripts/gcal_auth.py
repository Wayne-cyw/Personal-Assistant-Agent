"""One-time interactive OAuth flow to obtain a Google Calendar refresh
token for the owner's account (Issue #17).

This deliberately does NOT import app.config.settings — that module
requires GOOGLE_REFRESH_TOKEN to already be set, which is exactly the
value this script exists to produce. Pass the OAuth client id/secret from
Google Cloud Console directly as arguments instead (see README's "Google
Calendar setup" section for how to create them).

Usage:
    python scripts/gcal_auth.py --client-id ... --client-secret ...

This opens a browser for the owner to sign in with their Google account
and grant access. On success, the refresh token is printed — paste it into
.env as GOOGLE_REFRESH_TOKEN.
"""

from __future__ import annotations

import argparse

from google_auth_oauthlib.flow import InstalledAppFlow

# Must match app/tools/calendar.py's scope exactly — a token issued for a
# different scope won't authorize the calls the app actually makes.
_SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-id", required=True, help="OAuth 2.0 client ID from Google Cloud Console"
    )
    parser.add_argument(
        "--client-secret", required=True, help="OAuth 2.0 client secret from Google Cloud Console"
    )
    args = parser.parse_args()

    client_config = {
        "installed": {
            "client_id": args.client_id,
            "client_secret": args.client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "redirect_uris": ["http://localhost"],
        }
    }
    flow = InstalledAppFlow.from_client_config(client_config, scopes=_SCOPES)
    credentials = flow.run_local_server(port=0)

    if credentials.refresh_token is None:
        raise SystemExit(
            "No refresh token was returned. This usually means the owner's Google account "
            "already granted this app access previously — revoke access at "
            "https://myaccount.google.com/permissions and run this script again."
        )

    print("\nSuccess. Add this to your .env file:\n")
    print(f"GOOGLE_REFRESH_TOKEN={credentials.refresh_token}")


if __name__ == "__main__":
    main()
