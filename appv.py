# appv.py — Oura OAuth (server-side flow per docs)

import os
import requests
from flask import Flask, request, redirect, session, url_for
from requests_oauthlib import OAuth2Session
from requests.auth import HTTPBasicAuth

def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"{var_name} is not configured. Add it to your .env (see .env.example)."
        )
    return value


# ---- Oura OAuth config (from your app in the Oura portal) ----
CLIENT_ID     = _require_env('OURA_CLIENT_ID')
CLIENT_SECRET = _require_env('OURA_CLIENT_SECRET')
AUTH_URL      = 'https://cloud.ouraring.com/oauth/authorize'
TOKEN_URL     = 'https://api.ouraring.com/oauth/token'
REDIRECT_URI  = os.environ.get('REDIRECT_URI', 'http://localhost:5173/callback')
SCOPES        = os.environ.get('OURA_OAUTH_SCOPES', 'personal email').split()
# --------------------------------------------------------------

app = Flask(__name__)

@app.route('/oura_login')
def oura_login():
    """Redirect to the Oura OAuth consent page (no redirect_uri here)."""
    from hashlib import sha256
    import base64, os
    from requests_oauthlib import OAuth2Session

    # Set scope on the session (only here), but DO NOT set redirect_uri
    oura_session = OAuth2Session(CLIENT_ID, scope=SCOPES)

    # PKCE (S256)
    code_verifier  = base64.urlsafe_b64encode(os.urandom(40)).rstrip(b'=').decode('ascii')
    code_challenge = base64.urlsafe_b64encode(sha256(code_verifier.encode()).digest()).rstrip(b'=').decode('ascii')
    session['pkce_code_verifier'] = code_verifier
    # in /oura_login after generating the verifier
    print("PKCE code_verifier:", session['pkce_code_verifier'])


    # Build authorize URL WITHOUT redirect_uri (this avoids the 403)
    authorization_url, state = oura_session.authorization_url(
        AUTH_URL,
        code_challenge=code_challenge,
        code_challenge_method='S256'
    )
    session['oauth_state'] = state
    return redirect(authorization_url)


@app.route('/callback')
def callback():
    """
    Exchange ?code= for tokens. IMPORTANT: because /oura_login did NOT send
    redirect_uri, we must NOT send redirect_uri here either (per Oura docs).
    We use confidential-client token auth (HTTP Basic).
    """
    import urllib.parse

    parsed = urllib.parse.urlparse(request.url)
    qs     = urllib.parse.parse_qs(parsed.query)
    code   = qs.get('code', [''])[0]
    if not code:
        return "<pre>No ?code= in callback URL.</pre>", 400

    # Token payload: NO redirect_uri because it was omitted at authorize.
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "client_id": CLIENT_ID,                       # harmless w/ Basic, some servers like it
        # DO NOT include redirect_uri here
        # DO NOT include code_verifier when using confidential flow (not needed)
        # (You can keep PKCE if you want, but Basic is sufficient and simpler)
    }

    headers = {"Accept": "application/json"}

    # Confidential client auth (Basic)
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)

    # Official endpoints per Oura docs: api first, then cloud as fallback
    endpoints = [
        "https://api.ouraring.com/oauth/token",
        "https://cloud.ouraring.com/oauth/token",
    ]

    errors = []
    token  = None

    for url in endpoints:
        try:
            r = requests.post(url, data=payload, headers=headers, auth=auth, timeout=30)
            trace = f"{url} -> HTTP {r.status_code}\n{r.text}"
            if r.status_code == 200:
                try:
                    j = r.json()
                    if "access_token" in j:
                        token = j
                        break
                    errors.append(f"{trace}\n(no access_token in body)")
                except Exception as e:
                    errors.append(f"{trace}\n(JSON parse error: {e})")
            else:
                errors.append(trace)
        except Exception as e:
            errors.append(f"{url} -> request error: {e}")

    if not token:
        return "<pre>Token exchange failed:\n\n" + "\n\n---\n\n".join(errors) + "</pre>", 500

    session["oauth"] = token
    return redirect(url_for(".profile"))



@app.route('/profile')
def profile():
    """
    Proof that it worked. (You can swap to v2 once tokens flow.)
    """
    access_token = session['oauth']['access_token']

    # v1 example (as you already had):
    r = requests.get(
        'https://api.ouraring.com/v1/userinfo?access_token=' + access_token,
        timeout=30,
    )

    # v2 example (recommended):
    # r = requests.get(
    #     'https://api.ouraring.com/v2/usercollection/personal-info',
    #     headers={'Authorization': f'Bearer {access_token}'},
    #     timeout=30,
    # )

    return str(r.json())


if __name__ == '__main__':
    # Dev-only: allow HTTP for localhost
    os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
    app.secret_key = os.urandom(24)
    app.run(debug=False, host='localhost', port=5173)
