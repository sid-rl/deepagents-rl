import base64
import os
import requests
import time

import jwt


def _get_jwt_token():
    key = base64.urlsafe_b64decode(
        os.getenv("GH_BOT_KEY").encode("utf-8")
    ).decode("utf-8")
    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + (10 * 60),
        "iss": os.getenv("GH_BOT_APP_ID"),
    }
    token = jwt.encode(payload, key, algorithm="RS256")
    return token

def get_access_token():
   jwt_token = _get_jwt_token()
   installation_id = os.getenv("GH_BOT_INSTALLATION_ID")
   headers = {
       "Authorization": f"Bearer {jwt_token}",
       "Accept": "application/vnd.github.v3+json",
   }

   url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
   response = requests.post(url, headers=headers)
   response.raise_for_status()

   return response.json()["token"]
