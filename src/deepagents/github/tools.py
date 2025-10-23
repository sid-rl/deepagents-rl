import logging
import requests

from langchain_core.tools import tool

from deepagents.github.auth import get_access_token


logger = logging.getLogger(__name__)


BASE_URL = "https://api.github.com/repos/langchain-ai"


def comment_on_issue(repo: str, issue_number: int, comment: str) -> None:
    """Comment on a Github issue.

    Args:
        repo: repository name (e.g., "langchain")
        issue_number: Number of the issue to comment on
        comment: Text of the comment to post
    """
    url = f"{BASE_URL}/{repo}/issues/{issue_number}/comments"

    access_token = get_access_token()
    headers = {
        "Authorization": f"token {access_token}",
        "Accept": "application/vnd.github.v3+json"
    }

    data = {"body": comment}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 201:
        logger.info(f"Commented on issue {issue_number}.")
    else:
        error_message = (
            f"Failed to comment: {response.status_code}\n\n"
            f"{response.json()}"
        )
        logger.error(error_message)
