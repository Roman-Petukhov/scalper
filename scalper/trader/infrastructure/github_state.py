"""Файлы состояния бота в ветке репозитория GitHub (REST «contents»): запись снимка и чтение недельного отчёта.
Нужен токен с правом записи в содержимое одного репозитория; ветку создаёт сам, если её нет."""
from __future__ import annotations

import base64

import httpx

API = "https://api.github.com"
HEADERS = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}


class GitHubState:
    def __init__(self, token: str, repo: str, branch: str = "bot-state",
                 client: httpx.AsyncClient | None = None) -> None:
        self.repo, self.branch = repo, branch
        self.client = client or httpx.AsyncClient(timeout=30, base_url=API)
        self._headers = {**HEADERS, "Authorization": f"Bearer {token}"}

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _call(self, method: str, url: str, **kw) -> httpx.Response:
        r = await self.client.request(method, url, headers=kw.pop("headers", self._headers), **kw)
        if r.status_code >= 400 and r.status_code != 404:
            raise RuntimeError(f"GitHub {method} {url.split('?')[0]}: {r.status_code} {r.text[:120]}")
        return r

    async def _ensure_branch(self) -> None:
        if (await self._call("GET", f"/repos/{self.repo}/branches/{self.branch}")).status_code != 404:
            return
        default = (await self._call("GET", f"/repos/{self.repo}")).json()["default_branch"]
        sha = (await self._call("GET", f"/repos/{self.repo}/git/ref/heads/{default}")).json()["object"]["sha"]
        await self._call("POST", f"/repos/{self.repo}/git/refs", json={"ref": f"refs/heads/{self.branch}", "sha": sha})

    async def put(self, path: str, data: bytes, message: str) -> None:
        await self._ensure_branch()
        url = f"/repos/{self.repo}/contents/{path}"
        cur = await self._call("GET", url, params={"ref": self.branch})
        body = {"message": message, "content": base64.b64encode(data).decode(), "branch": self.branch}
        if cur.status_code != 404:
            body["sha"] = cur.json()["sha"]
        await self._call("PUT", url, json=body)

    async def get(self, path: str) -> str | None:
        r = await self._call("GET", f"/repos/{self.repo}/contents/{path}", params={"ref": self.branch},
                             headers={**self._headers, "Accept": "application/vnd.github.raw+json"})
        return None if r.status_code == 404 else r.text
