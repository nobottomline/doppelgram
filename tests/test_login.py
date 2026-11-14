from __future__ import annotations

import pytest
from telethon import errors

from doppelgram import cli


class FakeClient:
    def __init__(self, qr_raises: bool, wrong_first: int):
        self.qr_raises = qr_raises
        self.wrong_left = wrong_first
        self.passwords: list[str] = []

    async def qr_login(self):
        if self.qr_raises:
            raise errors.SessionPasswordNeededError(request=None)
        raise AssertionError("QR flow not expected in this test")

    async def sign_in(self, password=None):
        self.passwords.append(password)
        if self.wrong_left:
            self.wrong_left -= 1
            raise errors.PasswordHashInvalidError(request=None)


@pytest.fixture
def typed(monkeypatch):
    answers = iter(["wrong", "right", "x", "x", "x", "x"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(answers))


async def test_resumes_at_password_step_and_retries_wrong_password(typed, capsys):
    client = FakeClient(qr_raises=True, wrong_first=1)
    await cli._qr_login(client)
    assert client.passwords == ["wrong", "right"]
    assert "Wrong password. Attempts left: 4." in capsys.readouterr().out


async def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": "nope")
    client = FakeClient(qr_raises=True, wrong_first=100)
    with pytest.raises(SystemExit):
        await cli._qr_login(client)
    assert len(client.passwords) == cli.MAX_PASSWORD_ATTEMPTS


async def test_flood_wait_on_password_exits_with_a_clear_message(monkeypatch):
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": "pw")

    class Flooded(FakeClient):
        async def sign_in(self, password=None):
            raise errors.FloodWaitError(request=None, capture=600)

    with pytest.raises(SystemExit, match="wait 600 s"):
        await cli._qr_login(Flooded(qr_raises=True, wrong_first=0))
