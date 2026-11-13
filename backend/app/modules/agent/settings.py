"""The agent's settings and API keys (spec 026, R-AG-P3).

Settings live in `<workspace>/agent/settings.json`. One key per vendor, read from the environment
(`QUANTILO_<NAME>` then `<NAME>`, e.g. GEMINI_API_KEY), otherwise from
`<workspace>/agent/secrets.env`, written when the user pastes a key in the app. A pasted key is
filed under the vendor its shape reveals. Keys never leave the backend: the API only says whether
one is set and shows its last four characters."""

import contextlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, get_args

from app.modules.agent.engine.provider import Effort
from app.modules.agent.providers.vendors import BY_ID, PREFERENCE, VENDORS, Vendor, detect

EFFORTS: tuple[Effort, ...] = get_args(Effort)
KeySource = Literal["env", "file", "none"]
RowsSharing = Literal["never", "ask", "allow"]
SHARING: tuple[RowsSharing, ...] = get_args(RowsSharing)
AUTO = "auto"
MAX_UNMASKED = 200  # personal columns let through unmasked, per dataset


@dataclass
class AgentSettings:
    provider: str = AUTO  # a vendor id, or "auto": the first vendor with a key
    model: str = ""  # empty: the vendor's default
    base_url: str | None = None  # "custom" vendor only
    effort: Effort = "high"  # Claude only
    budget_usd: float = 1.0  # per conversation (R-AG-B2)
    # Raw rows (sample_rows, R-AG-P2). "never" by default, not the spec's "ask": the assistant
    # keeps the V1 promise of never seeing a row until the user opts in.
    rows_sharing: RowsSharing = "never"
    unmasked: dict[str, list[str]] = field(default_factory=dict)  # dataset id → column names


@dataclass(frozen=True)
class KeyStatus:
    configured: bool
    source: KeySource
    hint: str | None  # "…a1b2"


class SettingsStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self._refile()

    @property
    def _settings_path(self) -> Path:
        return self.folder / "settings.json"

    @property
    def _secrets_path(self) -> Path:
        return self.folder / "secrets.env"

    # --- settings ---------------------------------------------------------------------------

    def load(self) -> AgentSettings:
        try:
            raw = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return AgentSettings()
        found = AgentSettings()
        if raw.get("provider") in (AUTO, *BY_ID):
            found.provider = raw["provider"]
        if isinstance(raw.get("model"), str) and len(raw["model"]) <= 200:
            found.model = raw["model"]
        if isinstance(raw.get("base_url"), str) and raw["base_url"].startswith(
            ("http://", "https://")
        ):
            found.base_url = raw["base_url"]
        if raw.get("effort") in EFFORTS:
            found.effort = raw["effort"]
        budget = raw.get("budget_usd")
        if isinstance(budget, int | float) and 0 < budget <= 100:
            found.budget_usd = float(budget)
        if raw.get("rows_sharing") in SHARING:
            found.rows_sharing = raw["rows_sharing"]
        unmasked = raw.get("unmasked")
        if isinstance(unmasked, dict):
            found.unmasked = {
                str(dataset): [n for n in names if isinstance(n, str)][:MAX_UNMASKED]
                for dataset, names in unmasked.items()
                if isinstance(names, list) and names
            }
        return found

    def save(self, settings: AgentSettings) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        self._settings_path.write_text(json.dumps(asdict(settings), indent=2), encoding="utf-8")

    def vendor(self, settings: AgentSettings | None = None) -> Vendor:
        """The vendor in use: the chosen one, or the first with a key."""
        settings = settings or self.load()
        if settings.provider in BY_ID:
            return BY_ID[settings.provider]
        for vendor_id in PREFERENCE:
            if self.key(BY_ID[vendor_id])[0]:
                return BY_ID[vendor_id]
        return BY_ID["anthropic"]

    # --- keys ---------------------------------------------------------------------------------

    def _secrets(self) -> dict[str, str]:
        found: dict[str, str] = {}
        try:
            for line in self._secrets_path.read_text(encoding="utf-8").splitlines():
                name, _, value = line.partition("=")
                if name.strip() and value.strip():
                    found[name.strip()] = value.strip()
        except OSError:
            pass
        return found

    def _write_secrets(self, secrets: dict[str, str]) -> None:
        if not secrets:
            self._secrets_path.unlink(missing_ok=True)
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        text = "".join(f"{name}={value}\n" for name, value in sorted(secrets.items()))
        self._secrets_path.write_text(text, encoding="utf-8")
        with contextlib.suppress(OSError):  # owner-only where the system allows it
            os.chmod(self._secrets_path, 0o600)

    def _refile(self) -> None:
        """A key saved under the wrong vendor (before keys were told apart) moves to its own."""
        secrets = self._secrets()
        moved = False
        for vendor in VENDORS:
            value = secrets.get(vendor.secret)
            owner = detect(value) if value else None
            if owner is not None and owner.id != vendor.id and owner.secret not in secrets:
                secrets[owner.secret] = secrets.pop(vendor.secret)
                moved = True
        if moved:
            self._write_secrets(secrets)

    def key(self, vendor: Vendor) -> tuple[str, KeySource]:
        for name in (f"QUANTILO_{vendor.secret}", vendor.secret):
            value = os.environ.get(name, "").strip()
            if value:
                return value, "env"
        value = self._secrets().get(vendor.secret, "")
        return (value, "file") if value else ("", "none")

    def key_status(self, vendor: Vendor) -> KeyStatus:
        value, source = self.key(vendor)
        return KeyStatus(bool(value), source, f"…{value[-4:]}" if len(value) >= 8 else None)

    def save_key(self, value: str, vendor: Vendor | None = None) -> Vendor:
        """Files the key under its vendor (told by its shape, else the one given) and returns it.
        An empty value removes the given vendor's key."""
        value = value.strip()
        secrets = self._secrets()
        if not value:
            if vendor is not None:
                secrets.pop(vendor.secret, None)
                self._write_secrets(secrets)
            return vendor or BY_ID["anthropic"]
        if any(ch.isspace() for ch in value) or "=" in value:
            raise ValueError("This does not look like an API key.")
        owner = detect(value) or vendor
        if owner is None:
            raise ValueError(
                "Can't tell which provider this key belongs to: choose the provider first."
            )
        secrets[owner.secret] = value
        self._write_secrets(secrets)
        return owner
