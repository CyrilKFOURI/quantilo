"""119, R-VIS-WEB-04: the browser version starts without Pillow; Vision is mounted once it is loaded
(lite/vision_gate.py). Run in a fresh interpreter: Pillow has to be absent when the app is
imported, which the test process, having imported everything, cannot undo."""

import subprocess
import sys
import textwrap
from pathlib import Path

LITE = Path(__file__).resolve().parents[3] / "lite"

SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {lite!r})
    sys.modules["PIL"] = None  # the page has not loaded Pillow

    import vision_gate
    assert vision_gate.defer() and vision_gate.deferred()

    from fastapi.testclient import TestClient
    from app.config import Settings
    from app.core.jobs import InlineJobRunner
    from app.main import API_PREFIX, create_app

    app = create_app(
        Settings(home={home!r}, lite=True, allowed_hosts=["testserver"]),
        job_runner=InlineJobRunner(),
    )
    assert "quantilo.vision.decode" not in sys.modules
    with TestClient(app) as client:
        assert client.get(API_PREFIX + "/health").status_code == 200
        assert client.get(API_PREFIX + "/modules/vision/weights").status_code == 404
        try:
            vision_gate.enable(app, API_PREFIX)
        except RuntimeError as error:
            assert "R-VIS-WEB-04" in str(error)
        else:
            raise AssertionError("enabled without Pillow")

        del sys.modules["PIL"]  # the worker loaded it
        assert vision_gate.enable(app, API_PREFIX)
        assert not vision_gate.deferred() and not vision_gate.enable(app, API_PREFIX)
        assert hasattr(app.state, "vision")
        response = client.get(API_PREFIX + "/modules/vision/weights")
        assert response.status_code == 200, response.text
        assert client.get(API_PREFIX + "/images/rules").status_code == 200
    print("ok")
    """
)


def test_the_app_starts_without_pillow_and_mounts_vision_later(tmp_path: Path) -> None:
    script = SCRIPT.format(lite=str(LITE), home=str(tmp_path / "workspace"))
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300
    )
    assert done.returncode == 0, done.stderr[-3000:]
    assert done.stdout.strip().endswith("ok")
