"""Capture the running dashboard for the project documentation."""

from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from playwright.sync_api import sync_playwright, expect


def main():
    output = Path("docs/screenshots")
    output.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--host", "127.0.0.1", "--port", "8000",
    ])
    try:
        for attempt in range(60):
            try:
                with urlopen("http://127.0.0.1:8000/api/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except OSError:
                if server.poll() is not None:
                    raise RuntimeError("Application exited before startup")
                time.sleep(0.5)
        else:
            raise RuntimeError("Application did not become healthy")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1600, "height": 1050}, device_scale_factor=1)
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto("http://127.0.0.1:8000/", wait_until="load")
            expect(page.locator("#status")).to_contain_text("Run ready", timeout=60000)
            expect(page.locator("#event-count")).not_to_have_text("—")
            expect(page.locator("#edge-list .edge-row").first).to_be_visible()
            # Exercise the real run button before capturing the default seed.
            page.locator("#run-button").click()
            expect(page.locator("#run-button")).to_have_text("Run analysis", timeout=60000)
            expect(page.locator("#status")).to_contain_text("Run ready")
            page.screenshot(path=str(output / "dashboard.png"), full_page=True, animations="disabled")
            page.locator(".top-grid").screenshot(path=str(output / "diffusion-network.png"), animations="disabled")
            page.locator(".bottom-grid").screenshot(path=str(output / "backtest.png"), animations="disabled")
            browser.close()
            if errors:
                raise RuntimeError("Browser errors: " + "; ".join(errors))
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
