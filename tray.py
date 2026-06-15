"""
tray.py — System-tray launcher for the Mean Reversion Bot.

Puts an orange "MR" icon in the Windows system tray. The app opens idle (it does
NOT auto-start the bot). Use the menu to start/stop main.py, open the dashboard,
view the log, or quit. The bot runs as a child of this app, so quitting the tray
app stops the bot too.

The child bot is launched with this app's own interpreter (``sys.executable``),
so as long as you start the tray with Python 3.14 the bot inherits 3.14 and the
shared ``quantcore`` package — no version footgun.

Run with:  py -3.14 tray.py
"""
import os
import subprocess
import sys
import webbrowser

from PIL import Image, ImageDraw, ImageFont
import pystray

# ── Paths / config ──────────────────────────────────────────────────────────────
HERE          = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT   = os.path.join(HERE, "main.py")
BOT_LOG       = os.path.join(HERE, "bot.log")
DASHBOARD_URL = "http://localhost:8501"

ORANGE = (255, 140, 0)   # icon background
WHITE  = (255, 255, 255) # "MR" text

# The running bot process, or None when stopped.
_bot: "subprocess.Popen | None" = None


# ── Bot process management ──────────────────────────────────────────────────────
def bot_running() -> bool:
    """True if the bot child process exists and hasn't exited."""
    return _bot is not None and _bot.poll() is None


def start_bot(icon, _item=None):
    global _bot
    if bot_running():
        return
    # Append the child's stdout/stderr to bot.log so nothing is lost when the
    # bot runs without a console window. main.py's own logging also targets
    # bot.log; both appends interleave cleanly.
    log = open(BOT_LOG, "a", buffering=1, encoding="utf-8")
    _bot = subprocess.Popen(
        [sys.executable, MAIN_SCRIPT],
        cwd=HERE,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    icon.update_menu()


def stop_bot(icon, _item=None):
    global _bot
    if bot_running():
        _bot.terminate()
        try:
            _bot.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _bot.kill()
    _bot = None
    if icon is not None:
        icon.update_menu()


def open_dashboard(_icon=None, _item=None):
    webbrowser.open(DASHBOARD_URL)


def view_log(_icon=None, _item=None):
    if os.path.exists(BOT_LOG):
        os.startfile(BOT_LOG)  # noqa: S606 — open in the user's default editor


def quit_app(icon, _item=None):
    stop_bot(None)
    icon.stop()


# ── Icon image ──────────────────────────────────────────────────────────────────
def make_icon() -> Image.Image:
    """An orange square with white 'MR' centered."""
    size = 64
    img = Image.new("RGB", (size, size), ORANGE)
    draw = ImageDraw.Draw(img)

    text = "MR"
    font = None
    for name in ("arialbd.ttf", "arial.ttf"):
        try:
            font = ImageFont.truetype(name, 30)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    x = (size - (right - left)) / 2 - left
    y = (size - (bottom - top)) / 2 - top
    draw.text((x, y), text, fill=WHITE, font=font)
    return img


# ── Menu ────────────────────────────────────────────────────────────────────────
def status_text(_item) -> str:
    return "● Bot: running" if bot_running() else "○ Bot: stopped"


def build_menu() -> pystray.Menu:
    return pystray.Menu(
        pystray.MenuItem(status_text, None, enabled=False),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Start bot", start_bot, visible=lambda i: not bot_running()),
        pystray.MenuItem("Stop bot", stop_bot, visible=lambda i: bot_running()),
        pystray.MenuItem("Open dashboard", open_dashboard),
        pystray.MenuItem("View log (bot.log)", view_log),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit", quit_app),
    )


def main():
    icon = pystray.Icon(
        "meansrev",
        icon=make_icon(),
        title="Mean Reversion Bot",
        menu=build_menu(),
    )
    icon.run()


if __name__ == "__main__":
    main()
