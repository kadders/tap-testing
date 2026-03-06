# Desktop shortcut for Tap Test Cycle GUI

## Windows

### Option A: Shortcut to the batch file (console window appears briefly)

1. **Right-click** on the desktop → **New** → **Shortcut**.
2. **Target**: Browse to your `tap-testing` folder and select:
   - `scripts\run_cycle_gui.bat`  
   Or type the full path, e.g.:  
   `Z:\repositories\tap-testing\scripts\run_cycle_gui.bat`
3. **Name** the shortcut (e.g. “Tap Test Cycle GUI”) → **Finish**.
4. (Optional) Right-click the new shortcut → **Properties** → **Change Icon** to pick an icon.

Double-click the shortcut to run the GUI. A console window will open and then the GUI window; you can minimize or ignore the console.

### Option B: Shortcut with no console window (VBS launcher)

1. Create a shortcut as above, but set the **Target** to:
   - `wscript.exe "Z:\repositories\tap-testing\scripts\run_cycle_gui_silent.vbs"`  
   (Use your actual path to `tap-testing`.)
2. Name it (e.g. “Tap Test Cycle GUI”) → **Finish**.

Double-clicking runs only the GUI; no console window.

### Requirements

- Python and the project dependencies must be installed (e.g. `pip install -r requirements.txt`).
- If you use a **venv** in the `tap-testing` folder, the batch and VBS scripts will activate it automatically when present.

---

## Linux / Raspberry Pi

### .desktop file for the GUI

1. Create a file named `tap-test-cycle-gui.desktop` on your desktop or in `~/.local/share/applications/`:

   ```ini
   [Desktop Entry]
   Type=Application
   Name=Tap Test Cycle GUI
   Comment=Run the tap test cycle with status window and RPM chart
   Exec=sh -c "cd /home/pi/tap-testing && (test -f venv/bin/activate && . venv/bin/activate; python -m tap_testing.cycle_gui)"
   Icon=utilities-system-monitor
   Terminal=false
   Categories=Utility;Science;
   ```

2. Replace `/home/pi/tap-testing` with the actual path to your `tap-testing` project.
3. Make it executable:  
   `chmod +x tap-test-cycle-gui.desktop`
4. If you want it on the desktop:  
   `cp tap-test-cycle-gui.desktop ~/Desktop/`

Double-click the “Tap Test Cycle GUI” entry to run the GUI. Set `Terminal=true` if you want to see output in a terminal.
