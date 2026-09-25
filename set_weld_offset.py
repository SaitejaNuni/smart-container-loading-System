#!/usr/bin/env python3
"""Change the weld relpose in the thesis MJCF quickly.

Usage: python3 set_weld_offset.py <x> <y> <z> <qw> <qx> <qy> <qz>

Examples:
  python3 set_weld_offset.py 0 0 -0.20 1 0 0 0    # down along wrist -Z
  python3 set_weld_offset.py 0 0 +0.20 1 0 0 0    # up along wrist +Z
"""
import re
import sys
from pathlib import Path

if len(sys.argv) != 8:
    print(__doc__)
    sys.exit(1)

vals = sys.argv[1:8]
new_relpose = " ".join(vals)

path = Path.home() / "thesis_mujoco_ws/src/ur10_mujoco_bringup/mjcf_thesis/thesis_scene.xml"
content = path.read_text()

pattern = re.compile(r'(<weld[^>]*?relpose=")[^"]*(")')
new_content, count = pattern.subn(r'\g<1>' + new_relpose + r'\g<2>', content)

path.write_text(new_content)
print(f"Updated {count} welds with relpose='{new_relpose}'")
