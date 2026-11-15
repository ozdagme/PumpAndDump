"""Nihai 53 şirket / 65 olay full_v4 TGN--MIL koşusunu başlatır.

Kullanım::

    python full_v4_runner.py

Asıl eğitim ve elektrik-kesintisi dayanıklı önbellek ``v5_pump_and_dump.py``
tarafından yürütülür. Bu küçük giriş dosyası yanlışlıkla eski ``dominant``
kapsamının seçilmesini önler.
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    os.environ["V4_EXPERIMENT_SCOPE"] = "full_v4"
    runner = Path(__file__).with_name("v5_pump_and_dump.py")
    if not runner.exists():
        raise FileNotFoundError(f"V5 yürütücüsü bulunamadı: {runner}")
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()