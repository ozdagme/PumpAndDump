"""Q1 için aktivite-kontrollü TGN deneyini güvenle başlatır.

Bu koşu, tamamlanmış ``full_v4`` TGN sonucunu yeniden üretmez. Aynı 2021--2025
ileri-zincirli foldlar ve beş seed üzerinde yalnız ``activity_controlled_tgn``
profilini eğitir. Profil, standart TGN logitine güçlü aktivite baseline'ıyla
özdeş 13 şirket-gün kovaryatından öğrenilen doğrusal bir terim ekler.

Kullanım::

    python activity_controlled_q1_runner.py

Çıktı::

    result_logs_v4_activity_controlled/

Canlı izleme (Linux/macOS)::

    tail -f result_logs_v4_activity_controlled/experiment.log

Tamamlanma kontrolü::

    test -f result_logs_v4_activity_controlled/V4_COMPLETED && echo TAMAMLANDI
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    os.environ["V4_EXPERIMENT_SCOPE"] = "activity_controlled"
    runner = Path(__file__).with_name("v5_pump_and_dump.py")
    if not runner.exists():
        raise FileNotFoundError(f"V5 yürütücüsü bulunamadı: {runner}")
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()