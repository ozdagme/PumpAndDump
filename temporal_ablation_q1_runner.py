"""Q1 zaman kodu ve günlük bellek sıfırlama ablasyonlarını başlatır.

Profiller:

* ``tgn_no_time_encoding``: zaman kodu yerine sıfır vektörü kullanır;
  günler arası kullanıcı belleği korunur.
* ``tgn_daily_memory_reset``: zaman kodu korunur; kullanıcı belleği her
  şirket-takvim günü başında sıfırlanır.

Çalıştırma::

    python temporal_ablation_q1_runner.py

Çıktı::

    result_logs_v4_temporal_ablations/
"""

from __future__ import annotations

import os
import runpy
from pathlib import Path


def main() -> None:
    os.environ["V4_EXPERIMENT_SCOPE"] = "temporal_ablations"
    runner = Path(__file__).with_name("v5_pump_and_dump.py")
    if not runner.exists():
        raise FileNotFoundError(f"V5 yürütücüsü bulunamadı: {runner}")
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()