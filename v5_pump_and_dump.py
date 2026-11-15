"""V5.1 bellek ve elektrik-kesintisi dayanıklı yürütücü.

Bu dosya mevcut ``v4_pump_and_dump.py`` deney tanımlarını yükler ve onları
değiştirmeden önce şu çalışma-zamanı iyileştirmelerini ekler:

* tamamlanmış deney varyantlarını özellik çıkarımından önce atlama,
* parça bazında BERT/SBERT ve şirket bazında zamansal kenar önbelleği,
* büyük özellik tensörlerini RAM yerine disk-eşlemeli dosyalarda tutma,
* 20 MB dönen dosya logu ve notebook'ta yalnız uyarı gösterimi,
* ``RUN_MODEL=True`` ve ``RESUME_RUN=True`` ile doğrudan devam.

Çalıştırma: ``python v5_pump_and_dump.py``
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import re
import runpy
from dataclasses import asdict
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Mapping

import torch


V4_PATH = Path(__file__).with_name("v4_pump_and_dump.py")
if not V4_PATH.exists():
    raise FileNotFoundError(f"Temel deney dosyası bulunamadı: {V4_PATH}")

# run_name farklı olduğu için V4'ün main() fonksiyonu yükleme sırasında çalışmaz.
# Fonksiyonların gerçek global sözlüğü yamalanır; runpy dönüş sözlüğü bazı
# Python sürümlerinde ayrı bir kopya olabildiğinden yalnız onu değiştirmek yetmez.
loaded_runtime = runpy.run_path(str(V4_PATH), run_name="_v4_runtime")
runtime = loaded_runtime["main"].__globals__
runtime["RUN_MODEL"] = True
runtime["RUN_SAVED_MODEL"] = False
runtime["RESUME_RUN"] = True

OUTPUT_DIR: Path = runtime["OUTPUT_DIR"]
CACHE_ROOT = Path(
    os.environ.get("V4_CACHE_ROOT", str(OUTPUT_DIR / "_v5_company_cache"))
)
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _text_digest(texts) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(str(text).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()


def configure_logging(output_dir: Path) -> None:
    """INFO'yu dönen dosyaya, yalnız uyarıları notebook/konsola gönderir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        output_dir / "experiment.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, console_handler],
        force=True,
    )


runtime["configure_logging"] = configure_logging
active_profile = {"name": "unknown"}
FEATURE_SHARD_MESSAGES = 8192


def _cache_profile(profile: str) -> str:
    """Veri ve kenar sözleşmesi özdeş profiller arasında ağır önbelleği paylaşır."""
    if profile in {
        "activity_controlled_tgn",
        "tgn_no_time_encoding",
        "tgn_daily_memory_reset",
    }:
        return "full_v4"
    return profile


def _cache_root(profile: str) -> Path:
    """Kontrollü koşuda tamamlanmış full_v4 ağır önbelleğini doğrudan kullanır."""
    if profile in {
        "activity_controlled_tgn",
        "tgn_no_time_encoding",
        "tgn_daily_memory_reset",
    }:
        return Path("result_logs_v4_full_v4") / "_v5_company_cache"
    return CACHE_ROOT


# ---------------------------------------------------------------------------
# Şirket bazlı metin özelliği önbelleği
# ---------------------------------------------------------------------------
original_encode = runtime["FeatureExtractors"].encode


def _mapped_tensor(path: Path, rows: int, columns: int) -> torch.Tensor:
    """Float32 ham dosyayı RAM'e kopyalamadan tensör olarak açar."""
    return torch.from_file(
        str(path), shared=False, size=rows * columns, dtype=torch.float32
    ).reshape(rows, columns)


def _mapped_cache_if_valid(meta_path: Path, identity: dict):
    if not meta_path.exists():
        return None
    try:
        meta = _torch_load(meta_path)
        if not isinstance(meta, dict) or meta.get("identity") != identity:
            return None
        rows = int(meta["rows"])
        bert_dim = int(meta["bert_dim"])
        sbert_dim = int(meta["sbert_dim"])
        bert_path = meta_path.with_name(meta["bert_file"])
        sbert_path = meta_path.with_name(meta["sbert_file"])
        if bert_path.stat().st_size != rows * bert_dim * 4:
            return None
        if sbert_path.stat().st_size != rows * sbert_dim * 4:
            return None
        return (
            _mapped_tensor(bert_path, rows, bert_dim),
            _mapped_tensor(sbert_path, rows, sbert_dim),
        )
    except Exception as exc:
        logging.warning("Disk-eşlemeli önbellek geçersiz (%s): %s", meta_path, exc)
        return None


def _write_raw_feature_files(shard_paths, meta_path: Path, identity: dict, rows: int):
    """Parça tensörlerini iki ham float32 dosyasına akış hâlinde birleştirir."""
    bert_path = meta_path.with_name("bert_features.f32")
    sbert_path = meta_path.with_name("sbert_features.f32")
    bert_tmp = bert_path.with_suffix(bert_path.suffix + ".tmp")
    sbert_tmp = sbert_path.with_suffix(sbert_path.suffix + ".tmp")
    bert_dim = sbert_dim = None
    written_rows = 0
    try:
        with bert_tmp.open("wb") as bert_handle, sbert_tmp.open("wb") as sbert_handle:
            for shard_path in shard_paths:
                payload = _torch_load(shard_path)
                bert = payload["bert"].float().contiguous()
                sbert = payload["sbert"].float().contiguous()
                bert_dim = int(bert.shape[1])
                sbert_dim = int(sbert.shape[1])
                bert.numpy().tofile(bert_handle)
                sbert.numpy().tofile(sbert_handle)
                written_rows += int(bert.shape[0])
                del payload, bert, sbert
                gc.collect()
        if written_rows != rows or bert_dim is None or sbert_dim is None:
            raise RuntimeError(f"Özellik parça sayımı uyuşmadı: {written_rows} != {rows}")
        os.replace(bert_tmp, bert_path)
        os.replace(sbert_tmp, sbert_path)
        _atomic_torch_save(
            {
                "identity": identity,
                "rows": rows,
                "bert_dim": bert_dim,
                "sbert_dim": sbert_dim,
                "bert_file": bert_path.name,
                "sbert_file": sbert_path.name,
            },
            meta_path,
        )
    finally:
        for temporary in (bert_tmp, sbert_tmp):
            if temporary.exists():
                temporary.unlink()
    return _mapped_cache_if_valid(meta_path, identity)


def cached_encode(self, texts, progress_label="Metin özellikleri"):
    company = progress_label.removesuffix(" özellik çıkarımı")
    requested_profile = active_profile["name"]
    profile = _cache_profile(requested_profile)
    company_dir = _cache_root(requested_profile) / profile / _safe_name(company)
    company_dir.mkdir(parents=True, exist_ok=True)
    meta_path = company_dir / "features_meta.pt"
    identity = {
        "cache_version": 2,
        "profile": profile,
        "company": company,
        "row_count": len(texts),
        "text_sha256": _text_digest(texts),
        "max_length": int(self.config.max_length),
    }
    mapped = _mapped_cache_if_valid(meta_path, identity)
    if mapped is not None:
        logging.warning("ELEKTRİK-RESUME: %s özellikleri disk eşlemesinden.", company)
        return mapped

    # Eski V5 şirket önbelleğini model çalıştırmadan disk eşlemesine dönüştür.
    legacy_path = _cache_root(requested_profile) / profile / f"{_safe_name(company)}_features.pt"
    legacy_identity = {key: value for key, value in identity.items() if key != "cache_version"}
    if legacy_path.exists():
        try:
            payload = _torch_load(legacy_path)
            if isinstance(payload, dict) and payload.get("identity") == legacy_identity:
                conversion_path = company_dir / "legacy_conversion.pt"
                _atomic_torch_save(
                    {"bert": payload["bert"], "sbert": payload["sbert"]}, conversion_path
                )
                del payload
                mapped = _write_raw_feature_files(
                    [conversion_path], meta_path, identity, len(texts)
                )
                conversion_path.unlink(missing_ok=True)
                legacy_path.unlink(missing_ok=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if mapped is not None:
                    logging.warning(
                        "ELEKTRİK-RESUME: %s eski önbellekten disk eşlemesine çevrildi.",
                        company,
                    )
                    return mapped
        except Exception as exc:
            logging.warning("Eski özellik önbelleği dönüştürülemedi (%s): %s", legacy_path, exc)

    shard_paths = []
    shard_count = max(1, (len(texts) + FEATURE_SHARD_MESSAGES - 1) // FEATURE_SHARD_MESSAGES)
    for shard_index, start in enumerate(
        range(0, len(texts), FEATURE_SHARD_MESSAGES), start=1
    ):
        end = min(start + FEATURE_SHARD_MESSAGES, len(texts))
        shard_path = company_dir / f"shard_{start:07d}_{end:07d}.pt"
        shard_identity = {
            "parent_sha256": identity["text_sha256"],
            "start": start,
            "end": end,
            "max_length": identity["max_length"],
        }
        valid_shard = False
        if shard_path.exists():
            try:
                payload = _torch_load(shard_path)
                valid_shard = (
                    isinstance(payload, dict)
                    and payload.get("identity") == shard_identity
                )
                del payload
            except Exception:
                valid_shard = False
        if not valid_shard:
            bert, sbert = original_encode(
                self,
                texts[start:end],
                progress_label=f"{company} özellik parçası {shard_index}/{shard_count}",
            )
            _atomic_torch_save(
                {
                    "identity": shard_identity,
                    "bert": bert.float().cpu(),
                    "sbert": sbert.float().cpu(),
                },
                shard_path,
            )
            del bert, sbert
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            logging.warning(
                "ELEKTRİK-RESUME: %s özellik parçası %d/%d hazır.",
                company,
                shard_index,
                shard_count,
            )
        shard_paths.append(shard_path)

    mapped = _write_raw_feature_files(shard_paths, meta_path, identity, len(texts))
    if mapped is None:
        raise RuntimeError(f"{company} disk-eşlemeli özellik önbelleği oluşturulamadı.")
    for shard_path in shard_paths:
        shard_path.unlink(missing_ok=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return mapped


runtime["FeatureExtractors"].encode = cached_encode


# ---------------------------------------------------------------------------
# Şirket bazlı zamansal kenar önbelleği
# ---------------------------------------------------------------------------
original_build_events = runtime["build_past_only_events"]


def _relation_filter() -> set[int] | None:
    """Return relation ids requested for a revision edge-type ablation."""
    raw = os.environ.get("V4_EDGE_RELATIONS", "all").strip().lower()
    if raw in {"", "all"}:
        return None
    groups = {
        "semantic": {0, 1, 2, 3},
        "quote": {4},
        "exact": {5},
        "url": {6},
    }
    selected: set[int] = set()
    for name in (item.strip() for item in raw.split(",")):
        if name not in groups:
            raise ValueError(
                "V4_EDGE_RELATIONS yalnız all, semantic, quote, exact veya url "
                f"değerlerini kabul eder; alınan: {name!r}"
            )
        selected.update(groups[name])
    return selected


def _filter_events(events_by_date, allowed: set[int] | None):
    if allowed is None:
        connected = {
            index
            for events in events_by_date.values()
            for event in events
            for index in (
                [int(event["source_idx"])]
                + [int(item[0]) for item in event["interactions"]]
            )
        }
        return events_by_date, connected
    filtered = {}
    connected: set[int] = set()
    for day, events in events_by_date.items():
        kept_events = []
        for event in events:
            interactions = [
                item for item in event["interactions"] if int(item[1]) in allowed
            ]
            if not interactions:
                continue
            kept_events.append(
                {"source_idx": int(event["source_idx"]), "interactions": interactions}
            )
            connected.add(int(event["source_idx"]))
            connected.update(int(item[0]) for item in interactions)
        if kept_events:
            filtered[day] = kept_events
    return filtered, connected


def cached_build_events(
    frame,
    sbert_features,
    config,
    allow_short_exact_edges=True,
    progress_label="Geçmişe dönük kenar çıkarımı",
):
    company = progress_label.removesuffix(" zamansal kenarları")
    requested_profile = active_profile["name"]
    profile = _cache_profile(requested_profile)
    cache_path = (
        _cache_root(requested_profile) / profile / f"{_safe_name(company)}_edges.pt"
    )
    identity = {
        "profile": profile,
        "company": company,
        "row_count": len(frame),
        "first_message": str(frame.iloc[0]["Mesaj_ID"]) if len(frame) else "",
        "last_message": str(frame.iloc[-1]["Mesaj_ID"]) if len(frame) else "",
        "allow_short_exact_edges": bool(allow_short_exact_edges),
        "time_window_seconds": int(config.time_window_seconds),
        "cosine_threshold": float(config.cosine_threshold),
        "top_k": int(config.top_k),
        "edge_relations": os.environ.get("V4_EDGE_RELATIONS", "all").strip().lower(),
    }
    if cache_path.exists():
        try:
            payload = _torch_load(cache_path)
            if isinstance(payload, dict) and payload.get("identity") == identity:
                logging.warning("ELEKTRİK-RESUME: %s zamansal kenarları önbellekten.", company)
                return _filter_events(payload["events_by_date"], _relation_filter())
        except Exception as exc:
            logging.warning("Kenar önbelleği okunamadı (%s): %s", cache_path, exc)
    events_by_date, connected_indices = original_build_events(
        frame,
        sbert_features,
        config,
        allow_short_exact_edges=allow_short_exact_edges,
        progress_label=progress_label,
    )
    _atomic_torch_save(
        {
            "identity": identity,
            "events_by_date": events_by_date,
            "connected_indices": connected_indices,
        },
        cache_path,
    )
    return _filter_events(events_by_date, _relation_filter())


runtime["build_past_only_events"] = cached_build_events


# ---------------------------------------------------------------------------
# Tamamlanmış varyantları pahalı ön işlemeden önce atlama
# ---------------------------------------------------------------------------
original_build_streams = runtime["build_streams"]


def _variant_is_complete(forum, config, profile: str) -> bool:
    folds = runtime["build_outer_folds"](forum, config)
    dominant = runtime["DOMINANT_EVENT_EXCLUSIONS"]
    if profile in dominant:
        event_year = dominant[profile][2]
        folds = [fold for fold in folds if fold.test_year >= event_year]
    if runtime["QUICK_SMOKE_TEST"] and folds:
        folds = [folds[-1]]

    config_hash = runtime["_stable_payload_hash"](asdict(config))
    forum_hash = runtime["file_sha256"](runtime["FORUM_CSV"])
    market_hash = runtime["file_sha256"](runtime["MARKET_CSV"])
    expected_experiments = 0
    for fold in folds:
        if len(fold.train_dates) <= config.rolling_folds + 1:
            continue
        for loss_mode in runtime["profile_loss_modes"](profile, config):
            expected_experiments += 1
            experiment_dir = (
                OUTPUT_DIR / profile / f"test_{fold.test_year}" / loss_mode
            )
            identity = runtime["_resume_identity"](
                config_hash,
                forum_hash,
                market_hash,
                data_variant=profile,
                test_year=fold.test_year,
                loss_mode=loss_mode,
            )
            if runtime["_load_experiment_cache"](experiment_dir, identity) is None:
                return False
    return expected_experiments > 0


def v5_build_streams(forum, extractor, config, experiment_profile="keep_all"):
    active_profile["name"] = experiment_profile
    # Tamamlanmış bir profil için boş akış döndürmek güvenli değildir:
    # aynı yapısal anahtarı kullanan daha sonraki, kısmen tamamlanmış profil
    # ``structural_stream_cache`` üzerinden bu boş sözlüğü devralabilir. Bu
    # durumda rolling-origin seçimi diskte bulunsa bile olay/sınıf desteği
    # denetimi 0/0 görür. Disk eşlemeli özellik ve kenar önbelleklerini yeniden
    # yüklemek ucuzdur ve profiller arası paylaşımı doğru tutar.
    return original_build_streams(
        forum, extractor, config, experiment_profile=experiment_profile
    )


runtime["build_streams"] = v5_build_streams

# Boş akış, yalnız tamamlanmış deneyin erken atlandığını gösterir. Var olan
# kenar denetim dosyasını bu durumda boş bir tabloyla ezme.
original_write_edge_audit = runtime["write_edge_audit"]


def v5_write_edge_audit(streams: Mapping, output_path: Path):
    if not streams and output_path.exists():
        return runtime["pd"].read_csv(output_path)
    return original_write_edge_audit(streams, output_path)


runtime["write_edge_audit"] = v5_write_edge_audit


if __name__ == "__main__":
    runtime["main"]()