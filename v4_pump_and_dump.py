"""V4: major-revizyon ve çizge-gürültüsü kontrollü TGN--MIL deney paketi.

Bu dosya ``# %%`` işaretleri sayesinde VS Code, Spyder veya Jupyter uyumlu
editörlerde hücreler hâlinde çalıştırılabilir. Ana bölümler şunlardır:

1. AYARLAR: Veri yollarını, ileri-zincirli dış test yıllarını, kısa ileti
   ablasyonlarını ve deney aşamalarını tanımlar.
2. DENETİM: Büyük CSV dosyalarını modele vermeden önce etiket, şirket-gün,
   yinelenen ileti, piyasa eşleşmesi, olay sayısı ve test sınıf desteği bakımından
   inceler; paylaşılabilir küçük bir ZIP raporu üretir.
3. MODEL: BERTurk/SBERT özelliklerini; alıntı, tam metin, URL ve çok ölçekli
   anlamsal ilişkileri; olay-güdümlü TGN belleğini ve dikkat tabanlı MIL'i kurar.
   Standart BCE ile PU-duyarlı risk yaklaşımını aynı dış fold'larda karşılaştırır.
4. FİNANSAL DEĞERLENDİRME: Ham, sert filtreli ve sürekli ağırlıklandırılmış
   duyarlılık sinyallerini uzun/nakit ve kesitsel top-k stratejilerinde sınar.
5. KAYIT/YENİDEN KULLANIM: Her seed modelini ve validasyonda en iyi modeli
   checkpoint olarak saklar; ``SavedModelPredictor`` yeni forum CSV'lerinde
   şirket-gün olasılıkları üretir.

V4, V3'ün değişmez ``keep_all`` referansını korur; kısa metinlerden doğan
kenarları mesajları silmeden kapatan ``no_short_edges`` koşulunu ve nedensel
kullanıcı aktivitesi ile kapılı metin--TGN birleşimini kullanan ``revised_tgn``
koşulunu önceden tanımlı ikincil deneyler olarak ekler. Uzman etiketli veri ile
like/dislike alanları bu sürümün veri sözleşmesinde ve deney matrisinde yoktur.

Başlıca çıktılar ``AUDIT_DIR`` ve ``OUTPUT_DIR`` altına yazılır. Her veri
varyantı, dış fold, kayıp türü, seed ve finansal strateji ayrı anahtarlarla
kaydedilir. Daha önce incelenmiş 2021--2025 dönemleri keşifsel sağlamlık
analizidir; yeni gelecekteki olaylar doğrulayıcı değerlendirme için saklanmalıdır.
"""

from __future__ import annotations

# %% HÜCRE 1 — AYARLAR
# HÜCRE 1/5 — Dosya yolları, kilitli test ve model kullanım ayarları
import os
from pathlib import Path

# Uzun V4 matrisi aşamalı çalıştırılır. ``primary`` yalnız önceden kayıtlı
# birincil hipotezi (keep_all + BCE) sınar; metin ve statik çizge temelleri aynı
# deney içinde zaten üretilir. ``dominant`` yalnız baskın olay dışlama
# yeniden-eğitimlerini, ``priority`` temel sağlamlık profillerini, ``full`` ise
# yayın paketindeki bütün ikincil/keşifsel analizleri çalıştırır.
EXPERIMENT_SCOPE = os.environ.get("V4_EXPERIMENT_SCOPE", "full_v4").strip().lower()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_int_tuple(name: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    value = os.environ.get(name)
    if not value:
        return default
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError(f"{name} en az bir tamsayı içermelidir.")
    return parsed

FORUM_CSV = Path("forum_verisi_v4.csv")
MARKET_CSV = Path("piyasa_verisi_v4.csv")
AUDIT_DIR = Path("dataset_audit_v4")
OUTPUT_DIR = Path(
    os.environ.get("V4_OUTPUT_DIR", f"result_logs_v4_{EXPERIMENT_SCOPE}")
)
ARCHIVE_ALL_SEED_CURVES = os.environ.get(
    "V4_ARCHIVE_ALL_SEED_CURVES", "0"
).strip().lower() in {"1", "true", "yes"}
ARCHIVE_USER_SCORES = os.environ.get(
    "V4_ARCHIVE_USER_SCORES", "0"
).strip().lower() in {"1", "true", "yes"}
ARCHIVE_TRAINING_CURVES = os.environ.get(
    "V4_ARCHIVE_TRAINING_CURVES", "0"
).strip().lower() in {"1", "true", "yes"}

# İncelenmiş tarihsel dönemler yalnız keşifsel ileri-zincirli dış testlerdir.
# Her yılın testi yalnız o yıldan önceki tarihlerle eğitilir.
OUTER_TEST_YEARS = (2021, 2022, 2023, 2024, 2025)
# Tarihsel V4 paneli bu aralıkta dondurulur. 2026 ve sonrası, etiketleri
# olgunlaştıktan sonra ayrı bir doğrulayıcı holdout olarak açılmalıdır.
ANALYSIS_START = "2015-10-15 00:00:00"
ANALYSIS_END = "2025-12-31 23:59:59.999999"
# Eski yardımcıların ve kayıtlı-model arayüzünün uyumluluğu için son dış fold
# başlangıcı ayrıca tutulur; V4 tarihsel deney paketi tek bir holdout'a dayanmaz.
LOCKED_TEST_START = f"{max(OUTER_TEST_YEARS)}-01-01"

# Ana analiz tüm boş olmayan iletileri korur. ``min3_ablation`` yalnız kısa
# ileti kararının etkisini ölçen karşılaştırmadır; ana veri temizliği değildir.
# The public code-only release omits private firm/event mappings. Event-specific
# exclusion profiles therefore remain unavailable in this release.
RUN_DOMINANT_EVENT_ABLATIONS = False
DOMINANT_EVENT_EXCLUSIONS: Mapping[str, Tuple[str, str, int]] = {}
FULL_BASE_DATA_VARIANTS = (
    "keep_all",
    "keep_all_drop_unknown",
    "no_short_edges",
    "revised_tgn",
    "revised_tgn_long",
    "revised_no_activity",
    "revised_no_gate",
    "min3_ablation",
)
FULL_DATA_VARIANTS = FULL_BASE_DATA_VARIANTS + (
    tuple(DOMINANT_EVENT_EXCLUSIONS)
    if RUN_DOMINANT_EVENT_ABLATIONS
    else tuple()
)
if EXPERIMENT_SCOPE == "primary":
    DATA_VARIANTS = ("keep_all",)
elif EXPERIMENT_SCOPE == "full_v4":
    # Nihai 53 şirket / 65 olay veri setinin hiçbir pozitif olay çıkarılmadan
    # çalıştırıldığı ana referans koşulu. Baskın-olay profilleri bunun sağlamlık
    # analizidir; full_v4 ayrı çıktı dizininde tutulur.
    DATA_VARIANTS = ("full_v4",)
elif EXPERIMENT_SCOPE == "activity_controlled":
    # Q1 hakem kontrolü: full_v4 ile aynı veri/kenarlar ve yalnızca güçlü
    # baseline'daki 13 şirket-gün aktivite kovaryatını ekleyen tek profil.
    DATA_VARIANTS = ("activity_controlled_tgn",)
elif EXPERIMENT_SCOPE == "temporal_ablations":
    # Q1 hakem kontrolü: standart full_v4 TGN'nin iki zamansal bileşenini
    # birbirinden ayırır. Veri, kenarlar, foldlar ve diğer hiperparametreler
    # değiştirilmez.
    DATA_VARIANTS = ("tgn_no_time_encoding", "tgn_daily_memory_reset")
elif EXPERIMENT_SCOPE == "dominant":
    if not RUN_DOMINANT_EVENT_ABLATIONS:
        raise ValueError(
            "dominant aşaması için RUN_DOMINANT_EVENT_ABLATIONS=True olmalıdır."
        )
    DATA_VARIANTS = tuple(DOMINANT_EVENT_EXCLUSIONS)
elif EXPERIMENT_SCOPE == "priority":
    DATA_VARIANTS = (
        "keep_all",
        "no_short_edges",
        "revised_tgn",
        "min3_ablation",
    )
elif EXPERIMENT_SCOPE == "full":
    DATA_VARIANTS = FULL_DATA_VARIANTS
else:
    raise ValueError(
        "EXPERIMENT_SCOPE yalnız 'full_v4', 'activity_controlled', "
        "'temporal_ablations', 'primary', 'dominant', 'priority' veya 'full' olabilir."
    )
RUN_REVISED_COMPONENT_ABLATIONS = True

# Denetim onaylanana kadar ikisi de False kalmalıdır.
CONFIRM_UNSEEN_TEST = False
RUN_MODEL = False

# Hızlı hata ayıklamada True yapılabilir. Bu mod tek seed, az epoch ve az
# bootstrap kullanır; yayın sonucu olarak raporlanmamalıdır.
QUICK_SMOKE_TEST = False


# Elektrik kesintisi/yeniden başlatma sonrası tamamlanan işleri atla.
RESUME_RUN = True
# Eğitim bittikten sonra her rassal başlangıç için model ağırlıkları kaydedilir.
SAVE_MODELS = True

# Daha önce kaydedilmiş bir modeli yalnız tahmin amacıyla çalıştırmak için:
# 1) RUN_MODEL=False bırakın.
# 2) RUN_SAVED_MODEL=True yapın.
# 3) Aşağıdaki iki yolu kendi dosyalarınıza göre değiştirin.
RUN_SAVED_MODEL = False
SAVED_MODEL_PATH = OUTPUT_DIR / "saved_models" / "best_model.pt"
INFERENCE_FORUM_CSV = Path("yeni_forum_verisi.csv")
INFERENCE_OUTPUT_CSV = OUTPUT_DIR / "saved_model_predictions.csv"

print("Forum:", FORUM_CSV.resolve())
print("Piyasa:", MARKET_CSV.resolve())
print("Keşifsel dış test yılları:", OUTER_TEST_YEARS)
print("V4 çalışma aşaması:", EXPERIMENT_SCOPE)
print("V4 veri/model profilleri:", DATA_VARIANTS)
print("Model kayıt dizini:", (OUTPUT_DIR / "saved_models").resolve())

# %% HÜCRE 2 — DENETİM FONKSİYONLARI
# HÜCRE 2/5 — CSV okuma ve veri sözleşmesi denetim yardımcıları
import hashlib
import re
import shutil
import pandas as pd


def read_csv_auto(path):
    """CSV dosyasını ayraç ve ondalık biçimini otomatik algılayarak okur.

    Amaç: Forum ve piyasa dosyalarının virgül ya da noktalı virgül ile
    kaydedilmiş olabilmesini tek bir okuma noktasında yönetmektir. İlk satırdaki
    ayraç sıklığına göre ``sep`` seçilir; noktalı virgüllü dosyalarda Türkçe
    Excel çıktılarıyla uyumlu olması için ondalık ayırıcı virgül kabul edilir.

    Girdi: ``path`` okunacak CSV yoludur.
    Çıktı: Bütün sütunları içeren bir ``pandas.DataFrame`` döndürür.
    Yan etki: Dosyayı diskten okur; veri üzerinde temizlik yapmaz.
    """
    with open(path, "r", encoding="utf-8-sig", errors="replace") as handle:
        preview = "".join(handle.readline() for _ in range(20))
    header = preview.splitlines()[0]
    separator = ";" if header.count(";") > header.count(",") else ","
    options = {"sep": separator, "encoding": "utf-8-sig", "low_memory": False}
    if separator == ";" and re.search(r"(?<!\d)\d+,\d+(?!\d)", preview):
        options["decimal"] = ","
    return pd.read_csv(path, **options)


def file_sha256(path):
    """Bir dosyanın değişmez içerik parmak izini SHA-256 ile hesaplar.

    Büyük veri dosyaları belleğe bütünüyle alınmadan 1 MB'lık bloklarla okunur.
    Üretilen özet, kullanılan forum ve piyasa sürümünün deney manifestinde kesin
    olarak tanımlanmasını ve sonraki koşularla karşılaştırılmasını sağlar.

    Girdi: Karma değeri hesaplanacak dosya yolu.
    Çıktı: 64 karakterlik onaltılık SHA-256 metni.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame, columns, name):
    """Bir veri tablosunun zorunlu sütun sözleşmesini sağladığını denetler.

    Beklenen alanlardan biri yoksa işlem erken durdurulur; böylece yanlış sütun
    adlarının ileride sessizce eksik etiket, boş tarih veya hatalı backtest
    üretmesi engellenir.

    Girdiler: ``frame`` denetlenecek DataFrame, ``columns`` zorunlu alanlar,
    ``name`` ise hata mesajında kullanılacak veri seti adıdır.
    Çıktı: Başarılı durumda değer döndürmez; eksikte ``ValueError`` yükseltir.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} eksik sütunlar içeriyor: {missing}")


print("Denetim fonksiyonları hazır. Hücre 3'ü çalıştırın.")

# %% HÜCRE 3 — DENETİM RAPORUNU ÜRET
# HÜCRE 3/5 — Modeli çalıştırmadan küçük denetim ZIP'i üretir
#
# AMAÇ:
# Bu blok ham ileti metinlerini dışarı aktarmadan yeni veri sürümünün modele
# uygunluğunu ölçer. Forum ve piyasa dosyalarını okur; şirket, ileti, kullanıcı,
# şirket-gün ve etiket dağılımlarını çıkarır. Yinelenen ileti, karışık günlük
# etiket, eksik piyasa eşleşmesi, aşırı getiri ve kilitli testte tek sınıf gibi
# kritik sorunları raporlar.
#
# ÜRETTİĞİ DOSYALAR:
# - dataset_audit_report.txt: İnsan tarafından okunabilir ana özet ve PASS/FAIL.
# - company_summary.csv: Şirket bazlı tarih, ileti, kullanıcı ve sınıf sayıları.
# - year_summary.csv: Yıl ve etiket bazlı ileti/şirket dağılımı.
# - market_anomalies.csv: Piyasa dosyasındaki bütün şüpheli fiyat satırları.
# - panel_market_anomalies.csv: Forum analiz paneline gerçekten düşen anomaliler.
# - unmatched_company_days.csv: Piyasa verisiyle eşleşmeyen forum şirket-günleri.
# - dataset_audit_v2.zip: Yukarıdaki küçük raporların paylaşılabilir paketi.
#
# Bu hücre model eğitmez, BERT/SBERT yüklemez ve test olasılığı üretmez.
if not FORUM_CSV.exists() or not MARKET_CSV.exists():
    raise FileNotFoundError("Hücre 1'deki forum veya piyasa dosya adını kontrol edin.")

forum_audit = read_csv_auto(FORUM_CSV)
market_audit = read_csv_auto(MARKET_CSV)
require_columns(
    forum_audit,
    ["Zaman_Damgasi", "Hisse_Kodu", "Mesaj_ID", "Alinti_Mesaj_ID",
     "Mesaj_Metni", "Kullanici_ID", "Kullanici_Bilinmiyor", "SPK_Etiketi",
     "Olay_ID"],
    "Forum",
)
require_columns(
    market_audit,
    ["Tarih", "Hisse_Kodu", "T1_Acilis", "T1_Kapanis", "Gunluk_Risksiz_Getiri"],
    "Piyasa",
)

forum_audit["Zaman_Damgasi"] = pd.to_datetime(forum_audit["Zaman_Damgasi"], errors="coerce")
forum_audit["Date"] = forum_audit["Zaman_Damgasi"].dt.normalize()
audit_start = pd.Timestamp(ANALYSIS_START)
audit_end = pd.Timestamp(ANALYSIS_END)
forum_audit = forum_audit.loc[
    forum_audit["Zaman_Damgasi"].between(audit_start, audit_end)
].copy()
forum_audit["Hisse_Kodu"] = forum_audit["Hisse_Kodu"].astype(str).str.strip().str.upper()
forum_audit["SPK_Etiketi"] = pd.to_numeric(forum_audit["SPK_Etiketi"], errors="coerce")
forum_audit["Olay_ID_Normalize"] = (
    forum_audit["Olay_ID"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.replace(r"\.0$", "", regex=True)
)
forum_audit["Olay_Isaretli"] = ~forum_audit["Olay_ID_Normalize"].isin(["", "0"])
unknown_audit_raw = (
    forum_audit["Kullanici_Bilinmiyor"].astype(str).str.strip().str.casefold()
)
unknown_audit_map = {
    "1": True, "true": True, "evet": True, "yes": True,
    "0": False, "false": False, "hayir": False, "hayır": False, "no": False,
}
unknown_audit_invalid = ~unknown_audit_raw.isin(unknown_audit_map)
forum_audit["Kullanici_Bilinmiyor_Normalize"] = unknown_audit_raw.map(
    unknown_audit_map
)
unknown_audit_rows = forum_audit[
    forum_audit["Kullanici_Bilinmiyor_Normalize"] == True
].copy()
unknown_audit_rows["Teknik_Kullanici_ID"] = (
    "UNKNOWN_MSG_"
    + unknown_audit_rows["Hisse_Kodu"].astype(str)
    + "_"
    + unknown_audit_rows["Mesaj_ID"].astype(str)
)
unknown_audit_duplicate_ids = int(
    unknown_audit_rows["Teknik_Kullanici_ID"].duplicated().sum()
)
known_audit_missing_ids = int(
    (
        (forum_audit["Kullanici_Bilinmiyor_Normalize"] == False)
        & forum_audit["Kullanici_ID"].fillna("").astype(str).str.strip().eq("")
    ).sum()
)
forum_audit["Kelime_Sayisi_Hesaplanan"] = (
    forum_audit["Mesaj_Metni"].fillna("").astype(str).str.split().str.len()
)
forum_audit["Kisa_Mesaj_Hesaplanan"] = forum_audit["Kelime_Sayisi_Hesaplanan"] < 3
market_audit["Tarih"] = pd.to_datetime(market_audit["Tarih"], errors="coerce").dt.normalize()
market_audit = market_audit.loc[
    market_audit["Tarih"].between(audit_start.normalize(), audit_end.normalize())
].copy()
market_audit["Hisse_Kodu"] = market_audit["Hisse_Kodu"].astype(str).str.strip().str.upper()
market_audit["T1_Acilis"] = pd.to_numeric(market_audit["T1_Acilis"], errors="coerce")
market_audit["T1_Kapanis"] = pd.to_numeric(market_audit["T1_Kapanis"], errors="coerce")

day_table = (
    forum_audit.dropna(subset=["Date", "Hisse_Kodu", "SPK_Etiketi"])
    .groupby(["Hisse_Kodu", "Date"])["SPK_Etiketi"]
    .agg(min_label="min", max_label="max", label_count="nunique", Mesaj="size")
    .reset_index()
)
company_day_label_conflicts = day_table.loc[day_table["label_count"] > 1].copy()
duplicate_message_ids = (
    forum_audit.loc[forum_audit["Mesaj_ID"].duplicated(keep=False)]
    .groupby("Mesaj_ID", as_index=False)
    .agg(
        Satir=("Mesaj_ID", "size"),
        Sirket=("Hisse_Kodu", "nunique"),
        Ilk_Zaman=("Zaman_Damgasi", "min"),
        Son_Zaman=("Zaman_Damgasi", "max"),
        Etiket_Sayisi=("SPK_Etiketi", "nunique"),
        Olay_ID_Sayisi=("Olay_ID_Normalize", "nunique"),
    )
    .sort_values(["Satir", "Mesaj_ID"], ascending=[False, True])
)
positive_missing_event_id = forum_audit.loc[
    forum_audit["SPK_Etiketi"].eq(1) & ~forum_audit["Olay_Isaretli"],
    ["Hisse_Kodu", "Date", "Mesaj_ID", "Olay_ID_Normalize"],
].copy()
event_marked_but_not_positive = forum_audit.loc[
    forum_audit["Olay_Isaretli"] & ~forum_audit["SPK_Etiketi"].eq(1)
].copy()
event_marked_but_not_positive_summary = (
    event_marked_but_not_positive
    .groupby(["Hisse_Kodu", "Olay_ID_Normalize"], as_index=False)
    .agg(
        Mesaj=("Mesaj_ID", "size"),
        Sirket_Gun=("Date", "nunique"),
        Ilk_Zaman=("Zaman_Damgasi", "min"),
        Son_Zaman=("Zaman_Damgasi", "max"),
        Etiketler=("SPK_Etiketi", lambda values: ",".join(
            sorted({str(value) for value in values.dropna().unique()})
        )),
    )
    .rename(columns={"Olay_ID_Normalize": "Olay_ID"})
    .sort_values(["Mesaj", "Hisse_Kodu"], ascending=[False, True])
)
company_rows = (
    forum_audit.groupby("Hisse_Kodu")
    .agg(
        Ilk_Zaman=("Zaman_Damgasi", "min"),
        Son_Zaman=("Zaman_Damgasi", "max"),
        Mesaj=("Mesaj_ID", "size"),
        Benzersiz_Mesaj=("Mesaj_ID", "nunique"),
        Kullanici=("Kullanici_ID", "nunique"),
        Pozitif_Satir=("SPK_Etiketi", lambda x: int((x == 1).sum())),
        Negatif_Satir=("SPK_Etiketi", lambda x: int((x == 0).sum())),
    )
    .reset_index()
)
company_days = (
    day_table.groupby("Hisse_Kodu")
    .agg(
        Toplam_Gun=("Date", "size"),
        Pozitif_Gun=("max_label", lambda x: int((x == 1).sum())),
        Negatif_Gun=("min_label", lambda x: int((x == 0).sum())),
        Karisik_Gun=("label_count", lambda x: int((x > 1).sum())),
    )
    .reset_index()
)
company_summary = company_rows.merge(company_days, on="Hisse_Kodu", how="left")
year_summary = (
    forum_audit.dropna(subset=["Date", "SPK_Etiketi"])
    .assign(Yil=lambda x: x["Date"].dt.year)
    .groupby(["Yil", "SPK_Etiketi"])
    .agg(Mesaj=("Mesaj_ID", "size"), Sirket=("Hisse_Kodu", "nunique"))
    .reset_index()
)
short_message_summary = (
    forum_audit.dropna(subset=["Date", "SPK_Etiketi"])
    .assign(Yil=lambda x: x["Date"].dt.year)
    .groupby(["Yil", "SPK_Etiketi"], as_index=False)
    .agg(
        Mesaj=("Mesaj_ID", "size"),
        Kisa_Mesaj=("Kisa_Mesaj_Hesaplanan", "sum"),
        Ortalama_Kelime=("Kelime_Sayisi_Hesaplanan", "mean"),
        Sirket=("Hisse_Kodu", "nunique"),
    )
)
unknown_user_summary = (
    forum_audit.dropna(subset=["Date", "SPK_Etiketi"])
    .assign(Yil=lambda x: x["Date"].dt.year)
    .groupby(["Yil", "SPK_Etiketi"], as_index=False)
    .agg(
        Mesaj=("Mesaj_ID", "size"),
        Bilinmeyen_Mesaj=("Kullanici_Bilinmiyor_Normalize", "sum"),
    )
)
unknown_user_summary["Bilinmeyen_Orani"] = (
    unknown_user_summary["Bilinmeyen_Mesaj"]
    / unknown_user_summary["Mesaj"].clip(lower=1)
)

forum_keys = day_table[["Hisse_Kodu", "Date"]].rename(columns={"Date": "Tarih"}).drop_duplicates()
market_keys = market_audit[["Hisse_Kodu", "Tarih"]].dropna().drop_duplicates()
coverage = forum_keys.merge(
    market_keys.assign(Piyasa_Eslesmesi=1), on=["Hisse_Kodu", "Tarih"], how="left"
)
unmatched = coverage[coverage["Piyasa_Eslesmesi"].isna()].drop(columns="Piyasa_Eslesmesi")
market_audit["Gunici_Getiri"] = market_audit["T1_Kapanis"] / market_audit["T1_Acilis"] - 1
market_anomalies = market_audit[
    (market_audit["T1_Acilis"] <= 0)
    | (market_audit["T1_Kapanis"] <= 0)
    | market_audit["Gunici_Getiri"].isna()
    | (market_audit["Gunici_Getiri"].abs() > 0.50)
]
panel_market_anomalies = market_anomalies.merge(
    forum_keys, on=["Hisse_Kodu", "Tarih"], how="inner"
)
market_panel = forum_keys.merge(
    market_audit[["Hisse_Kodu", "Tarih", "Gunici_Getiri"]],
    on=["Hisse_Kodu", "Tarih"],
    how="inner",
)
market_panel["Yil"] = market_panel["Tarih"].dt.year
market_scale_summary = (
    market_panel.groupby(["Yil", "Hisse_Kodu"], as_index=False)
    .agg(
        Gun=("Gunici_Getiri", "size"),
        Ortalama_Gunici_Getiri=("Gunici_Getiri", "mean"),
        Medyan_Gunici_Getiri=("Gunici_Getiri", "median"),
    )
)
# En az 10 gözlemde yüzde 8'i aşan sistematik ortalama/medyan gün içi getiri,
# tekil fiyat sıçramasından çok açılış ve kapanış serilerinin farklı düzeltme
# ölçeklerinde olabileceğine işaret eder.
market_scale_suspicious = market_scale_summary[
    (market_scale_summary["Gun"] >= 10)
    & (
        (market_scale_summary["Ortalama_Gunici_Getiri"].abs() > 0.08)
        | (market_scale_summary["Medyan_Gunici_Getiri"].abs() > 0.08)
    )
].copy()

test_start = pd.Timestamp(LOCKED_TEST_START)
development_days = day_table[day_table["Date"] < test_start]
test_days = day_table[day_table["Date"] >= test_start]
positive_companies = company_summary[company_summary["Pozitif_Gun"] > 0]
mixed_companies = positive_companies[positive_companies["Negatif_Gun"] > 0]
official_events = "Olay_ID sütunu yok"
positive_event_ids_complete = False
event_id_summary = pd.DataFrame(
    columns=[
        "Hisse_Kodu", "Olay_ID", "Mesaj_Sayisi", "Benzersiz_Mesaj",
        "Ilk_Zaman", "Son_Zaman", "Sirket_Gun",
    ]
)
if "Olay_ID" in forum_audit.columns:
    positive_events = forum_audit.loc[forum_audit["SPK_Etiketi"] == 1].copy()
    positive_event_ids_complete = positive_events["Olay_Isaretli"].all()
    event_id_summary = (
        positive_events.loc[positive_events["Olay_Isaretli"]]
        .groupby(["Hisse_Kodu", "Olay_ID_Normalize"], as_index=False)
        .agg(
            Mesaj_Sayisi=("Mesaj_ID", "size"),
            Benzersiz_Mesaj=("Mesaj_ID", "nunique"),
            Ilk_Zaman=("Zaman_Damgasi", "min"),
            Son_Zaman=("Zaman_Damgasi", "max"),
            Sirket_Gun=("Date", "nunique"),
        )
        .rename(columns={"Olay_ID_Normalize": "Olay_ID"})
        .sort_values(["Hisse_Kodu", "Ilk_Zaman", "Olay_ID"], kind="mergesort")
        .reset_index(drop=True)
    )
    official_events = len(event_id_summary)

cleaning_marker_status = "UNKNOWN"
if "Temizleme_Etiket_Kullandi" in forum_audit.columns:
    cleaning_used_label = forum_audit["Temizleme_Etiket_Kullandi"].astype(str).str.lower().isin(
        ["1", "true", "evet", "yes"]
    )
    cleaning_marker_status = "FAIL" if cleaning_used_label.any() else "PASS"

report = [
    "YENİ VERİ SETİ DENETİM RAPORU",
    f"Forum SHA256: {file_sha256(FORUM_CSV)}",
    f"Piyasa SHA256: {file_sha256(MARKET_CSV)}",
    "",
    f"Forum satırı: {len(forum_audit):,}",
    f"Şirket sayısı: {forum_audit['Hisse_Kodu'].nunique():,}",
    f"Tarih: {forum_audit['Zaman_Damgasi'].min()} -- {forum_audit['Zaman_Damgasi'].max()}",
    f"Benzersiz mesaj: {forum_audit['Mesaj_ID'].nunique():,}",
    f"Yinelenen mesaj: {forum_audit['Mesaj_ID'].duplicated().sum():,}",
    f"Pozitif mesaj: {(forum_audit['SPK_Etiketi'] == 1).sum():,}",
    f"Negatif mesaj: {(forum_audit['SPK_Etiketi'] == 0).sum():,}",
    f"Olay_ID var fakat SPK_Etiketi 1 değil: {len(event_marked_but_not_positive):,}",
    f"SPK_Etiketi 1 fakat Olay_ID boş/0: {len(positive_missing_event_id):,}",
    f"Üç kelimeden kısa mesaj: {forum_audit['Kisa_Mesaj_Hesaplanan'].sum():,}",
    f"Kullanıcısı bilinmeyen mesaj: {len(unknown_audit_rows):,}",
    f"Bilinmeyen kullanıcıda yinelenen kimlik: {unknown_audit_duplicate_ids:,}",
    f"Kimliği bilinen fakat Kullanici_ID boş mesaj: {known_audit_missing_ids:,}",
    f"Geçersiz etiket: {(~forum_audit['SPK_Etiketi'].isin([0, 1])).sum():,}",
    f"Şirket-gün: {len(day_table):,}",
    f"Pozitif şirket-gün: {(day_table['max_label'] == 1).sum():,}",
    f"Negatif şirket-gün: {(day_table['min_label'] == 0).sum():,}",
    f"Karışık şirket-gün: {(day_table['label_count'] > 1).sum():,}",
    f"Pozitif şirket: {len(positive_companies):,}",
    f"Hem 0 hem 1 günü olan pozitif şirket: {len(mixed_companies):,}/{len(positive_companies):,}",
    f"Kesin şirket-olay sayısı: {official_events}",
    "",
    f"Piyasa satırı: {len(market_audit):,}",
    f"Piyasa şirketi: {market_audit['Hisse_Kodu'].nunique():,}",
    f"Piyasa eşleşmesi: {coverage['Piyasa_Eslesmesi'].notna().mean():.2%}",
    f"Anormal piyasa satırı: {len(market_anomalies):,}",
    f"Forum analiz paneline düşen anormal piyasa satırı: {len(panel_market_anomalies):,}",
    f"Sistematik fiyat ölçeği şüpheli şirket-yıl: {len(market_scale_suspicious):,}",
    "",
    f"Test başlangıcı: {LOCKED_TEST_START}",
    f"Development şirket-gün: {len(development_days):,}",
    f"Test şirket-gün: {len(test_days):,}",
    f"Test şirketi: {test_days['Hisse_Kodu'].nunique():,}",
    f"Test pozitif gün: {(test_days['max_label'] == 1).sum():,}",
    f"Test negatif gün: {(test_days['min_label'] == 0).sum():,}",
    "",
    "KRİTİK KONTROLLER",
    f"Etiket tutarlılığı: {'PASS' if (day_table['label_count'] <= 1).all() else 'FAIL'}",
    f"Pozitif şirketlerde 0/1 ayrımı: {'PASS' if len(positive_companies) and len(mixed_companies) == len(positive_companies) else 'FAIL'}",
    f"Testte iki sınıf: {'PASS' if (test_days['max_label'] == 1).any() and (test_days['min_label'] == 0).any() else 'FAIL'}",
    f"Development/test dolu: {'PASS' if len(development_days) and len(test_days) else 'FAIL'}",
    f"Analiz panelinde fiyat anomalisi yok: {'PASS' if panel_market_anomalies.empty else 'FAIL'}",
    f"Sistematik piyasa fiyat ölçeği tutarlı: {'PASS' if market_scale_suspicious.empty else 'FAIL'}",
    f"Pozitif Olay_ID eksiksiz: {'PASS' if positive_event_ids_complete else 'FAIL'}",
    f"Kullanici_Bilinmiyor biçimi: {'PASS' if not unknown_audit_invalid.any() else 'FAIL'}",
    f"Bilinmeyen kullanıcı kimlikleri mesaj-benzersiz: {'PASS' if unknown_audit_duplicate_ids == 0 else 'FAIL'}",
    f"Bilinen kullanıcı kimlikleri dolu: {'PASS' if known_audit_missing_ids == 0 else 'FAIL'}",
    f"Temizleme etiketten bağımsız işaretli: {cleaning_marker_status}",
]

if not event_id_summary.empty:
    report.extend(["", "OLAY_ID DÖKÜMÜ", "Hisse_Kodu | Olay_ID | Mesaj_Sayisi"])
    report.extend(
        f"{row.Hisse_Kodu} | {row.Olay_ID} | {int(row.Mesaj_Sayisi)}"
        for row in event_id_summary.itertuples(index=False)
    )

AUDIT_DIR.mkdir(parents=True, exist_ok=True)
(AUDIT_DIR / "dataset_audit_report.txt").write_text("\n".join(report), encoding="utf-8")
company_summary.to_csv(AUDIT_DIR / "company_summary.csv", index=False)
year_summary.to_csv(AUDIT_DIR / "year_summary.csv", index=False)
short_message_summary.to_csv(AUDIT_DIR / "short_message_summary.csv", index=False)
unknown_user_summary.to_csv(AUDIT_DIR / "unknown_user_summary.csv", index=False)
event_id_summary.to_csv(AUDIT_DIR / "event_id_summary.csv", index=False)
company_day_label_conflicts.to_csv(
    AUDIT_DIR / "company_day_label_conflicts.csv", index=False
)
duplicate_message_ids.to_csv(
    AUDIT_DIR / "duplicate_message_ids.csv", index=False
)
positive_missing_event_id.to_csv(
    AUDIT_DIR / "positive_missing_event_id.csv", index=False
)
event_marked_but_not_positive_summary.to_csv(
    AUDIT_DIR / "event_marked_but_not_positive_summary.csv", index=False
)
market_anomalies.to_csv(AUDIT_DIR / "market_anomalies.csv", index=False)
panel_market_anomalies.to_csv(
    AUDIT_DIR / "panel_market_anomalies.csv", index=False
)
market_scale_summary.to_csv(
    AUDIT_DIR / "market_scale_company_year.csv", index=False
)
market_scale_suspicious.to_csv(
    AUDIT_DIR / "market_scale_suspicious_company_year.csv", index=False
)
unmatched.to_csv(AUDIT_DIR / "unmatched_company_days.csv", index=False)
zip_path = Path(shutil.make_archive(str(AUDIT_DIR), "zip", root_dir=AUDIT_DIR))
print("\n".join(report))
print("\nBANA GÖNDERECEĞİNİZ DOSYA:", zip_path.resolve())
print("Model henüz çalıştırılmadı.")

# %% HÜCRE 4 — TGN-MIL MODELİ
"""HÜCRE 4/5 — Sızıntısız TGN--MIL eğitim, değerlendirme ve yeniden kullanım bloğu.

Bu blok model seçimini kilitli kronolojik testten ayırır; gece yarısını ve veri
bölme sınırlarını aşabilen fakat yalnız geçmişe bakan kenarları korur. TGN--MIL
modelini yalnız metin ve statik komşuluk temelleriyle aynı şirket-gün panelinde
karşılaştırır. Karar eşikleri yalnız validasyon verisinden seçilir. Ham ve
filtrelenmiş duyarlılık stratejileri aynı fiyat, tarih ve maliyet koşullarında
backtest edilir.

Eğitim sırasında ilerleme ve kalan süre raporlanır; her seed checkpoint olarak,
en iyi validasyon modeli ayrıca ``best_model.pt`` olarak kaydedilir. Kayıtlı model
``SavedModelPredictor`` sınıfıyla etiketsiz yeni forum verisine uygulanabilir.

ÖNEMLİ: Kod, daha önce incelenmiş bir test dönemini yeniden "görülmemiş" hâle
getiremez. Doğrulayıcı koşuda ``LOCKED_TEST_START`` gerçekten dokunulmamış bir
dönem olmalı ve kullanıcı bunu ``CONFIRM_UNSEEN_TEST=True`` ile açıkça onaylamalıdır.
"""


import argparse
import hashlib
import json
import logging
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


@dataclass(frozen=True)
class ExperimentConfig:
    """Deneyin tekrarlanabilir hiperparametre ve çalışma ayarlarını taşır.

    Bu değişmez veri sınıfı; rassal başlangıçları, rolling-origin fold sayısını,
    eğitim epochlarını, kenar çıkarma eşiklerini, işlem maliyetlerini ve bootstrap
    tekrarlarını tek yerde toplar. Checkpoint ve manifest içine yazıldığı için
    bir modelin hangi ayarlarla üretildiği daha sonra yeniden kurulabilir.

    Çıktı üretmez; boru hattındaki fonksiyonlara salt-okunur yapılandırma nesnesi
    olarak aktarılır.
    """
    seeds: Tuple[int, ...] = _env_int_tuple(
        "V4_SEEDS", (42, 123, 2024, 555, 999)
    )
    parameter_grid: Tuple[Tuple[int, float], ...] = (
        (50, 0.00),
        (50, 0.10),
        (50, 0.20),
        (75, 0.00),
        (75, 0.10),
        (75, 0.20),
    )
    soft_filter_alphas: Tuple[float, ...] = (0.5, 1.0, 2.0)
    round_trip_costs: Tuple[float, ...] = (0.0, 0.002, 0.004, 0.008)
    selection_round_trip_cost: float = 0.004
    rolling_folds: int = 3
    epochs: int = _env_int("V4_EPOCHS", 3)
    revised_epochs: int = _env_int("V4_REVISED_EPOCHS", 8)
    learning_rate: float = _env_float("V4_LEARNING_RATE", 1e-3)
    max_length: int = 128
    feature_batch_size: int = 64
    time_window_seconds: int = _env_int("V4_TIME_WINDOW_SECONDS", 3600)
    semantic_windows_seconds: Tuple[int, ...] = (60, 300, 900, 3600)
    cosine_threshold: float = _env_float("V4_COSINE_THRESHOLD", 0.85)
    top_k: int = _env_int("V4_TOP_K", 3)
    bootstrap_repetitions: int = 1000
    optimization_batch_days: int = 32
    pu_unlabeled_weight: float = 0.25
    outer_test_years: Tuple[int, ...] = OUTER_TEST_YEARS
    data_variants: Tuple[str, ...] = DATA_VARIANTS
    minimum_test_positive_days: int = 20
    minimum_test_negative_days: int = 20
    top_fraction: float = 0.20
    activity_feature_dim: int = 4
    day_activity_feature_dim: int = 13
    experiment_scope: str = EXPERIMENT_SCOPE


@dataclass
class CompanyStream:
    """Tek bir şirketin kronolojik forum akışını ve önhesaplanmış özelliklerini tutar.

    ``frame`` temizlenmiş iletileri; ``bert_features`` metin temsillerini;
    ``sbert_features`` benzerlik temsillerini; ``events_by_date`` geçmişe-bakan
    çizge olaylarını; ``indices_by_date`` günlük ileti indekslerini taşır.
    ``first_index_by_user`` yeni kullanıcı belleğinin başlangıç özelliğini bulur.

    Bu yapı eğitim sırasında tekrar tokenizasyon ve kenar çıkarımı yapılmasını
    önleyerek aynı şirket verisini bütün fold ve seed'lerde yeniden kullanır.
    """
    company: str
    frame: pd.DataFrame
    bert_features: torch.Tensor
    sbert_features: torch.Tensor
    events_by_date: Mapping[object, List[dict]]
    indices_by_date: Mapping[object, List[int]]
    first_index_by_user: Mapping[str, int]
    activity_features: torch.Tensor
    day_activity_features: Mapping[object, torch.Tensor]
    experiment_profile: str


def profile_uses_short_edges(profile: str) -> bool:
    return profile in {
        "full_v4", "keep_all", "keep_all_drop_unknown", "activity_controlled_tgn",
        "tgn_no_time_encoding", "tgn_daily_memory_reset",
    } or profile in DOMINANT_EVENT_EXCLUSIONS


def profile_uses_activity(profile: str) -> bool:
    return profile in {"revised_tgn", "revised_tgn_long", "revised_no_gate"}


def profile_uses_gate(profile: str) -> bool:
    return profile in {"revised_tgn", "revised_tgn_long", "revised_no_activity"}


def profile_uses_day_activity(profile: str) -> bool:
    """Güçlü temel modelle aynı 13 şirket-gün aktivite kovaryatını kullanan profil."""
    return profile == "activity_controlled_tgn"


def profile_uses_time_encoding(profile: str) -> bool:
    """Zaman kodu ablasyonu dışında öğrenilebilir zaman temsilini etkinleştirir."""
    return profile != "tgn_no_time_encoding"


def profile_carries_memory_across_days(profile: str) -> bool:
    """Günlük bellek sıfırlama ablasyonu dışında kullanıcı durumunu ileri taşır."""
    return profile != "tgn_daily_memory_reset"


def profile_epochs(profile: str, config: ExperimentConfig) -> int:
    return config.revised_epochs if profile == "revised_tgn_long" else config.epochs


def profile_loss_modes(profile: str, config: ExperimentConfig) -> Tuple[str, ...]:
    """Çalışma aşamasına göre bir profilin kayıp türlerini döndürür."""
    if QUICK_SMOKE_TEST:
        return ("bce", "nnpu") if profile == "keep_all" else ("bce",)
    if config.experiment_scope == "primary":
        return ("bce",)
    if config.experiment_scope == "priority":
        if profile == "keep_all":
            return ("nnpu",)
        return ("bce", "nnpu") if profile == "min3_ablation" else ("bce",)
    return (
        ("bce", "nnpu")
        if profile in {"keep_all", "min3_ablation"}
        else ("bce",)
    )


def experiment_profile_table(config: ExperimentConfig) -> pd.DataFrame:
    rows = []
    for profile in config.data_variants:
        rows.append(
            {
                "Experiment_Profile": profile,
                "Primary_Reference": profile in {"full_v4", "keep_all"},
                "Keep_Short_Messages": profile != "min3_ablation",
                "Keep_Unknown_User_Messages": profile != "keep_all_drop_unknown",
                "Allow_Exact_Short_Edges": profile_uses_short_edges(profile),
                "Causal_Activity_Features": profile_uses_activity(profile),
                "Company_Day_Activity_Control": profile_uses_day_activity(profile),
                "Use_Time_Encoding": profile_uses_time_encoding(profile),
                "Carry_Memory_Across_Days": profile_carries_memory_across_days(profile),
                "Gated_Text_TGN_Fusion": profile_uses_gate(profile),
                "Epochs": profile_epochs(profile, config),
                "Loss_Modes": ",".join(profile_loss_modes(profile, config)),
                "Excluded_Dominant_Event": (
                    "::".join(DOMINANT_EVENT_EXCLUSIONS[profile][:2])
                    if profile in DOMINANT_EVENT_EXCLUSIONS
                    else ""
                ),
                "Interpretation": (
                    "final full-data reference"
                    if profile == "full_v4"
                    else "primary historical reference"
                    if profile == "keep_all"
                    else "dominant-event exclusion robustness"
                    if profile in DOMINANT_EVENT_EXCLUSIONS
                    else "temporal component ablation"
                    if profile in {"tgn_no_time_encoding", "tgn_daily_memory_reset"}
                    else "secondary exploratory ablation"
                ),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ModelBundle:
    """Birlikte eğitilen ana model ve karşılaştırma modellerini paketler.

    Paket; olay güdümlü TGN, dikkat tabanlı MIL başlığı, yalnız metin modeli ve
    statik komşuluk temel modelini içerir. Tek nesne kullanılması eğitim,
    değerlendirme, checkpoint kaydı ve yeniden yükleme kodunu sadeleştirir.
    """
    tgn: nn.Module
    mil: nn.Module
    text_baseline: nn.Module
    graph_baseline: nn.Module


def format_duration(seconds: Optional[float]) -> str:
    """Saniye değerini kullanıcı dostu saat/dakika/saniye metnine çevirir.

    İlerleme göstergelerinin geçen ve kalan süre alanlarında kullanılır. Geçersiz,
    sonsuz veya henüz hesaplanamayan değerlerde yanıltıcı sayı göstermek yerine
    ``hesaplanıyor`` metnini döndürür.
    """
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "hesaplanıyor"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d} sa {minutes:02d} dk {seconds:02d} sn"
    if minutes:
        return f"{minutes:d} dk {seconds:02d} sn"
    return f"{seconds:d} sn"


class ProgressReporter:
    """Notebook ve log dosyasında yüzde, hız ve tahmini kalan süre gösterir.

    İlerleme mesajları her döngüde yazılmaz; belirlenen süre dolduğunda veya
    önemli yüzdelik eşikler geçildiğinde yazılır. Böylece uzun çalışmalarda
    çıktı okunabilir kalır.
    """

    def __init__(
        self,
        label: str,
        total: int,
        log_interval_seconds: float = 20.0,
        bar_width: int = 24,
    ) -> None:
        """Yeni bir ilerleme sayacı başlatır.

        ``label`` log satırındaki aşama adıdır; ``total`` beklenen toplam işlem
        sayısıdır. ``log_interval_seconds`` zaman bazlı raporlama sıklığını,
        ``bar_width`` görsel çubuğun genişliğini belirler. Başlangıç zamanını ve
        ilk sayaç durumunu kaydeder, ancak hesaplama yapmaz.
        """
        self.label = label
        self.total = max(0, int(total))
        self.log_interval_seconds = float(log_interval_seconds)
        self.bar_width = int(bar_width)
        self.started_at = time.perf_counter()
        self.last_log_at = self.started_at
        self.completed = 0
        self.last_percent_bucket = -1
        logging.info("%s başladı: toplam %d işlem", self.label, self.total)

    def update(
        self,
        increment: int = 1,
        detail: str = "",
        force: bool = False,
    ) -> None:
        """Tamamlanan iş miktarını artırıp gerektiğinde ETA logu üretir.

        Her çağrıda sayaç güncellenir; çıktı yalnız süre aralığı dolduğunda,
        yüzde 5'lik yeni bir dilime geçildiğinde, iş bittiğinde veya ``force``
        istendiğinde yazılır. Ortalama işlem hızından kalan süre tahmin edilir.

        Çıktı: Konsol ve ``experiment.log`` içine yüzde, sayaç, hız, geçen süre,
        tahmini kalan süre ve isteğe bağlı ``detail`` bilgisini yazar.
        """
        self.completed = min(self.total, self.completed + int(increment))
        now = time.perf_counter()
        elapsed = now - self.started_at
        fraction = self.completed / self.total if self.total else 1.0
        percent_bucket = int(fraction * 20)  # Her yüzde 5'lik ilerlemede zorunlu log.
        should_log = (
            force
            or self.completed >= self.total
            or now - self.last_log_at >= self.log_interval_seconds
            or percent_bucket > self.last_percent_bucket
        )
        if not should_log:
            return
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = (
            (self.total - self.completed) / rate
            if rate > 0 and self.completed < self.total
            else 0.0
        )
        filled = int(round(self.bar_width * fraction))
        bar = "█" * filled + "░" * (self.bar_width - filled)
        detail_text = f" | {detail}" if detail else ""
        logging.info(
            "%s [%s] %6.2f%% (%d/%d) | geçen %s | kalan ~%s | %.2f işlem/sn%s",
            self.label,
            bar,
            fraction * 100.0,
            self.completed,
            self.total,
            format_duration(elapsed),
            format_duration(remaining),
            rate,
            detail_text,
        )
        self.last_log_at = now
        self.last_percent_bucket = percent_bucket

    def finish(self, detail: str = "") -> None:
        """İlerleme sayacını yüzde 100'e tamamlar ve son durum satırını yazar.

        Döngü erken sona ermiş veya toplam sıfır olsa bile tek bir kesin kapanış
        mesajı üretilmesini sağlar. Hesaplama sonucunu değiştirmez.
        """
        if self.completed < self.total:
            self.update(self.total - self.completed, detail=detail, force=True)
        elif self.total == 0:
            self.update(0, detail=detail, force=True)


def configure_logging(output_dir: Path) -> None:
    """Konsol ve dosya loglamasını deney çıktı dizini için yapılandırır.

    Çıktı dizinini oluşturur ve bütün INFO/uyarı mesajlarını hem ekrana hem de
    ``experiment.log`` dosyasına gönderir. ``force=True`` notebook'ta önceki
    hücrelerden kalan logging handler'larının yinelenen satır üretmesini önler.

    Girdi: Deney sonuçlarının yazılacağı dizin.
    Yan etki: Dizin ve log dosyası oluşturur; global logging ayarını değiştirir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "experiment.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def set_seed(seed: int) -> None:
    """Python, NumPy ve PyTorch rassallığını aynı seed ile sabitler.

    Amaç, aynı veri ve yapılandırmayla yinelenen koşuların mümkün olduğunca aynı
    ağırlık başlangıçlarını ve örneklemeleri üretmesidir. CUDA ve cuDNN için de
    deterministik seçenekler etkinleştirilir. Donanım/kütüphane farkları nedeniyle
    bit düzeyinde özdeşlik her platformda garanti edilmez.

    Girdi: Tamsayı rassal başlangıç değeri. Çıktı üretmez.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file_sha256(path: Path) -> str:
    """Model bölümünde kullanılan dosyanın SHA-256 parmak izini döndürür.

    Audit bölümündeki eş adlı yardımcıyla aynı amacı taşır; checkpoint ve deney
    manifestine veri sürümünü yazmak için ``Path`` arayüzüyle blok bazında okur.
    Girdi dosya yolu, çıktı 64 karakterlik onaltılık karma metnidir.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    """Model girdisinin zorunlu sütunları içerdiğini erken aşamada doğrular.

    Eksik alanlar alfabetik listelenerek ``ValueError`` üretilir. Böylece modelin
    yanlış kolonlarla çalışıp sessiz veri kaybı veya hatalı etiket üretmesi
    engellenir. Başarılı durumda değer döndürmez.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} eksik sütunlar içeriyor: {missing}")


def prepare_forum(path: Path, require_labels: bool = True) -> pd.DataFrame:
    """Forum CSV'sini eğitim veya etiketsiz tahmin kullanımı için hazırlar.

    Zorunlu sütunları denetler, zaman damgalarını ayrıştırır, eksik temel alanları
    çıkarır, şirket/kullanıcı kodlarını standartlaştırır ve ``Date`` ile Unix
    saniyesi alanlarını üretir. İletiler şirket, zaman ve ileti kimliğine göre
    kararlı biçimde sıralanır. Aynı şirket-günde birden fazla etiket varsa veri
    sızıntısı ve hedef belirsizliğini önlemek için işlem durdurulur.

    ``require_labels=False`` kayıtlı modelle etiketsiz tahmine izin verir; eksik
    etiket alanı yalnız teknik uyumluluk için sıfırla doldurulur ve tahmin girdisi
    olarak kullanılmaz. Çıktı temizlenmiş kronolojik DataFrame'dir.
    """
    frame = read_csv_auto(path)
    required = [
        "Mesaj_ID",
        "Alinti_Mesaj_ID",
        "Kullanici_ID",
        "Kullanici_Bilinmiyor",
        "Mesaj_Metni",
        "Zaman_Damgasi",
        "Hisse_Kodu",
    ]
    if require_labels:
        required.extend(["SPK_Etiketi", "Olay_ID"])
    require_columns(frame, required, "Forum verisi")
    if "SPK_Etiketi" not in frame.columns:
        # Tahmin verisinde gerçek etiket bilinmeyebilir; bu alan model girdisi değildir.
        frame["SPK_Etiketi"] = 0
    frame["Zaman_Damgasi"] = pd.to_datetime(frame["Zaman_Damgasi"], errors="coerce")
    frame = frame.dropna(subset=["Zaman_Damgasi", "Mesaj_Metni", "Hisse_Kodu", "Mesaj_ID"])
    if require_labels:
        before_cutoff = len(frame)
        start = pd.Timestamp(ANALYSIS_START)
        end = pd.Timestamp(ANALYSIS_END)
        frame = frame.loc[frame["Zaman_Damgasi"].between(start, end)].copy()
        clipped = before_cutoff - len(frame)
        if clipped:
            logging.warning(
                "Tarihsel analiz aralığı dışında kalan %d forum iletisi çıkarıldı: %s -- %s",
                clipped,
                start,
                end,
            )
        if frame.empty:
            raise ValueError("Tarih kesimi sonrasında forum verisi boş kaldı.")
    frame["Hisse_Kodu"] = frame["Hisse_Kodu"].astype(str).str.strip().str.upper()
    unknown_raw = frame["Kullanici_Bilinmiyor"].astype(str).str.strip().str.casefold()
    unknown_map = {
        "1": True, "true": True, "evet": True, "yes": True,
        "0": False, "false": False, "hayir": False, "hayır": False, "no": False,
    }
    invalid_unknown = ~unknown_raw.isin(unknown_map)
    if bool(invalid_unknown.any()):
        examples = sorted(unknown_raw[invalid_unknown].dropna().unique().tolist())[:8]
        raise ValueError(
            "Kullanici_Bilinmiyor yalnız 0/1 veya true/false içermelidir. "
            f"Geçersiz örnekler: {examples}"
        )
    frame["Kullanici_Bilinmiyor"] = unknown_raw.map(unknown_map).astype(bool)
    user_text = frame["Kullanici_ID"].fillna("").astype(str).str.strip()
    missing_known = (~frame["Kullanici_Bilinmiyor"]) & user_text.eq("")
    if bool(missing_known.any()):
        raise ValueError(
            "Kullanici_Bilinmiyor=0 olan satırlarda Kullanici_ID boş bırakılamaz."
        )
    # Bilinmeyen hesaplar aynı kişiymiş gibi birleştirilmez. Kimlik, şirket ve
    # mesaj anahtarından deterministik olarak üretilir ve yalnız teknik satır
    # takibi için kullanılır; kullanıcı düğümü olarak çizgeye alınmaz.
    generated_unknown = (
        "UNKNOWN_MSG_"
        + frame["Hisse_Kodu"].astype(str)
        + "_"
        + frame["Mesaj_ID"].astype(str)
    )
    frame["Kullanici_ID"] = user_text.where(
        ~frame["Kullanici_Bilinmiyor"], generated_unknown
    )
    frame["Mesaj_Metni"] = (
        frame["Mesaj_Metni"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    )
    frame = frame[frame["Mesaj_Metni"].ne("")].copy()
    frame["Kelime_Sayisi"] = frame["Mesaj_Metni"].str.split().str.len().astype(int)
    frame["Kisa_Mesaj"] = frame["Kelime_Sayisi"] < 3
    frame["Normalize_Metin"] = frame["Mesaj_Metni"].str.casefold()
    if "URL_Hash" not in frame.columns:
        frame["URL_Hash"] = ""
    frame["URL_Hash"] = frame["URL_Hash"].fillna("").astype(str).str.strip()
    frame["Date"] = frame["Zaman_Damgasi"].dt.date
    frame["Year"] = frame["Zaman_Damgasi"].dt.year.astype(int)
    frame["Hour"] = frame["Zaman_Damgasi"].dt.hour.astype(int)
    frame["After_Hours"] = (
        (frame["Hour"] < 9) | (frame["Hour"] >= 18)
    ).astype(int)
    frame["Weekend"] = (
        frame["Zaman_Damgasi"].dt.dayofweek >= 5
    ).astype(int)
    quote_values = frame["Alinti_Mesaj_ID"].fillna("").astype(str).str.strip()
    frame["Has_Quote"] = (
        ~quote_values.isin(["", "0", "nan", "None"])
    ).astype(int)
    frame["Has_URL"] = (
        ~frame["URL_Hash"].isin(["", "0", "nan", "None"])
    ).astype(int)
    frame["Timestamp_Sec"] = frame["Zaman_Damgasi"].astype("int64") // 10**9
    frame["SPK_Etiketi"] = pd.to_numeric(
        frame["SPK_Etiketi"], errors="raise"
    ).astype(int)
    if require_labels:
        normalized_event_ids = (
            frame["Olay_ID"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        positive_without_event = frame.loc[
            (frame["SPK_Etiketi"] == 1)
            & normalized_event_ids.isin(["", "0"])
        ]
        if not positive_without_event.empty:
            raise ValueError(
                "Pozitif iletilerde Olay_ID eksik veya 0. Olay-bazlı dış test için "
                "bütün pozitif satırlarda gerçek ve kararlı bir Olay_ID bulunmalıdır."
            )
        frame["Olay_ID"] = normalized_event_ids
    if "Temizleme_Etiket_Kullandi" in frame.columns:
        leaked = frame["Temizleme_Etiket_Kullandi"].astype(str).str.lower().isin(
            ["1", "true", "evet", "yes"]
        )
        if leaked.any():
            raise ValueError(
                "Veri sözleşmesi ihlali: temizleme sırasında hedef etiketinin "
                "kullanıldığı işaretlenmiş. V4 bu veriyle çalışmaz."
            )
    frame = frame.sort_values(
        ["Hisse_Kodu", "Timestamp_Sec", "Mesaj_ID"], kind="mergesort"
    ).reset_index(drop=True)

    label_counts = frame.groupby(["Hisse_Kodu", "Date"])["SPK_Etiketi"].nunique()
    inconsistent = label_counts[label_counts > 1]
    if not inconsistent.empty:
        raise ValueError("Aynı şirket-gün içinde birden fazla SPK etiketi bulundu.")
    unknown_rows = frame[frame["Kullanici_Bilinmiyor"]]
    if unknown_rows["Kullanici_ID"].duplicated().any():
        raise ValueError(
            "Bilinmeyen kullanıcı kimlikleri mesaj düzeyinde benzersiz olmalıdır. "
            "Mesaj_ID tekrarlarını denetleyin."
        )
    return frame


def write_unknown_user_audit(forum: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Bilinmeyen kullanıcı mesajlarının yıl ve sınıf dağılımını raporlar."""
    frame = forum.assign(Yil=pd.to_datetime(forum["Date"]).dt.year)
    summary = (
        frame.groupby(["Yil", "SPK_Etiketi"], as_index=False)
        .agg(
            Mesaj=("Mesaj_ID", "size"),
            Bilinmeyen_Mesaj=("Kullanici_Bilinmiyor", "sum"),
            Sirket_Gun=("Date", "nunique"),
        )
    )
    summary["Bilinmeyen_Orani"] = (
        summary["Bilinmeyen_Mesaj"] / summary["Mesaj"].clip(lower=1)
    )
    summary.to_csv(output_dir / "unknown_user_audit_by_year_label.csv", index=False)
    return summary


def prepare_market(path: Path, output_dir: Path) -> pd.DataFrame:
    """Piyasa CSV'sini güvenli T+1 backtest paneline dönüştürür.

    Tarih ve fiyat alanlarını doğrular; eksik, sıfır veya negatif açılış/kapanış
    kayıtlarını çıkarır; şirket-tarih tekrarlarını tekilleştirir ve gün içi brüt
    getiriyi hesaplar. Mutlak getirisi yüzde 50'yi aşan satırlar otomatik olarak
    silinmez; denetlenmek üzere ayrı CSV'ye yazılır.

    Girdiler: Piyasa dosyası ve audit çıktılarının yazılacağı dizin.
    Çıktı: Backtest birleşiminde kullanılacak temiz piyasa DataFrame'i.
    Yan etki: Gerekirse ``market_return_audit_over_50pct.csv`` üretir.
    """
    frame = read_csv_auto(path)
    require_columns(
        frame,
        [
            "Tarih",
            "Hisse_Kodu",
            "T1_Acilis",
            "T1_Kapanis",
            "Gunluk_Risksiz_Getiri",
        ],
        "Piyasa verisi",
    )
    frame["Tarih"] = pd.to_datetime(frame["Tarih"], errors="coerce").dt.date
    analysis_start_date = pd.Timestamp(ANALYSIS_START).date()
    analysis_end_date = pd.Timestamp(ANALYSIS_END).date()
    before_cutoff = len(frame)
    frame = frame.loc[
        frame["Tarih"].notna()
        & frame["Tarih"].between(analysis_start_date, analysis_end_date)
    ].copy()
    clipped = before_cutoff - len(frame)
    if clipped:
        logging.warning(
            "Tarihsel analiz aralığı dışında kalan %d piyasa satırı çıkarıldı: %s -- %s",
            clipped,
            analysis_start_date,
            analysis_end_date,
        )
    for column in ("T1_Acilis", "T1_Kapanis", "Gunluk_Risksiz_Getiri"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["Tarih", "Hisse_Kodu", "T1_Acilis", "T1_Kapanis"])
    frame = frame[(frame["T1_Acilis"] > 0) & (frame["T1_Kapanis"] > 0)]
    frame = frame.drop_duplicates(["Tarih", "Hisse_Kodu"], keep="last")
    frame["Gross_Return"] = frame["T1_Kapanis"] / frame["T1_Acilis"] - 1.0
    suspicious = frame[frame["Gross_Return"].abs() > 0.50]
    if not suspicious.empty:
        suspicious.to_csv(output_dir / "market_return_audit_over_50pct.csv", index=False)
        logging.warning(
            "%d adet |açılış-kapanış getirisi| > %%50 gözlemi bulundu; "
            "kurumsal işlem/fiyat düzeltmesi denetlenmelidir.",
            len(suspicious),
        )
    return frame


def write_market_regime_audit(
    forum: pd.DataFrame, market: pd.DataFrame, output_dir: Path, variant: str
) -> pd.DataFrame:
    """Getiri rejimini kaydeder ve sistematik fiyat ölçeği sorunlarını döndürür.

    En az on eşleşmiş gözlemi bulunan bir şirket-yılda mutlak ortalama veya
    medyan gün içi getiri yüzde 8'i aşıyorsa açılış ile kapanış serilerinin aynı
    düzeltme ölçeğinde olmayabileceği kabul edilir. Bu denetim, tekil yüzde 50
    eşiğinin yakalayamadığı süreğen fakat daha küçük ölçek uyumsuzluklarını bulur.
    """
    keys = forum[["Hisse_Kodu", "Date"]].drop_duplicates().rename(
        columns={"Date": "Tarih"}
    )
    panel = keys.merge(market, on=["Hisse_Kodu", "Tarih"], how="inner")
    if panel.empty:
        return pd.DataFrame()
    panel["Yil"] = pd.to_datetime(panel["Tarih"]).dt.year
    company_year = (
        panel.groupby(["Yil", "Hisse_Kodu"], as_index=False)
        .agg(
            Gun=("Gross_Return", "size"),
            Ortalama_Gunici_Getiri=("Gross_Return", "mean"),
            Medyan_Gunici_Getiri=("Gross_Return", "median"),
            Pozitif_Gun_Orani=("Gross_Return", lambda values: float((values > 0).mean())),
        )
    )
    company_year.insert(0, "Data_Variant", variant)
    company_year.to_csv(output_dir / "market_regime_company_year.csv", index=False)
    suspicious = company_year[
        (company_year["Gun"] >= 10)
        & (
            (company_year["Ortalama_Gunici_Getiri"].abs() > 0.08)
            | (company_year["Medyan_Gunici_Getiri"].abs() > 0.08)
        )
    ].copy()
    suspicious.to_csv(
        output_dir / "market_scale_suspicious_company_year.csv", index=False
    )
    daily = panel.groupby(["Yil", "Tarih"], as_index=False)["Gross_Return"].mean()
    year_rows = []
    for year, year_frame in daily.groupby("Yil"):
        year_rows.append(
            {
                "Data_Variant": variant,
                "Yil": int(year),
                "Gun": len(year_frame),
                "Always_Long_Cumulative_Return": float(
                    (1.0 + year_frame["Gross_Return"]).prod() - 1.0
                ),
                "Mean_Daily_Return": float(year_frame["Gross_Return"].mean()),
                "Positive_Day_Rate": float((year_frame["Gross_Return"] > 0).mean()),
            }
        )
    pd.DataFrame(year_rows).to_csv(
        output_dir / "market_regime_year_summary.csv", index=False
    )
    return suspicious


def write_label_structure_audit(forum: pd.DataFrame, output_dir: Path) -> bool:
    """Etiketlerin şirket kimliğiyle mekanik olarak karışıp karışmadığını denetler.

    Her şirket için tarih aralığı, satır sayısı, minimum/maksimum etiket ve pozitif
    satır sayısını çıkarır. Hiçbir şirkette hem 0 hem 1 bulunmuyor ve şirketler
    farklı tekil etiketlere ayrılıyorsa şirket kimliği hedefin kusursuz vekili
    sayılır; doğrulayıcı deney bu durumda durdurulabilir.

    Çıktı: Karışma varsa ``True``, yoksa ``False``. Ayrıca
    ``company_label_structure_audit.csv`` dosyasını üretir.
    """
    audit = (
        forum.groupby("Hisse_Kodu", as_index=False)
        .agg(
            First_Date=("Date", "min"),
            Last_Date=("Date", "max"),
            Rows=("Mesaj_ID", "size"),
            Label_Min=("SPK_Etiketi", "min"),
            Label_Max=("SPK_Etiketi", "max"),
            Positive_Rows=("SPK_Etiketi", "sum"),
        )
        .sort_values("Hisse_Kodu")
    )
    audit["Mixed_Labels"] = audit["Label_Min"] != audit["Label_Max"]
    audit.to_csv(output_dir / "company_label_structure_audit.csv", index=False)
    perfectly_company_confounded = bool(
        not audit["Mixed_Labels"].any()
        and audit["Label_Min"].nunique() > 1
        and audit["Label_Max"].nunique() > 1
    )
    if perfectly_company_confounded:
        logging.error(
            "Etiketler şirket kimliğiyle tamamen karışmış: hiçbir şirkette hem 0 hem 1 gün yok."
        )
    return perfectly_company_confounded


def write_event_concentration_audit(
    forum: pd.DataFrame, output_dir: Path, data_variant: str
) -> None:
    """Pozitif olayların yıl içindeki mesaj ve şirket-gün yoğunlaşmasını raporlar.

    Bir olay çok fazla ileti içerdiğinde modelin öğrendiği örüntü ve mikro
    şirket-gün metrikleri o olaya duyarlı olabilir. Bu denetim pozitif satırları
    şirket--Olay_ID düzeyinde toplar; yıl içi mesaj/gün paylarını, HHI'yı ve etkin
    olay sayısını yazar. Yüzde 50'yi aşan paylar otomatik olarak işaretlenir.
    """
    positive = forum.loc[forum["SPK_Etiketi"] == 1].copy()
    if positive.empty:
        _atomic_write_dataframe(
            pd.DataFrame(), output_dir / "positive_event_concentration_detail.csv"
        )
        _atomic_write_dataframe(
            pd.DataFrame(), output_dir / "positive_event_concentration_by_year.csv"
        )
        return
    positive["Olay_ID"] = positive["Olay_ID"].fillna("").astype(str).str.strip()
    positive["Yil"] = pd.to_datetime(positive["Date"]).dt.year.astype(int)
    positive["Event_Key"] = (
        positive["Hisse_Kodu"].astype(str) + "::" + positive["Olay_ID"]
    )
    detail = (
        positive.groupby(["Yil", "Hisse_Kodu", "Olay_ID", "Event_Key"], as_index=False)
        .agg(
            Message_Count=("Mesaj_ID", "size"),
            Positive_Days=("Date", "nunique"),
            First_Date=("Date", "min"),
            Last_Date=("Date", "max"),
        )
    )
    detail["Message_Share_Within_Year"] = detail["Message_Count"] / detail.groupby(
        "Yil"
    )["Message_Count"].transform("sum")
    detail["Positive_Day_Share_Within_Year"] = detail["Positive_Days"] / detail.groupby(
        "Yil"
    )["Positive_Days"].transform("sum")
    detail["Dominates_Message_Volume_50pct"] = (
        detail["Message_Share_Within_Year"] >= 0.50
    )
    detail["Dominates_Positive_Days_50pct"] = (
        detail["Positive_Day_Share_Within_Year"] >= 0.50
    )
    detail.insert(0, "Data_Variant", data_variant)

    year_rows: List[dict] = []
    for year, year_frame in detail.groupby("Yil", sort=True):
        message_hhi = float(np.square(year_frame["Message_Share_Within_Year"]).sum())
        day_hhi = float(np.square(year_frame["Positive_Day_Share_Within_Year"]).sum())
        year_rows.append(
            {
                "Data_Variant": data_variant,
                "Yil": int(year),
                "Positive_Event_Count": int(len(year_frame)),
                "Positive_Message_Count": int(year_frame["Message_Count"].sum()),
                "Positive_Day_Count_Summed_Across_Events": int(
                    year_frame["Positive_Days"].sum()
                ),
                "Largest_Event_Message_Share": float(
                    year_frame["Message_Share_Within_Year"].max()
                ),
                "Largest_Event_Day_Share": float(
                    year_frame["Positive_Day_Share_Within_Year"].max()
                ),
                "Message_HHI": message_hhi,
                "Positive_Day_HHI": day_hhi,
                "Effective_Event_Count_By_Messages": (
                    float(1.0 / message_hhi) if message_hhi > 0 else float("nan")
                ),
                "Effective_Event_Count_By_Days": (
                    float(1.0 / day_hhi) if day_hhi > 0 else float("nan")
                ),
            }
        )
    _atomic_write_dataframe(
        detail.sort_values(["Yil", "Message_Count"], ascending=[True, False]),
        output_dir / "positive_event_concentration_detail.csv",
    )
    _atomic_write_dataframe(
        pd.DataFrame(year_rows),
        output_dir / "positive_event_concentration_by_year.csv",
    )


class FeatureExtractors:
    """İleti metinlerinden BERTurk ve SBERT özelliklerini önceden hesaplar.

    BERTurk'un CLS vektörü model düğüm/olay içeriği olarak, SBERT vektörü ise
    yalnız geçmiş iletiler arasındaki anlamsal benzerlik kenarlarını kurmak için
    kullanılır. Her iki önceden eğitilmiş model değerlendirme modunda tutulur;
    bu çalışma sırasında ağırlıkları ince ayarlanmaz.

    Üretilen tensörler CPU'ya alınarak farklı seed ve fold'larda yeniden
    kullanılabilir; böylece pahalı metin kodlama işlemi tekrarlanmaz.
    """

    def __init__(self, device: torch.device, config: ExperimentConfig):
        """Tokenleştiricileri ve önceden eğitilmiş kodlayıcıları cihaza yükler.

        ``device`` CUDA veya CPU hedefidir; ``config`` maksimum dizi uzunluğu ve
        batch büyüklüğünü sağlar. Model dosyaları ilk kullanımda Hugging Face
        önbelleğinden yüklenir veya indirilir. Ağ ağırlıklarını değiştirmez.
        """
        self.device = device
        self.config = config
        self.bert_tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        self.bert_model = AutoModel.from_pretrained("dbmdz/bert-base-turkish-cased").to(device).eval()
        self.sbert_model = SentenceTransformer(
            "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
        ).to(device)

    def encode(
        self, texts: Sequence[str], progress_label: str = "Metin özellikleri"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bir metin dizisi için BERT ve SBERT tensörlerini batch'ler hâlinde üretir.

        BERT tarafında son gizli katmanın ilk token vektörü, SBERT tarafında
        cümle gömmesi hesaplanır. Gradyan tutulmaz ve sonuçlar CPU tensörleri
        olarak döndürülür. İki kodlayıcı için ayrı ilerleme/ETA mesajları yazar.

        Girdiler: Metin dizisi ve log etiketi.
        Çıktı: ``(bert_features, sbert_features)`` ikilisi; satır sırası girdinin
        ileti sırasıyla aynıdır.
        """
        bert_batches: List[torch.Tensor] = []
        batch_starts = list(range(0, len(texts), self.config.feature_batch_size))
        bert_progress = ProgressReporter(
            f"{progress_label} / BERT", len(batch_starts), log_interval_seconds=15
        )
        for start in batch_starts:
            batch = list(texts[start : start + self.config.feature_batch_size])
            inputs = self.bert_tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
            ).to(self.device)
            with torch.no_grad():
                encoded = self.bert_model(**inputs).last_hidden_state[:, 0, :]
            bert_batches.append(encoded.detach().cpu())
            bert_progress.update(detail=f"ileti {min(start + len(batch), len(texts))}/{len(texts)}")
        bert = torch.cat(bert_batches, dim=0)
        bert_progress.finish()

        sbert_batches: List[torch.Tensor] = []
        sbert_progress = ProgressReporter(
            f"{progress_label} / SBERT", len(batch_starts), log_interval_seconds=15
        )
        for start in batch_starts:
            batch = list(texts[start : start + self.config.feature_batch_size])
            with torch.no_grad():
                encoded = self.sbert_model.encode(
                    batch,
                    batch_size=self.config.feature_batch_size,
                    convert_to_tensor=True,
                    device=str(self.device),
                    show_progress_bar=False,
                ).detach().cpu()
            sbert_batches.append(encoded)
            sbert_progress.update(
                detail=f"ileti {min(start + len(batch), len(texts))}/{len(texts)}"
            )
        sbert_progress.finish()
        sbert = torch.cat(sbert_batches, dim=0)
        return bert, sbert


def build_past_only_events(
    frame: pd.DataFrame,
    sbert_features: torch.Tensor,
    config: ExperimentConfig,
    allow_short_exact_edges: bool = True,
    progress_label: str = "Geçmişe dönük kenar çıkarımı",
) -> Tuple[Mapping[object, List[dict]], set]:
    """Sızıntısız çok ilişkili ve çok ölçekli zamansal kenarları üretir.

    Kenar türleri: 0--3 sırasıyla 1, 5, 15 ve 60 dakikalık anlamsal benzerlik;
    4 doğrudan alıntı; 5 kısa iletide tam normalleştirilmiş metin tekrarı; 6 aynı
    URL karmasıdır. Alıntı hedefi zaman penceresinin dışında da geçmişteyse
    korunur. Diğer ilişkiler en fazla bir saatlik kayan geçmişten ve yalnız farklı
    kullanıcılar arasında kurulur. Aynı hedef birden fazla koşulu sağlarsa
    alıntı, URL, tam metin ve anlamsal ilişki önceliği uygulanır.

    Girdiler: Kronolojik şirket DataFrame'i, satır hizalı SBERT tensörü ve deney
    ayarları. Çıktı: Kaynak gününe göre olay sözlüğü ve en az bir olaya katılan
    ileti indeksleri kümesi. İlerleme/ETA logu üretir.
    """
    events_by_date: MutableMapping[object, List[dict]] = defaultdict(list)
    connected_indices: set = set()
    message_to_index = {message_id: idx for idx, message_id in enumerate(frame["Mesaj_ID"])}
    timestamps = frame["Timestamp_Sec"].to_numpy()
    users = frame["Kullanici_ID"].astype(str).to_numpy()
    unknown_users = frame["Kullanici_Bilinmiyor"].astype(bool).to_numpy()
    progress = ProgressReporter(progress_label, len(frame), log_interval_seconds=20)
    window_start = 0
    event_source_count = 0
    normalized_texts = frame["Normalize_Metin"].astype(str).to_numpy()
    word_counts = frame["Kelime_Sayisi"].astype(int).to_numpy()
    url_hashes = frame["URL_Hash"].astype(str).to_numpy()
    relation_priority = {4: 4, 6: 3, 5: 2, 0: 1, 1: 1, 2: 1, 3: 1}

    def add_interaction(
        container: Dict[int, dict], target_idx: int, relation_type: int, time_diff: int
    ) -> None:
        existing = container.get(target_idx)
        if existing is None or relation_priority[relation_type] > relation_priority[existing["type"]]:
            container[target_idx] = {"type": relation_type, "time_diff": int(time_diff)}

    for source_idx in range(len(frame)):
        interactions: Dict[int, dict] = {}
        if unknown_users[source_idx]:
            progress.update(detail=f"olaylı kaynak {event_source_count}")
            continue
        source_user = users[source_idx]
        quote_id = frame.iloc[source_idx]["Alinti_Mesaj_ID"]
        if pd.notna(quote_id) and quote_id != -1 and quote_id in message_to_index:
            target_idx = message_to_index[quote_id]
            if (
                target_idx < source_idx
                and not unknown_users[target_idx]
                and users[target_idx] != source_user
            ):
                add_interaction(
                    interactions,
                    target_idx,
                    4,
                    int(timestamps[source_idx] - timestamps[target_idx]),
                )

        while (
            window_start < source_idx
            and timestamps[source_idx] - timestamps[window_start]
            > config.time_window_seconds
        ):
            window_start += 1

        if source_idx > window_start:
            candidate_range = np.arange(window_start, source_idx)
            candidate_differences = (
                timestamps[source_idx] - timestamps[candidate_range]
            )
            candidates = candidate_range[
                (candidate_differences > 0)
                & (~unknown_users[candidate_range])
                & (users[candidate_range] != source_user)
            ]
            if len(candidates):
                differences = timestamps[source_idx] - timestamps[candidates]

                # Tek kelimelik ve sembol ağırlıklı kısa iletilerde SBERT yerine
                # farklı kullanıcıların birebir normalleştirilmiş tekrarları kullanılır.
                if (
                    allow_short_exact_edges
                    and word_counts[source_idx] < 2
                    and normalized_texts[source_idx]
                ):
                    exact_candidates = candidates[
                        normalized_texts[candidates] == normalized_texts[source_idx]
                    ][-config.top_k :]
                    for target_idx in exact_candidates:
                        add_interaction(
                            interactions,
                            int(target_idx),
                            5,
                            int(timestamps[source_idx] - timestamps[target_idx]),
                        )

                # Özgün URL veri setine yazılmaz; aynı bağlantıyı temsil eden
                # geri döndürülemez karma varsa koordinasyon ilişkisi kurulur.
                if url_hashes[source_idx]:
                    url_candidates = candidates[
                        url_hashes[candidates] == url_hashes[source_idx]
                    ][-config.top_k :]
                    for target_idx in url_candidates:
                        add_interaction(
                            interactions,
                            int(target_idx),
                            6,
                            int(timestamps[source_idx] - timestamps[target_idx]),
                        )

            if len(candidates) and word_counts[source_idx] >= 2:
                similarities = F.cosine_similarity(
                    sbert_features[source_idx].unsqueeze(0), sbert_features[candidates]
                )
                passed = similarities >= config.cosine_threshold
                if bool(passed.any()):
                    passed_candidates = candidates[passed.numpy()]
                    passed_similarities = similarities[passed]
                    count = min(config.top_k, len(passed_candidates))
                    selected = torch.topk(passed_similarities, count).indices.numpy()
                    for local_idx in selected:
                        target_idx = int(passed_candidates[local_idx])
                        time_diff = int(timestamps[source_idx] - timestamps[target_idx])
                        relation_type = next(
                            (
                                scale_idx
                                for scale_idx, boundary in enumerate(
                                    config.semantic_windows_seconds
                                )
                                if time_diff <= boundary
                            ),
                            len(config.semantic_windows_seconds) - 1,
                        )
                        add_interaction(
                            interactions, target_idx, relation_type, time_diff
                        )

        if interactions:
            event = {
                "source_idx": source_idx,
                "interactions": [
                    (target_idx, values["type"], values["time_diff"])
                    for target_idx, values in interactions.items()
                ],
            }
            source_date = frame.iloc[source_idx]["Date"]
            events_by_date[source_date].append(event)
            connected_indices.add(source_idx)
            connected_indices.update(interactions)
            event_source_count += 1
        progress.update(
            detail=f"olaylı kaynak {event_source_count}"
        )

    progress.finish(detail="kenar çıkarımı tamamlandı")
    return dict(events_by_date), connected_indices


def build_causal_activity_features(frame: pd.DataFrame) -> torch.Tensor:
    """Yalnız geçmiş mesajlardan dört kullanıcı aktivitesi özelliği üretir."""
    previous_messages: MutableMapping[str, int] = defaultdict(int)
    previous_days: MutableMapping[str, set] = defaultdict(set)
    previous_hours: MutableMapping[str, set] = defaultdict(set)
    current_day_counts: MutableMapping[Tuple[str, object], int] = defaultdict(int)
    rows: List[List[float]] = []
    for row in frame.itertuples(index=False):
        if bool(row.Kullanici_Bilinmiyor):
            rows.append([0.0, 0.0, 0.0, 0.0])
            continue
        user = str(row.Kullanici_ID)
        day = row.Date
        hour_bucket = pd.Timestamp(row.Zaman_Damgasi).floor("h")
        rows.append(
            [
                float(np.log1p(previous_messages[user])),
                float(np.log1p(len(previous_days[user]))),
                float(np.log1p(current_day_counts[(user, day)])),
                float(np.log1p(len(previous_hours[user]))),
            ]
        )
        previous_messages[user] += 1
        previous_days[user].add(day)
        previous_hours[user].add(hour_bucket)
        current_day_counts[(user, day)] += 1
    return torch.tensor(rows, dtype=torch.float32)


DAY_ACTIVITY_COLUMNS = (
    "Log_Message_Count",
    "Log_Unique_Users",
    "Unknown_User_Share",
    "Short_Message_Share",
    "Quote_Share",
    "URL_Share",
    "Mean_Word_Count",
    "SD_Word_Count",
    "Unique_Text_Share",
    "Max_User_Message_Share",
    "Hour_Entropy",
    "After_Hours_Share",
    "Weekend",
)


def build_company_day_activity_features(
    frame: pd.DataFrame,
) -> Mapping[object, torch.Tensor]:
    """Aktivite lojitiyle özdeş 13 şirket-gün kovaryatını üretir.

    Özellikler yalnız ilgili şirket-günün iletilerinden hesaplanır. Dış test
    standardizasyonu burada yapılmaz; eğitim günlerinin ortalama ve standart
    sapması ``train_models`` içinde hesaplanarak model tamponlarına kaydedilir.
    """
    keys = ["Hisse_Kodu", "Date"]
    grouped = frame.groupby(keys, sort=False)
    day = grouped.agg(
        Message_Count=("Mesaj_ID", "size"),
        Unique_Users=("Kullanici_ID", "nunique"),
        Unknown_Count=("Kullanici_Bilinmiyor", "sum"),
        Short_Count=("Kisa_Mesaj", "sum"),
        Quote_Count=("Has_Quote", "sum"),
        URL_Count=("Has_URL", "sum"),
        Mean_Word_Count=("Kelime_Sayisi", "mean"),
        SD_Word_Count=("Kelime_Sayisi", "std"),
        Unique_Texts=("Normalize_Metin", "nunique"),
        After_Hours_Count=("After_Hours", "sum"),
        Weekend=("Weekend", "max"),
    ).reset_index()
    user_counts = frame.groupby(keys + ["Kullanici_ID"], sort=False).size()
    max_user = (
        user_counts.groupby(level=[0, 1]).max().rename("Max_User_Count").reset_index()
    )
    hour_counts = (
        frame.groupby(keys + ["Hour"], sort=False).size().rename("N").reset_index()
    )
    totals = hour_counts.groupby(keys)["N"].transform("sum")
    probability = hour_counts["N"] / totals.clip(lower=1)
    hour_counts["EntropyPart"] = -probability * np.log(
        probability.clip(lower=1e-12)
    )
    entropy = hour_counts.groupby(keys, as_index=False)["EntropyPart"].sum()
    entropy["Hour_Entropy"] = entropy["EntropyPart"] / math.log(24)
    day = day.merge(max_user, on=keys, how="left").merge(
        entropy[keys + ["Hour_Entropy"]], on=keys, how="left"
    )
    denominator = day["Message_Count"].clip(lower=1)
    day["Log_Message_Count"] = np.log1p(day["Message_Count"])
    day["Log_Unique_Users"] = np.log1p(day["Unique_Users"])
    day["Unknown_User_Share"] = day["Unknown_Count"] / denominator
    day["Short_Message_Share"] = day["Short_Count"] / denominator
    day["Quote_Share"] = day["Quote_Count"] / denominator
    day["URL_Share"] = day["URL_Count"] / denominator
    day["Unique_Text_Share"] = day["Unique_Texts"] / denominator
    day["Max_User_Message_Share"] = day["Max_User_Count"] / denominator
    day["After_Hours_Share"] = day["After_Hours_Count"] / denominator
    day["SD_Word_Count"] = day["SD_Word_Count"].fillna(0.0)
    day["Hour_Entropy"] = day["Hour_Entropy"].fillna(0.0)
    return {
        row.Date: torch.tensor(
            [float(getattr(row, column)) for column in DAY_ACTIVITY_COLUMNS],
            dtype=torch.float32,
        )
        for row in day.itertuples(index=False)
    }


def training_day_activity_statistics(
    streams: Mapping[str, CompanyStream], train_dates: Sequence[object]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aktivite standardizasyonunu yalnız dış-fold eğitim günlerinden öğrenir."""
    values = [
        streams[company].day_activity_features[day]
        for day, company in chronological_day_keys(streams, train_dates)
    ]
    if not values:
        raise ValueError("Aktivite standardizasyonu için eğitim şirket-günü yok.")
    matrix = torch.stack(values)
    mean = matrix.mean(dim=0)
    std = matrix.std(dim=0, unbiased=False).clamp_min(1e-6)
    return mean, std


def build_streams(
    forum: pd.DataFrame,
    extractor: FeatureExtractors,
    config: ExperimentConfig,
    experiment_profile: str = "keep_all",
) -> Mapping[str, CompanyStream]:
    """Temiz forum tablosunu şirket bazlı yeniden kullanılabilir akışlara dönüştürür.

    Her şirket kronolojik sıralanır; metin özellikleri çıkarılır, geçmişe-bakan
    olaylar kurulur, gün ve kullanıcı indeksleri hazırlanır. Sonuçlar her şirket
    kodunu bir ``CompanyStream`` nesnesine bağlayan sözlükte tutulur.

    Bu aşama deneyin en maliyetli ön işlemlerinden biridir; şirket, BERT, SBERT ve
    kenar oluşturma seviyelerinde ilerleme gösterir. Eğitim etiketi üzerinde
    optimizasyon yapmaz ve test verisine ait geleceği kenarlara taşımaz.
    """
    streams: Dict[str, CompanyStream] = {}
    grouped = list(forum.groupby("Hisse_Kodu", sort=True))
    company_progress = ProgressReporter("Şirket akışlarının hazırlanması", len(grouped))
    for company, company_frame in grouped:
        company_frame = company_frame.sort_values(
            ["Timestamp_Sec", "Mesaj_ID"], kind="mergesort"
        ).reset_index(drop=True)
        logging.info("Özellikler çıkarılıyor: %s (%d ileti)", company, len(company_frame))
        bert, sbert = extractor.encode(
            company_frame["Mesaj_Metni"].astype(str).tolist(),
            progress_label=f"{company} özellik çıkarımı",
        )
        events_by_date, _ = build_past_only_events(
            company_frame,
            sbert,
            config,
            allow_short_exact_edges=profile_uses_short_edges(experiment_profile),
            progress_label=f"{company} zamansal kenarları",
        )
        activity_features = build_causal_activity_features(company_frame)
        day_activity_features = build_company_day_activity_features(company_frame)
        indices_by_date = {
            day: group.index.tolist() for day, group in company_frame.groupby("Date", sort=True)
        }
        first_by_user = (
            company_frame.reset_index()
            .groupby("Kullanici_ID", sort=False)["index"]
            .first()
            .astype(int)
            .to_dict()
        )
        streams[str(company)] = CompanyStream(
            company=str(company),
            frame=company_frame,
            bert_features=bert,
            sbert_features=sbert,
            events_by_date=events_by_date,
            indices_by_date=indices_by_date,
            first_index_by_user=first_by_user,
            activity_features=activity_features,
            day_activity_features=day_activity_features,
            experiment_profile=experiment_profile,
        )
        company_progress.update(detail=f"son şirket: {company}")
    company_progress.finish()
    return streams


def write_edge_audit(
    streams: Mapping[str, CompanyStream], output_path: Path
) -> pd.DataFrame:
    """Çok ilişkili çizgedeki kenar türü ve zaman farkı dağılımını kaydeder."""
    relation_names = {
        0: "semantic_1m",
        1: "semantic_5m",
        2: "semantic_15m",
        3: "semantic_60m",
        4: "quote",
        5: "exact_short_text",
        6: "same_url_hash",
    }
    rows: List[dict] = []
    for company, stream in streams.items():
        for day, events in stream.events_by_date.items():
            for event in events:
                for _, relation_type, time_difference in event["interactions"]:
                    rows.append(
                        {
                            "Hisse_Kodu": company,
                            "Tarih": day,
                            "Relation_Type": relation_type,
                            "Relation_Name": relation_names.get(relation_type, "unknown"),
                            "Time_Difference_Seconds": time_difference,
                        }
                    )
    edge_frame = pd.DataFrame(rows)
    if edge_frame.empty:
        pd.DataFrame(
            columns=[
                "Relation_Type",
                "Relation_Name",
                "Edges",
                "Companies",
                "Days",
                "Median_Time_Difference_Seconds",
            ]
        ).to_csv(output_path, index=False)
        return edge_frame
    summary = (
        edge_frame.groupby(["Relation_Type", "Relation_Name"], as_index=False)
        .agg(
            Edges=("Relation_Type", "size"),
            Companies=("Hisse_Kodu", "nunique"),
            Days=("Tarih", "nunique"),
            Median_Time_Difference_Seconds=("Time_Difference_Seconds", "median"),
        )
    )
    summary.to_csv(output_path, index=False)
    return edge_frame


class TimeEncoder(nn.Module):
    """Olaylar arasındaki saniye farkını öğrenilebilir kosinüs vektörüne kodlar.

    TGN güncellemesine ham zaman farkı yerine sabit boyutlu, periyodik ve
    öğrenilebilir bir temsil verir. Parametreleri ana TGN ile birlikte eğitilir.
    """

    def __init__(self, dimension: int = 16):
        """Zaman kodlama boyutunu belirler ve öğrenilebilir parametreleri başlatır.

        ``dimension`` her olay zaman farkı için üretilecek vektör uzunluğudur.
        Frekans ağırlıkları Xavier dağılımıyla, faz terimleri sıfırla başlatılır;
        her ikisi de TGN kaybı üzerinden öğrenilir.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1, dimension))
        self.bias = nn.Parameter(torch.zeros(1, dimension))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, time_difference: int, device: torch.device) -> torch.Tensor:
        """Tek bir saniye farkını ``dimension`` uzunluğunda kosinüs vektörüne çevirir.

        Girdi geleceğe ait olamaz; kenar çıkarma kodu pozitif geçmiş farklarını
        garanti eder. Çıktı seçilen cihazda bir PyTorch tensörüdür.
        """
        value = torch.tensor([[time_difference]], dtype=torch.float32, device=device)
        return torch.cos(value * self.weight + self.bias).squeeze(0)


class EventDrivenTGN(nn.Module):
    """Zaman damgalı kullanıcı etkileşimlerini sürekli bellekle işleyen TGN çekirdeğidir.

    Kullanıcıların şirket içindeki durumlarını taşır; ileti özelliği, karşı uç
    belleği, zaman kodu ve kenar türünü GRUCell ile birleştirir. Bellek şirketler
    arasında paylaşılmaz, günler arasında kronolojik olarak taşınır. Fonksiyonun
    çıktısı MIL başlığının şirket-gün olasılığı üretmesi için kullanıcı bellekleridir.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        memory_dim: int = 256,
        time_dim: int = 16,
        relation_types: int = 7,
        relation_dim: int = 8,
        use_activity_features: bool = False,
        activity_feature_dim: int = 4,
        use_time_encoding: bool = True,
        carry_memory_across_days: bool = True,
    ):
        """Özellik, kullanıcı belleği ve zaman kodu boyutlarıyla TGN katmanlarını kurar.

        İlk doğrusal katman yeni kullanıcıların BERT özelliğini bellek uzayına
        taşır. GRUCell ise ileti, karşı uç belleği, zaman kodu ve kenar türünden
        oluşan birleşik olay mesajıyla kullanıcı durumunu günceller.
        """
        super().__init__()
        self.use_activity_features = bool(use_activity_features)
        self.activity_feature_dim = int(activity_feature_dim)
        self.use_time_encoding = bool(use_time_encoding)
        self.carry_memory_across_days = bool(carry_memory_across_days)
        self.time_dim = int(time_dim)
        self.time_encoder = TimeEncoder(time_dim)
        self.relation_embedding = nn.Embedding(relation_types, relation_dim)
        self.feature_to_memory = nn.Linear(feature_dim, memory_dim)
        self.memory_updater = nn.GRUCell(
            feature_dim + memory_dim + time_dim + relation_dim
            + (self.activity_feature_dim if self.use_activity_features else 0),
            memory_dim,
        )

    def forward_day(
        self,
        stream: CompanyStream,
        day: object,
        global_memory: MutableMapping[str, Dict[str, torch.Tensor]],
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[str], Mapping[str, int], List[dict]]:
        """Bir şirket-günün olaylarını kronolojik kullanıcı belleğine uygular.

        Olay varsa etkileşim uçlarındaki kullanıcılar, olay yoksa o gün ileti
        yazan kullanıcılar torbaya alınır. Yeni kullanıcı belleği ilk kronolojik
        BERT özelliğinden başlatılır. Her etkileşimde iki ucun olay-öncesi kopyası
        kullanılarak eşzamanlı güncelleme yapılır; gün sonu bellekleri ayrıştırılıp
        ``global_memory`` içine taşınır.

        Çıktılar: MIL'e verilecek bellek matrisi, kullanıcı sırası, kullanıcı-konum
        sözlüğü ve günün olay listesi. Fonksiyon bellek sözlüğünü günceller.
        """
        current_indices = stream.indices_by_date[day]
        known_current_indices = [
            index
            for index in current_indices
            if not bool(stream.frame.iloc[index]["Kullanici_Bilinmiyor"])
        ]
        day_events = stream.events_by_date.get(day, [])
        endpoint_indices: set = set()
        if day_events:
            for event in day_events:
                endpoint_indices.add(event["source_idx"])
                endpoint_indices.update(target for target, _, _ in event["interactions"])
        else:
            endpoint_indices.update(known_current_indices)

        if not endpoint_indices:
            # Gün yalnız kullanıcı kimliği bilinmeyen mesajlardan oluşuyorsa
            # kalıcı/sahte kullanıcı belleği yaratılmaz. Şirket-gün kayıptan
            # düşmesin diye o günün ortalama metni tek kullanımlık bağlam
            # temsiline dönüştürülür; kullanıcı skoru üretilmez.
            day_features = stream.bert_features[current_indices].to(device)
            context_memory = torch.relu(
                self.feature_to_memory(day_features.mean(dim=0))
            ).unsqueeze(0)
            return context_memory, [], {}, day_events

        endpoint_users = list(
            dict.fromkeys(stream.frame.iloc[sorted(endpoint_indices)]["Kullanici_ID"].astype(str))
        )
        user_to_position = {user: idx for idx, user in enumerate(endpoint_users)}
        if self.carry_memory_across_days:
            company_memory = global_memory.setdefault(stream.company, {})
        else:
            # Ablasyon: gün içindeki olay sırası ve GRU güncellemeleri korunur,
            # fakat önceki takvim gününden taşınan kullanıcı durumu kullanılmaz.
            company_memory = {}
            global_memory[stream.company] = company_memory
        user_memory: Dict[str, torch.Tensor] = {}
        for user in endpoint_users:
            if user in company_memory:
                user_memory[user] = company_memory[user].detach().clone()
            else:
                first_idx = stream.first_index_by_user[user]
                first_feature = stream.bert_features[first_idx].to(device)
                user_memory[user] = torch.relu(self.feature_to_memory(first_feature))

        for event in day_events:
            source_idx = event["source_idx"]
            source_user = str(stream.frame.iloc[source_idx]["Kullanici_ID"])
            source_feature = stream.bert_features[source_idx].to(device)
            for target_idx, edge_type, time_difference in event["interactions"]:
                target_user = str(stream.frame.iloc[target_idx]["Kullanici_ID"])
                source_pre = user_memory[source_user].clone()
                target_pre = user_memory[target_user].clone()
                encoded_time = (
                    self.time_encoder(time_difference, device)
                    if self.use_time_encoding
                    else torch.zeros(self.time_dim, dtype=torch.float32, device=device)
                )
                encoded_type = self.relation_embedding(
                    torch.tensor(edge_type, dtype=torch.long, device=device)
                )
                activity = (
                    stream.activity_features[source_idx].to(device)
                    if self.use_activity_features else None
                )
                source_parts = [source_feature, target_pre, encoded_time, encoded_type]
                target_parts = [source_feature, source_pre, encoded_time, encoded_type]
                if activity is not None:
                    source_parts.append(activity)
                    target_parts.append(activity)
                source_input = torch.cat(source_parts).unsqueeze(0)
                target_input = torch.cat(target_parts).unsqueeze(0)
                user_memory[source_user] = self.memory_updater(
                    source_input, source_pre.unsqueeze(0)
                ).squeeze(0)
                user_memory[target_user] = self.memory_updater(
                    target_input, target_pre.unsqueeze(0)
                ).squeeze(0)

        for user in endpoint_users:
            company_memory[user] = user_memory[user].detach().clone()
        stacked = torch.stack([user_memory[user] for user in endpoint_users])
        return stacked, endpoint_users, user_to_position, day_events


class MILClassifier(nn.Module):
    """Kullanıcı belleklerinden şirket-gün ve kullanıcı şüphe skorları üretir.

    Çoklu örnek öğrenmesi yaklaşımında şirket-gün bir torba, kullanıcılar ise
    torbadaki örneklerdir. Dikkat ağırlıkları hangi kullanıcı durumlarının günlük
    karara daha çok katkı verdiğini öğrenirken örnek sınıflandırıcısı kullanıcı
    düzeyinde zayıf denetimli logit üretir.
    """

    def __init__(
        self,
        memory_dim: int = 256,
        feature_dim: int = 768,
        use_gated_text_fusion: bool = False,
        use_day_activity_features: bool = False,
        day_activity_feature_dim: int = 13,
        day_activity_mean: Optional[torch.Tensor] = None,
        day_activity_std: Optional[torch.Tensor] = None,
    ):
        """Dikkat ağı ve kullanıcı örnek sınıflandırıcısını kurar.

        Her iki alt ağ aynı kullanıcı belleklerini görür ancak farklı amaç taşır:
        dikkat ağı torba içi göreli katkıyı, örnek sınıflandırıcısı kullanıcı
        şüphe logitini üretir. Boyutlar ana TGN bellek boyutuyla uyumlu olmalıdır.
        """
        super().__init__()
        self.use_gated_text_fusion = bool(use_gated_text_fusion)
        self.use_day_activity_features = bool(use_day_activity_features)
        self.attention = nn.Sequential(
            nn.Linear(memory_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        self.instance_classifier = nn.Sequential(
            nn.Linear(memory_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        if self.use_gated_text_fusion:
            self.text_projection = nn.Sequential(nn.Linear(feature_dim, memory_dim), nn.ReLU())
            self.fusion_gate = nn.Sequential(
                nn.Linear(memory_dim * 2, memory_dim), nn.Sigmoid()
            )
            self.fused_classifier = nn.Sequential(
                nn.Linear(memory_dim, 64), nn.ReLU(), nn.Linear(64, 1)
            )
        if self.use_day_activity_features:
            mean = (
                day_activity_mean.detach().clone().float()
                if day_activity_mean is not None
                else torch.zeros(day_activity_feature_dim, dtype=torch.float32)
            )
            std = (
                day_activity_std.detach().clone().float()
                if day_activity_std is not None
                else torch.ones(day_activity_feature_dim, dtype=torch.float32)
            )
            self.register_buffer("day_activity_mean", mean)
            self.register_buffer("day_activity_std", std.clamp_min(1e-6))
            # Aktivite terimi mevcut TGN logitine eklenir. Böylece yeni model,
            # aktivite katsayıları sıfırken tam olarak standart TGN başlığına iner.
            self.day_activity_classifier = nn.Linear(day_activity_feature_dim, 1)

    def forward(
        self,
        memories: torch.Tensor,
        day_features: Optional[torch.Tensor] = None,
        day_activity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Kullanıcı bellek matrisini günlük logit ve kullanıcı logitlerine dönüştürür.

        Dikkat skorları kullanıcı boyutunda softmax ile normalize edilir; günlük
        logit, kullanıcı logitlerinin dikkat ağırlıklı toplamıdır. Çıktı henüz
        olasılık değildir; sigmoid değerlendirme aşamasında uygulanır.
        """
        user_logits = self.instance_classifier(memories).squeeze(-1)
        weights = torch.softmax(self.attention(memories).squeeze(-1), dim=0)
        if self.use_gated_text_fusion:
            if day_features is None:
                raise ValueError("Kapılı MIL için day_features zorunludur.")
            temporal_pool = torch.sum(weights.unsqueeze(-1) * memories, dim=0)
            text_pool = self.text_projection(day_features.mean(dim=0))
            gate = self.fusion_gate(torch.cat([temporal_pool, text_pool]))
            fused = gate * temporal_pool + (1.0 - gate) * text_pool
            day_logit = self.fused_classifier(fused).squeeze()
        else:
            day_logit = torch.sum(weights * user_logits)
        if self.use_day_activity_features:
            if day_activity is None:
                raise ValueError("Aktivite-kontrollü TGN için day_activity zorunludur.")
            standardised = (
                day_activity.to(memories.device) - self.day_activity_mean
            ) / self.day_activity_std
            day_logit = day_logit + self.day_activity_classifier(
                standardised
            ).squeeze()
        return day_logit, user_logits


class TextOnlyBaseline(nn.Module):
    """Çizge bilgisi kullanmadan günlük ortalama BERT özelliğini sınıflandırır.

    TGN'nin ek katkısını ölçmek için aynı şirket-gün etiketleri üzerinde çalışan
    basit metin temel modelidir. Kullanıcı kimliği, zaman farkı ve etkileşim
    kenarlarını görmez.
    """

    def __init__(self, feature_dim: int = 768):
        """Ortalama metin temsilini ikili logite eşleyen iki katmanlı ağı kurar.

        Girdi boyutu BERTurk vektör boyutuna eşittir; ara ReLU katmanı 128 birim,
        çıkış ise BCEWithLogitsLoss ile eğitilecek tek logittir.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(self, day_features: torch.Tensor) -> torch.Tensor:
        """Bir günün ileti özelliklerini ortalayıp tek şirket-gün logiti döndürür.

        ``day_features`` biçimi ileti sayısı çarpı özellik boyutudur. İletiler eşit
        ağırlıkla havuzlanır; kullanıcı ve etkileşim bilgisi kullanılmaz. Çıktıya
        sigmoid uygulanmamıştır.
        """
        return self.classifier(day_features.mean(dim=0)).squeeze()


class StaticNeighborhoodBaseline(nn.Module):
    """Zaman kodu ve bellek olmadan tek-atlamalı statik komşuluk temeli üretir.

    Gün içindeki çıkarılmış kenarları yalnız komşuluk ilişkisi olarak kullanır;
    olay sırası, zaman farkı ve günler arası kullanıcı durumu yoktur. Böylece
    TGN'nin zamansal bellek bileşeninin statik çizgeye göre ek değeri sınanır.
    """

    def __init__(self, feature_dim: int = 768):
        """Kendi özellik ve komşu ortalaması birleşimini sınıflandıran ağı kurar.

        Bir ileti için iki BERT vektörü yan yana getirildiğinden giriş boyutu
        ``2 * feature_dim`` olur. Ara 128 birimli ReLU katmanı gün düzeyinde tek
        ikili logit üretir.
        """
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, 128), nn.ReLU(), nn.Linear(128, 1)
        )

    def forward(
        self,
        stream: CompanyStream,
        day: object,
        device: torch.device,
    ) -> torch.Tensor:
        """Bir şirket-günün statik tek-atlamalı komşuluk logitini hesaplar.

        Her güncel ileti için kendi BERT vektörü ile bağlı komşuların ortalaması
        birleştirilir; komşusuz iletilerde sıfır vektörü kullanılır. İleti
        temsilleri gün düzeyinde ortalanıp tek logite dönüştürülür.
        """
        current_indices = stream.indices_by_date[day]
        day_events = stream.events_by_date.get(day, [])
        neighbors: MutableMapping[int, set] = defaultdict(set)
        current_set = set(current_indices)
        for event in day_events:
            source_idx = event["source_idx"]
            for target_idx, _, _ in event["interactions"]:
                neighbors[source_idx].add(target_idx)
                if target_idx in current_set:
                    neighbors[target_idx].add(source_idx)

        representations: List[torch.Tensor] = []
        for index in current_indices:
            own = stream.bert_features[index].to(device)
            adjacent = sorted(neighbors.get(index, set()))
            if adjacent:
                neighbor_mean = stream.bert_features[adjacent].to(device).mean(dim=0)
            else:
                neighbor_mean = torch.zeros_like(own)
            representations.append(torch.cat([own, neighbor_mean]))
        pooled = torch.stack(representations).mean(dim=0)
        return self.classifier(pooled).squeeze()


class TurkishSentiment:
    """Ham ve filtrelenmiş iletilerden kullanıcı-dengeli Türkçe duyarlılık üretir.

    Önceden eğitilmiş sınıflandırıcının olumlu ve olumsuz sınıf olasılıklarının
    farkını ileti skoru olarak kullanır. Çok ileti yazan tek bir kullanıcının gün
    sinyalini domine etmemesi için önce kullanıcı içinde, sonra kullanıcılar
    arasında ortalama alınır. Hesaplanan çerçeveler önbelleğe alınır.
    """

    def __init__(self, device: torch.device, config: ExperimentConfig):
        """Türkçe duyarlılık modelini yükler ve sınıf indekslerini yapılandırmadan okur.

        Sabit sınıf sırasına güvenmek yerine modelin ``label2id`` alanını kullanır.
        Model değerlendirme modundadır ve bu deneyde yeniden eğitilmez.
        """
        self.device = device
        self.config = config
        self.model_name = "savasy/bert-base-turkish-sentiment-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(device).eval()
        label_map = {str(key).lower(): value for key, value in self.model.config.label2id.items()}
        self.positive_index = label_map.get("positive", 1)
        self.negative_index = label_map.get("negative", 0)
        self.cache: Dict[Tuple[str, ...], float] = {}
        self.message_score_cache: Dict[str, float] = {}

    def message_scores(self, frame: pd.DataFrame) -> Mapping[str, float]:
        """İleti kimliği bazında duyarlılık skorlarını önbellekli olarak döndürür."""
        missing_rows = frame[
            ~frame["Mesaj_ID"].astype(str).isin(self.message_score_cache)
        ]
        if not missing_rows.empty:
            ids = missing_rows["Mesaj_ID"].astype(str).tolist()
            texts = missing_rows["Mesaj_Metni"].astype(str).tolist()
            for start in range(0, len(texts), self.config.feature_batch_size):
                batch_texts = texts[start : start + self.config.feature_batch_size]
                batch_ids = ids[start : start + self.config.feature_batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.device)
                with torch.no_grad():
                    probabilities = torch.softmax(self.model(**inputs).logits, dim=1)
                differences = (
                    probabilities[:, self.positive_index]
                    - probabilities[:, self.negative_index]
                ).detach().cpu().numpy()
                for message_id, score in zip(batch_ids, differences):
                    self.message_score_cache[message_id] = float(score)
        return {
            str(message_id): self.message_score_cache[str(message_id)]
            for message_id in frame["Mesaj_ID"].astype(str)
        }

    def user_balanced_score(
        self,
        frame: pd.DataFrame,
        user_weights: Optional[Mapping[str, float]] = None,
    ) -> float:
        """Bir şirket-gün ileti çerçevesi için kullanıcı-dengeli duyarlılık döndürür.

        Her iletiye ``P(positive)-P(negative)`` skoru verir; kullanıcı bazında ve
        ardından kullanıcılar arasında ortalar. Boş çerçevede 0 döndürür. Aynı
        ileti kimliği dizisi daha önce hesaplandıysa GPU çağrısı yerine önbelleği
        kullanır. Çıktı yaklaşık ``[-1, 1]`` aralığında kayan noktalı sayıdır.
        """
        if frame.empty:
            return 0.0
        message_key = tuple(frame["Mesaj_ID"].astype(str).tolist())
        if user_weights is None and message_key in self.cache:
            return self.cache[message_key]
        message_scores = self.message_scores(frame)
        user_scores: List[float] = []
        weights: List[float] = []
        grouping = frame["Kullanici_ID"].astype(str).copy()
        if "Kullanici_Bilinmiyor" in frame.columns:
            grouping = grouping.where(
                ~frame["Kullanici_Bilinmiyor"].astype(bool),
                "__UNKNOWN_MESSAGE_POOL__",
            )
        grouped_frame = frame.assign(_Sentiment_User_Group=grouping)
        for user, user_frame in grouped_frame.groupby(
            "_Sentiment_User_Group", sort=False
        ):
            values = [message_scores[str(value)] for value in user_frame["Mesaj_ID"]]
            user_scores.append(float(np.mean(values)))
            weights.append(
                max(0.0, float(user_weights.get(str(user), 1.0)))
                if user_weights is not None
                else 1.0
            )
        score = (
            float(np.average(user_scores, weights=weights))
            if sum(weights) > 0
            else 0.0
        )
        if user_weights is None:
            self.cache[message_key] = score
        return score


def instantiate_models(
    device: torch.device,
    experiment_profile: str = "keep_all",
    day_activity_mean: Optional[torch.Tensor] = None,
    day_activity_std: Optional[torch.Tensor] = None,
) -> ModelBundle:
    """Ana TGN--MIL modeli ile iki temel modeli aynı cihazda oluşturur.

    Her eğitim seed'i için taze ağırlıklar üretmek ve checkpoint yüklerken aynı
    mimariyi yeniden kurmak amacıyla merkezi fabrika fonksiyonudur. Çıktı dört
    PyTorch modülünü içeren ``ModelBundle`` nesnesidir.
    """
    return ModelBundle(
        tgn=EventDrivenTGN(
            use_activity_features=profile_uses_activity(experiment_profile),
            use_time_encoding=profile_uses_time_encoding(experiment_profile),
            carry_memory_across_days=profile_carries_memory_across_days(
                experiment_profile
            ),
        ).to(device),
        mil=MILClassifier(
            use_gated_text_fusion=profile_uses_gate(experiment_profile),
            use_day_activity_features=profile_uses_day_activity(experiment_profile),
            day_activity_feature_dim=13,
            day_activity_mean=day_activity_mean,
            day_activity_std=day_activity_std,
        ).to(device),
        text_baseline=TextOnlyBaseline().to(device),
        graph_baseline=StaticNeighborhoodBaseline().to(device),
    )


def chronological_day_keys(
    streams: Mapping[str, CompanyStream],
    allowed_dates: Iterable[object],
    excluded_company_days: Optional[Iterable[Tuple[str, object]]] = None,
) -> List[Tuple[object, str]]:
    """İzin verilen tarihlerdeki şirket-gün anahtarlarını küresel kronolojiye dizer.

    Bütün şirket akışlarından ``(gün, şirket)`` çiftlerini toplar ve önce tarih,
    sonra şirket koduna göre sıralar. Eğitim ve değerlendirme bu listeyi kullanarak
    gelecek günün geçmişten önce işlenmesini engeller.

    Çıktı: Kronolojik şirket-gün tuple listesi.
    """
    allowed = set(allowed_dates)
    excluded = {
        (str(company), day) for company, day in (excluded_company_days or [])
    }
    keys: List[Tuple[object, str]] = []
    for company, stream in streams.items():
        keys.extend(
            (day, company)
            for day in stream.indices_by_date
            if day in allowed and (str(company), day) not in excluded
        )
    return sorted(keys, key=lambda value: (value[0], value[1]))


def day_label(stream: CompanyStream, day: object) -> int:
    """Bir şirket-günün tek ve tutarlı SPK etiketini döndürür.

    Veri hazırlama aşaması gün içi etiket tutarlılığını doğruladığı için ilk
    iletinin etiketi tüm günün hedefi olarak güvenle kullanılabilir. Çıktı 0 veya
    1 tamsayısıdır.
    """
    indices = stream.indices_by_date[day]
    return int(stream.frame.iloc[indices[0]]["SPK_Etiketi"])


def positive_weight(
    streams: Mapping[str, CompanyStream], dates: Iterable[object], device: torch.device
) -> torch.Tensor:
    """Eğitim dilimindeki sınıf dengesizliği için pozitif kayıp ağırlığını hesaplar.

    Negatif şirket-gün sayısını pozitif sayısına böler ve ağırlığı en az 1 tutar.
    Pozitif örnek bulunmayan geçici bir fold'da eğitim çökmemesi için 1 kullanıp
    uyarı verir. Çıktı BCEWithLogitsLoss'a verilecek cihaz tensörüdür.
    """
    labels = [day_label(streams[company], day) for day, company in chronological_day_keys(streams, dates)]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0:
        logging.warning("Eğitim diliminde pozitif örnek yok; pos_weight=1 kullanılıyor.")
        value = 1.0
    else:
        value = max(1.0, negatives / positives)
    return torch.tensor(value, dtype=torch.float32, device=device)


def observed_positive_prior(
    streams: Mapping[str, CompanyStream], dates: Iterable[object]
) -> float:
    """Eğitim dilimindeki gözlenen pozitif şirket-gün oranını döndürür.

    SPK dışı günler kesin negatif olmadığından bu değer gerçek sınıf öncülü değil,
    nnPU riskinde kullanılan tutucu bir alt-sınır yaklaşımıdır. Fold bazında
    manifest ve metrik dosyalarına yazılarak duyarlılık analizine açık tutulur.
    """
    labels = np.asarray(
        [
            day_label(streams[company], day)
            for day, company in chronological_day_keys(streams, dates)
        ],
        dtype=float,
    )
    return float(labels.mean()) if len(labels) else 0.0


def batch_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mode: str,
    positive_prior: float,
    pos_weight: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
    """Standart BCE veya non-negative positive--unlabeled riskini hesaplar.

    ``bce`` doğrulanmış sıfır etiketlerini negatif kabul eden klasik temeldir.
    ``nnpu`` ise 1 etiketlerini pozitif, 0 etiketlerini etiketsiz kabul eder.
    Bir mini-batch yalnız tek sınıf içerirse sayısal kararlılık için etiketsiz
    örnekleri azaltılmış ağırlıkla kullanan PU-duyarlı BCE'ye geri dönülür.
    """
    logits = logits.reshape(-1)
    labels = labels.float().reshape(-1)
    if loss_mode == "bce":
        return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
    if loss_mode != "nnpu":
        raise ValueError(f"Bilinmeyen loss_mode: {loss_mode}")

    positive_mask = labels == 1
    unlabeled_mask = labels == 0
    if not bool(positive_mask.any()) or not bool(unlabeled_mask.any()):
        sample_weights = torch.where(
            positive_mask,
            torch.ones_like(labels),
            torch.full_like(labels, config.pu_unlabeled_weight),
        )
        losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        return (sample_weights * losses).sum() / sample_weights.sum().clamp_min(1.0)

    prior = torch.tensor(
        min(max(float(positive_prior), 1e-4), 1.0 - 1e-4),
        dtype=logits.dtype,
        device=logits.device,
    )
    positive_positive_risk = F.softplus(-logits[positive_mask]).mean()
    positive_negative_risk = F.softplus(logits[positive_mask]).mean()
    unlabeled_negative_risk = F.softplus(logits[unlabeled_mask]).mean()
    negative_risk = unlabeled_negative_risk - prior * positive_negative_risk
    return prior * positive_positive_risk + torch.clamp(negative_risk, min=0.0)


def train_models(
    streams: Mapping[str, CompanyStream],
    train_dates: Sequence[object],
    seed: int,
    device: torch.device,
    config: ExperimentConfig,
    loss_mode: str = "bce",
) -> Tuple[ModelBundle, MutableMapping[str, Dict[str, torch.Tensor]]]:
    """TGN--MIL, yalnız metin ve statik çizge modellerini aynı eğitim günlerinde eğitir.

    Her seed için modeller ve optimizer'lar sıfırdan kurulur. Sınıf dengesizliği
    ``pos_weight`` ile ele alınır. Her epoch başında kullanıcı belleği boşaltılır;
    epoch içinde şirket günleri kronolojik işlenir ve bellek günler arasında
    taşınır. TGN gradyanları kararlılık için kırpılır.

    Girdiler: Önceden hazırlanmış şirket akışları, eğitim tarihleri, seed, cihaz
    ve yapılandırma. Çıktılar: Eğitilmiş ``ModelBundle`` ve son epoch sonundaki
    ayrıştırılmış kullanıcı belleği. Epoch/gün bazlı loss ve ETA logları üretir.
    """
    set_seed(seed)
    experiment_profile = next(iter(streams.values())).experiment_profile
    day_activity_mean = None
    day_activity_std = None
    if profile_uses_day_activity(experiment_profile):
        day_activity_mean, day_activity_std = training_day_activity_statistics(
            streams, train_dates
        )
    models = instantiate_models(
        device,
        experiment_profile,
        day_activity_mean=day_activity_mean,
        day_activity_std=day_activity_std,
    )
    epochs_to_run = profile_epochs(experiment_profile, config)
    optimizers = {
        "tgn": torch.optim.Adam(
            list(models.tgn.parameters()) + list(models.mil.parameters()),
            lr=config.learning_rate,
        ),
        "text": torch.optim.Adam(models.text_baseline.parameters(), lr=config.learning_rate),
        "graph": torch.optim.Adam(models.graph_baseline.parameters(), lr=config.learning_rate),
    }
    pos_weight = positive_weight(streams, train_dates, device)
    positive_prior = observed_positive_prior(streams, train_dates)
    last_memory: MutableMapping[str, Dict[str, torch.Tensor]] = {}
    training_keys = chronological_day_keys(streams, train_dates)
    progress = ProgressReporter(
        f"Seed {seed} model eğitimi",
        epochs_to_run * len(training_keys),
        log_interval_seconds=20,
    )

    for epoch in range(epochs_to_run):
        models.tgn.train()
        models.mil.train()
        models.text_baseline.train()
        models.graph_baseline.train()
        memory: MutableMapping[str, Dict[str, torch.Tensor]] = {}
        losses: List[float] = []
        batch_text: List[torch.Tensor] = []
        batch_graph: List[torch.Tensor] = []
        batch_tgn: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        def optimize_batch() -> None:
            if not batch_labels:
                return
            labels_tensor = torch.stack(batch_labels)
            text_loss = batch_classification_loss(
                torch.stack(batch_text), labels_tensor, loss_mode,
                positive_prior, pos_weight, config
            )
            graph_loss = batch_classification_loss(
                torch.stack(batch_graph), labels_tensor, loss_mode,
                positive_prior, pos_weight, config
            )
            tgn_loss = batch_classification_loss(
                torch.stack(batch_tgn), labels_tensor, loss_mode,
                positive_prior, pos_weight, config
            )
            optimizers["text"].zero_grad()
            text_loss.backward()
            optimizers["text"].step()
            optimizers["graph"].zero_grad()
            graph_loss.backward()
            optimizers["graph"].step()
            optimizers["tgn"].zero_grad()
            tgn_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(models.tgn.parameters()) + list(models.mil.parameters()), 5.0
            )
            optimizers["tgn"].step()
            losses.append(float(tgn_loss.detach().cpu()))
            batch_text.clear()
            batch_graph.clear()
            batch_tgn.clear()
            batch_labels.clear()

        for key_index, (day, company) in enumerate(training_keys, start=1):
            stream = streams[company]
            indices = stream.indices_by_date[day]
            label = torch.tensor(float(day_label(stream, day)), device=device)

            text_logit = models.text_baseline(stream.bert_features[indices].to(device))
            graph_logit = models.graph_baseline(stream, day, device)
            memories, _, _, _ = models.tgn.forward_day(stream, day, memory, device)
            day_features = stream.bert_features[indices].to(device)
            tgn_logit, _ = models.mil(
                memories,
                day_features,
                stream.day_activity_features[day],
            )
            batch_text.append(text_logit)
            batch_graph.append(graph_logit)
            batch_tgn.append(tgn_logit)
            batch_labels.append(label)
            if (
                len(batch_labels) >= config.optimization_batch_days
                or key_index == len(training_keys)
            ):
                optimize_batch()
            progress.update(
                detail=(
                    f"{loss_mode} epoch {epoch + 1}/{epochs_to_run}, "
                    f"tarih {day}, şirket {company}"
                )
            )
        last_memory = memory
        logging.info(
            "Seed %d %s epoch %d/%d: TGN loss=%.5f, observed_prior=%.4f",
            seed,
            loss_mode,
            epoch + 1,
            epochs_to_run,
            float(np.mean(losses)) if losses else float("nan"),
            positive_prior,
        )
        if ARCHIVE_TRAINING_CURVES:
            diagnostic_records = evaluate_models(
                models,
                {},
                streams,
                train_dates,
                device,
                progress_label=(
                    f"Seed {seed} epoch {epoch + 1}/{epochs_to_run} eğitim tanısı"
                ),
            )
            diagnostic_labels = np.asarray(
                [record["Label"] for record in diagnostic_records], dtype=int
            )
            diagnostic_probabilities = np.clip(
                np.asarray(
                    [record["TGN_Prob"] for record in diagnostic_records], dtype=float
                ),
                1e-7,
                1.0 - 1e-7,
            )
            positive_weight_value = float(pos_weight.detach().cpu())
            sample_weights = np.where(
                diagnostic_labels == 1, positive_weight_value, 1.0
            )
            point_losses = -(
                sample_weights * diagnostic_labels * np.log(diagnostic_probabilities)
                + (1 - diagnostic_labels) * np.log(1 - diagnostic_probabilities)
            )
            raw_reference = float(np.log(2.0) * sample_weights.mean())
            row = pd.DataFrame(
                [
                    {
                        "Profile": experiment_profile,
                        "Loss_Mode": loss_mode,
                        "Seed": int(seed),
                        "Epoch": int(epoch + 1),
                        "Epochs_Configured": int(epochs_to_run),
                        "Learning_Rate": float(config.learning_rate),
                        "Train_First_Date": min(train_dates),
                        "Train_Last_Date": max(train_dates),
                        "Train_N": int(len(diagnostic_labels)),
                        "Train_Prevalence": float(diagnostic_labels.mean()),
                        "Logged_Batch_Loss_Mean": (
                            float(np.mean(losses)) if losses else float("nan")
                        ),
                        "Train_Weighted_BCE_Raw": float(point_losses.mean()),
                        "Constant_0p5_Weighted_BCE_Raw": raw_reference,
                        "Train_Weighted_BCE_Normalized": float(
                            point_losses.sum() / sample_weights.sum()
                        ),
                        "Constant_0p5_Weighted_BCE_Normalized": float(np.log(2.0)),
                        "Train_AUROC": (
                            float(roc_auc_score(
                                diagnostic_labels, diagnostic_probabilities
                            ))
                            if len(set(diagnostic_labels.tolist())) == 2
                            else float("nan")
                        ),
                        "Train_AUPRC": (
                            float(average_precision_score(
                                diagnostic_labels, diagnostic_probabilities
                            ))
                            if len(set(diagnostic_labels.tolist())) == 2
                            else float("nan")
                        ),
                    }
                ]
            )
            diagnostic_path = OUTPUT_DIR / "training_learning_curves.csv"
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            row.to_csv(
                diagnostic_path,
                mode="a",
                header=not diagnostic_path.exists(),
                index=False,
            )
    progress.finish(detail="eğitim tamamlandı")
    return models, last_memory


def evaluate_models(
    models: ModelBundle,
    memory: MutableMapping[str, Dict[str, torch.Tensor]],
    streams: Mapping[str, CompanyStream],
    dates: Sequence[object],
    device: torch.device,
    progress_label: str = "Model değerlendirmesi",
    excluded_company_days: Optional[Iterable[Tuple[str, object]]] = None,
) -> List[dict]:
    """Eğitilmiş modelleri verilen tarihlerde parametre güncellemeden değerlendirir.

    Modüller ``eval`` moduna alınır ve ``torch.no_grad`` altında aynı kronolojik
    şirket-gün panelinde çalıştırılır. Verilen TGN belleği değerlendirme boyunca
    ileri taşınır; böylece validasyon/test başlangıcındaki tarihsel durum korunur.

    Her kayıt gerçek etiket, üç model olasılığı, kullanıcı şüphe skorları ve olay
    bulunup bulunmadığını içerir. Çıktı sözlük listesidir; ilerleme/ETA logu yazar.
    """
    models.tgn.eval()
    models.mil.eval()
    models.text_baseline.eval()
    models.graph_baseline.eval()
    records: List[dict] = []
    evaluation_keys = chronological_day_keys(
        streams, dates, excluded_company_days=excluded_company_days
    )
    progress = ProgressReporter(progress_label, len(evaluation_keys), log_interval_seconds=20)
    with torch.no_grad():
        for day, company in evaluation_keys:
            stream = streams[company]
            indices = stream.indices_by_date[day]
            text_probability = torch.sigmoid(
                models.text_baseline(stream.bert_features[indices].to(device))
            ).item()
            graph_probability = torch.sigmoid(
                models.graph_baseline(stream, day, device)
            ).item()
            memories, users, user_to_position, events = models.tgn.forward_day(
                stream, day, memory, device
            )
            day_features = stream.bert_features[indices].to(device)
            tgn_logit, user_logits = models.mil(
                memories,
                day_features,
                stream.day_activity_features[day],
            )
            user_probabilities = torch.sigmoid(user_logits).cpu().numpy()
            records.append(
                {
                    "Tarih": day,
                    "Hisse_Kodu": company,
                    "Label": day_label(stream, day),
                    "Olay_ID": str(
                        stream.frame.iloc[indices[0]].get("Olay_ID", "") or ""
                    ),
                    "TGN_Prob": float(torch.sigmoid(tgn_logit).item()),
                    "Text_Prob": float(text_probability),
                    "Graph_Prob": float(graph_probability),
                    "User_Scores": {
                        user: float(user_probabilities[position])
                        for user, position in user_to_position.items()
                    },
                    "Has_Events": bool(events),
                }
            )
            progress.update(detail=f"tarih {day}, şirket {company}")
    progress.finish(detail="değerlendirme tamamlandı")
    return records


def optimal_f1_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Validasyon Precision--Recall eğrisinde F1'i en büyükleyen eşiği seçer.

    Test verisine bakmadan karar eşiği kilitlemek için kullanılır. Tek sınıflı
    dilimde veya eşik üretilemediğinde güvenli varsayılan 0.5 döndürür. Çıktı
    kayan noktalı olasılık eşiğidir.
    """
    if len(set(labels)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def classification_metrics(
    labels: Sequence[int], probabilities: Sequence[float], threshold: float
) -> dict:
    """Olasılık ve sabit eşikten şirket-gün sınıflandırma ölçülerini hesaplar.

    AUROC ve AUPRC yalnız iki sınıf varsa hesaplanır; kesinlik, duyarlılık, F1,
    Brier, eşik, prevalans ve örnek sayısı her durumda raporlanır. Bu fonksiyon
    eşik seçmez ve veri üzerinde model eğitmez.

    Çıktı: Sonuç CSV'lerine doğrudan yazılabilecek metrik sözlüğü.
    """
    labels_array = np.asarray(labels, dtype=int)
    probabilities_array = np.asarray(probabilities, dtype=float)
    predictions = (probabilities_array >= threshold).astype(int)
    two_classes = len(np.unique(labels_array)) == 2
    return {
        "AUROC": float(roc_auc_score(labels_array, probabilities_array)) if two_classes else np.nan,
        "AUPRC": float(average_precision_score(labels_array, probabilities_array)) if two_classes else np.nan,
        "Precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "Recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "F1": float(f1_score(labels_array, predictions, zero_division=0)),
        "Brier": float(brier_score_loss(labels_array, probabilities_array)),
        "Threshold": float(threshold),
        "Prevalence": float(labels_array.mean()),
        "N": int(len(labels_array)),
    }


def generalization_macro_metrics(
    predictions: pd.DataFrame, probability_column: str
) -> Mapping[str, float]:
    """Şirket-makro ve olay-birimli ölçüleri hesaplar.

    Şirket-makro AUPRC yalnız test içinde iki sınıfı da bulunan şirketlerde
    hesaplanır. Olay-birimli ölçüde pozitif günler şirket--Olay_ID birimi içinde
    ortalanır; negatif şirket-günler ayrı değerlendirme birimi olarak kalır.
    Böylece her pozitif olay AUPRC hesabına tek bir pozitif birim olarak girer;
    çok mesajlı veya uzun bir olay yalnız büyüklüğü nedeniyle baskınlaşamaz.

    ``Positive_Event_Macro_Mean_Probability`` de her olayın gün-ortalaması
    alınarak olaylar arasında eşit ağırlıklı ortalamayı verir. Bu değer tek
    başına ayırt etme metriği değildir; olay-birimli AUPRC ile birlikte okunur.
    """
    company_scores: List[float] = []
    for _, company_frame in predictions.groupby("Hisse_Kodu"):
        if company_frame["Label"].nunique() == 2:
            company_scores.append(
                float(
                    average_precision_score(
                        company_frame["Label"], company_frame[probability_column]
                    )
                )
            )

    frame = predictions.copy()
    frame["Olay_ID"] = (
        frame["Olay_ID"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    frame["Positive_Event_Key"] = np.where(
        frame["Label"] == 1,
        frame["Hisse_Kodu"].astype(str) + "::" + frame["Olay_ID"],
        "",
    )
    frame["Evaluation_Unit"] = np.where(
        frame["Label"] == 1,
        "EVENT_" + frame["Positive_Event_Key"],
        "NEG_"
        + frame["Hisse_Kodu"].astype(str)
        + "_"
        + frame["Tarih"].astype(str),
    )
    units = (
        frame.groupby("Evaluation_Unit", as_index=False)
        .agg(Label=("Label", "max"), Probability=(probability_column, "mean"))
    )
    event_unit_auprc = (
        float(average_precision_score(units["Label"], units["Probability"]))
        if units["Label"].nunique() == 2
        else float("nan")
    )
    positive_event_probabilities = (
        frame.loc[frame["Label"] == 1]
        .groupby("Positive_Event_Key")[probability_column]
        .mean()
    )
    return {
        "Company_Macro_AUPRC": float(np.mean(company_scores))
        if company_scores
        else float("nan"),
        "Company_Macro_N": int(len(company_scores)),
        "Event_Unit_AUPRC": event_unit_auprc,
        "Event_Unit_N": int(len(units)),
        "Positive_Event_Macro_Mean_Probability": (
            float(positive_event_probabilities.mean())
            if len(positive_event_probabilities)
            else float("nan")
        ),
        "Positive_Event_Macro_SD_Probability": (
            float(positive_event_probabilities.std(ddof=1))
            if len(positive_event_probabilities) > 1
            else float("nan")
        ),
        "Positive_Event_Macro_N": int(len(positive_event_probabilities)),
    }


def leave_one_positive_event_out_metrics(
    predictions: pd.DataFrame, probability_column: str
) -> pd.DataFrame:
    """Her pozitif şirket--olayı sırayla çıkarıp AUPRC duyarlılığını ölçer.

    Model yeniden eğitilmez; bu bir test-kompozisyonu etki analizidir. Önce bütün
    şirket-günlerle AUPRC hesaplanır, sonra tek bir pozitif olayın bütün günleri
    test tablosundan çıkarılarak metrik yinelenir. ``Delta_Without_Minus_Full``
    pozitifse olay çıkarılınca sonuç iyileşmiş, negatifse kötüleşmiştir. Mutlak
    farkın büyük olması ana sonucun o olaya duyarlı olduğunu gösterir.
    """
    required = {"Label", "Hisse_Kodu", "Olay_ID", probability_column}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Leave-one-event-out için eksik sütunlar: {sorted(missing)}")
    frame = predictions.copy()
    frame["Olay_ID"] = (
        frame["Olay_ID"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    positive = frame.loc[frame["Label"] == 1].copy()
    positive["Positive_Event_Key"] = (
        positive["Hisse_Kodu"].astype(str) + "::" + positive["Olay_ID"]
    )
    if frame["Label"].nunique() < 2 or positive.empty:
        return pd.DataFrame()
    full_auprc = float(average_precision_score(frame["Label"], frame[probability_column]))
    rows: List[dict] = []
    for event_key, event_frame in positive.groupby("Positive_Event_Key", sort=True):
        company = str(event_frame["Hisse_Kodu"].iloc[0])
        event_id = str(event_frame["Olay_ID"].iloc[0])
        keep_mask = ~(
            (frame["Label"] == 1)
            & frame["Hisse_Kodu"].astype(str).eq(company)
            & frame["Olay_ID"].eq(event_id)
        )
        reduced = frame.loc[keep_mask]
        without_auprc = (
            float(average_precision_score(reduced["Label"], reduced[probability_column]))
            if reduced["Label"].nunique() == 2
            else float("nan")
        )
        delta = without_auprc - full_auprc if np.isfinite(without_auprc) else float("nan")
        rows.append(
            {
                "Model": probability_column.replace("_Prob", ""),
                "Hisse_Kodu": company,
                "Olay_ID": event_id,
                "Event_Key": event_key,
                "Event_Positive_Days": int(len(event_frame)),
                "Event_Mean_Probability": float(event_frame[probability_column].mean()),
                "Full_AUPRC": full_auprc,
                "AUPRC_Without_Event": without_auprc,
                "Delta_Without_Minus_Full": delta,
                "Absolute_Delta": abs(delta) if np.isfinite(delta) else float("nan"),
                "Interpretation": (
                    "result_improves_without_event"
                    if np.isfinite(delta) and delta > 0
                    else "result_worsens_without_event"
                    if np.isfinite(delta) and delta < 0
                    else "no_change_or_undefined"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("Absolute_Delta", ascending=False)


def model_thresholds(records: Sequence[dict]) -> Mapping[str, float]:
    """Üç model için validasyon kayıtlarından ayrı F1-optimal eşikler üretir.

    TGN, yalnız metin ve statik çizge olasılıkları birbirinden farklı kalibre
    olabileceği için her model kendi validasyon eşiğini alır. Çıktı ``TGN``,
    ``Text`` ve ``Graph`` anahtarlarını olasılık eşiklerine bağlayan sözlüktür.
    """
    labels = [record["Label"] for record in records]
    return {
        "TGN": optimal_f1_threshold(labels, [record["TGN_Prob"] for record in records]),
        "Text": optimal_f1_threshold(labels, [record["Text_Prob"] for record in records]),
        "Graph": optimal_f1_threshold(labels, [record["Graph_Prob"] for record in records]),
    }


def _cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    """Model ağırlıklarını cihazdan bağımsız kaydetmek için CPU'ya taşır.

    Tensörleri gradyan grafiğinden ayırıp CPU kopyaları olarak döndürür. Böylece
    CUDA ile eğitilmiş checkpoint, daha sonra CPU veya farklı GPU'da yüklenebilir.
    Model nesnesinin kendisini değiştirmez.
    """
    return {
        name: tensor.detach().cpu()
        for name, tensor in module.state_dict().items()
    }


def save_model_checkpoint(
    path: Path,
    models: ModelBundle,
    config: ExperimentConfig,
    thresholds: Mapping[str, float],
    selected_parameters: Tuple[int, float],
    seed: int,
    validation_auprc: float,
    forum_sha256: str,
    market_sha256: str,
    locked_test_start: str,
    experiment_metadata: Optional[Mapping[str, Any]] = None,
) -> None:
    """Eğitilmiş ağırlıkları ve tahmin için gereken ayarları tek checkpoint'te saklar.

    Kullanıcı belleği özellikle kaydedilmez; bellek belirli bir veri akışına aittir.
    Yeni veri tahmininde bellek, verilen iletiler kronolojik olarak oynatılarak
    yeniden oluşturulur. Dosyada dört modelin CPU ağırlıkları, validasyon eşikleri,
    downstream filtre parametreleri, seed, validasyon AUPRC'si, veri karmaları,
    test sınırı, yapılandırma ve sabit ön-eğitim modeli adları bulunur.

    Girdi: Hedef yol ve koşuya ait model/metadata nesneleri.
    Çıktı: Değer döndürmez; ``torch.save`` ile taşınabilir ``.pt`` dosyası üretir.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": 4,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "seed": int(seed),
        "validation_auprc": float(validation_auprc),
        "thresholds": {key: float(value) for key, value in thresholds.items()},
        "selected_pruning_percentile": int(selected_parameters[0]),
        "selected_sentiment_threshold": float(selected_parameters[1]),
        "locked_test_start": str(locked_test_start),
        "forum_sha256": forum_sha256,
        "market_sha256": market_sha256,
        "experiment_metadata": dict(experiment_metadata or {}),
        "config": asdict(config),
        "pretrained_models": {
            "bert": "dbmdz/bert-base-turkish-cased",
            "sbert": "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
            "sentiment": "savasy/bert-base-turkish-sentiment-cased",
        },
        "state_dicts": {
            "tgn": _cpu_state_dict(models.tgn),
            "mil": _cpu_state_dict(models.mil),
            "text_baseline": _cpu_state_dict(models.text_baseline),
            "graph_baseline": _cpu_state_dict(models.graph_baseline),
        },
    }
    torch.save(payload, path)
    logging.info(
        "Model kaydedildi: %s | seed=%d | validasyon AUPRC=%.6f",
        path,
        seed,
        validation_auprc,
    )


class SavedModelPredictor:
    """Kaydedilmiş TGN--MIL modelini yeni forum verisinde çalıştırır.

    Yeni CSV'nin eğitim sütunlarından ``SPK_Etiketi`` dışında kalanlarını
    içermesi yeterlidir. Zaman belleği sıfırdan kurulduğu için, devam eden bir
    şirket akışında tahmin yapılacaksa yalnız son günü değil, makul bir tarihsel
    ısınma dönemini de CSV'ye eklemek gerekir.

    Örnek
    -----
    predictor = SavedModelPredictor("result_logs_v3/keep_all/test_2025/nnpu/saved_models/seed_42.pt")
    tahminler = predictor.predict_csv(
        "yeni_forum_verisi.csv", "yeni_tahminler.csv"
    )
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        device: Optional[str] = None,
    ) -> None:
        """Checkpoint'i doğrular, mimariyi kurar ve ağırlıkları seçilen cihaza yükler.

        Cihaz verilmezse kullanılabilir CUDA tercih edilir. Yeni PyTorch sürümünde
        güvenilir yerel checkpoint için ``weights_only=False`` açıkça kullanılır;
        eski sürümlerle uyumluluk için geri dönüş yolu vardır. Kayıt sürümü,
        deney yapılandırması, eşikler ve dört state_dict yeniden yüklenir.

        Girdi: ``best_model.pt`` veya seed checkpoint yolu ve isteğe bağlı cihaz.
        Yan etki: Model ağırlıklarını belleğe/GPU'ya yükler ve bilgi logu yazar.
        """
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Kayıtlı model bulunamadı: {self.checkpoint_path}")
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        try:
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
        except TypeError:  # Eski PyTorch sürümleri weights_only parametresini bilmez.
            self.checkpoint = torch.load(
                self.checkpoint_path, map_location=self.device
            )
        if self.checkpoint.get("checkpoint_version") not in {3, 4}:
            raise ValueError("Desteklenmeyen kayıtlı model sürümü.")

        config_values = dict(self.checkpoint["config"])
        config_values["seeds"] = tuple(config_values["seeds"])
        config_values["parameter_grid"] = tuple(
            tuple(item) for item in config_values["parameter_grid"]
        )
        config_values["round_trip_costs"] = tuple(
            config_values["round_trip_costs"]
        )
        config_values["soft_filter_alphas"] = tuple(config_values["soft_filter_alphas"])
        config_values["semantic_windows_seconds"] = tuple(
            config_values["semantic_windows_seconds"]
        )
        config_values["outer_test_years"] = tuple(config_values["outer_test_years"])
        config_values["data_variants"] = tuple(config_values["data_variants"])
        self.config = ExperimentConfig(**config_values)
        self.thresholds = self.checkpoint["thresholds"]
        metadata = self.checkpoint.get("experiment_metadata", {})
        self.experiment_profile = str(metadata.get("data_variant", "keep_all"))
        self.models = instantiate_models(self.device, self.experiment_profile)
        state_dicts = self.checkpoint["state_dicts"]
        self.models.tgn.load_state_dict(state_dicts["tgn"])
        self.models.mil.load_state_dict(state_dicts["mil"])
        self.models.text_baseline.load_state_dict(state_dicts["text_baseline"])
        self.models.graph_baseline.load_state_dict(state_dicts["graph_baseline"])
        self.models.tgn.eval()
        self.models.mil.eval()
        self.models.text_baseline.eval()
        self.models.graph_baseline.eval()
        logging.info(
            "Kayıtlı model yüklendi: %s | seed=%s | cihaz=%s",
            self.checkpoint_path,
            self.checkpoint.get("seed"),
            self.device,
        )

    def predict_csv(
        self,
        forum_csv: Path | str,
        output_csv: Optional[Path | str] = None,
    ) -> pd.DataFrame:
        """Etiketsiz veya etiketli yeni forum CSV'si için şirket-gün tahmini üretir.

        CSV eğitimle aynı temel ileti sütunlarını taşımalıdır; ``SPK_Etiketi``
        zorunlu değildir. Metin özellikleri ve geçmişe-bakan olaylar yeniden
        hesaplanır, kullanıcı belleği dosyanın başından kronolojik olarak ısıtılır
        ve üç modelin olasılıkları checkpoint eşikleriyle ikili karara çevrilir.

        Çıktı sütunları tarih, şirket kodu, TGN/metin/statik çizge olasılık ve
        kararları ile olay göstergesidir. ``output_csv`` verilirse sonuç diske
        yazılır; her durumda tahmin DataFrame'i döndürülür. Yeni akışta yeterli
        tarihsel ısınma verisi sağlamak çağıranın sorumluluğundadır.
        """
        forum = prepare_forum(Path(forum_csv), require_labels=False)
        extractor = FeatureExtractors(self.device, self.config)
        streams = build_streams(
            forum, extractor, self.config, experiment_profile=self.experiment_profile
        )
        dates = sorted(forum["Date"].unique())
        records = evaluate_models(
            self.models,
            {},
            streams,
            dates,
            self.device,
            progress_label="Kayıtlı model ile yeni veri tahmini",
        )
        rows = []
        for record in records:
            rows.append(
                {
                    "Tarih": record["Tarih"],
                    "Hisse_Kodu": record["Hisse_Kodu"],
                    "TGN_Olasilik": record["TGN_Prob"],
                    "TGN_Tahmin": int(
                        record["TGN_Prob"] >= self.thresholds["TGN"]
                    ),
                    "Metin_Olasilik": record["Text_Prob"],
                    "Metin_Tahmin": int(
                        record["Text_Prob"] >= self.thresholds["Text"]
                    ),
                    "Statik_Cizge_Olasilik": record["Graph_Prob"],
                    "Statik_Cizge_Tahmin": int(
                        record["Graph_Prob"] >= self.thresholds["Graph"]
                    ),
                    "Etkilesim_Var": record["Has_Events"],
                }
            )
        predictions = pd.DataFrame(rows)
        if output_csv is not None:
            output_path = Path(output_csv)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_path, index=False)
            logging.info("Yeni veri tahminleri kaydedildi: %s", output_path)
        return predictions


def filtered_day_frame(
    stream: CompanyStream,
    record: dict,
    tgn_threshold: float,
    pruning_percentile: int,
) -> pd.DataFrame:
    """TGN kararına göre bir şirket-günün yüksek şüphe skorlu iletilerini filtreler.

    Gün olay içermiyorsa veya şirket-gün olasılığı validasyon eşiğini aşmıyorsa
    çerçeve değişmeden döner. Aksi durumda yalnız o gün ileti yazan ve kullanıcı
    skoru bulunan hesapların dağılımından yüzdelik kesim hesaplanır; kesimin
    üzerindeki bağlı kullanıcıların iletileri çıkarılır. İzole/skorsuz kullanıcılar
    korunur.

    Çıktı: Ham DataFrame'in indekslerini koruyan filtrelenmiş görünümü. Girdi
    kayıtlarını yerinde değiştirmez.
    """
    indices = stream.indices_by_date[record["Tarih"]]
    day_frame = stream.frame.iloc[indices]
    if not record["Has_Events"] or record["TGN_Prob"] <= tgn_threshold:
        return day_frame
    current_users = list(dict.fromkeys(day_frame["Kullanici_ID"].astype(str)))
    scored_users = {
        user: record["User_Scores"][user]
        for user in current_users
        if user in record["User_Scores"]
    }
    if not scored_users:
        return day_frame
    cutoff = float(np.percentile(list(scored_users.values()), pruning_percentile))
    keep = day_frame["Kullanici_ID"].astype(str).map(
        lambda user: user not in scored_users or scored_users[user] <= cutoff
    )
    return day_frame[keep]


def attach_sentiment_scores(
    records: List[dict],
    streams: Mapping[str, CompanyStream],
    sentiment: TurkishSentiment,
    tgn_threshold: float,
    pruning_values: Iterable[int],
    soft_alphas: Iterable[float] = (),
    progress_label: str = "Duyarlılık skorları",
) -> None:
    """Değerlendirme kayıtlarına ham ve alternatif filtreli duyarlılık skorları ekler.

    Her şirket-gün için kullanıcı-dengeli ham skor hesaplanır. İstenen her budama
    yüzdeliğinde ``filtered_day_frame`` uygulanıp ayrı skor üretilir. Sonuçlar
    kayıt sözlüklerine yerinde ``Raw_Sentiment`` ve ``Filtered_Sentiment`` olarak
    eklenir; yeni liste döndürülmez. Hesaplama önbellek kullanır ve ETA raporlar.
    """
    progress = ProgressReporter(progress_label, len(records), log_interval_seconds=20)
    for record in records:
        stream = streams[record["Hisse_Kodu"]]
        day_frame = stream.frame.iloc[stream.indices_by_date[record["Tarih"]]]
        record["Raw_Sentiment"] = sentiment.user_balanced_score(day_frame)
        record["Filtered_Sentiment"] = {}
        record["Soft_Sentiment"] = {}
        for pruning in pruning_values:
            cleaned = filtered_day_frame(stream, record, tgn_threshold, pruning)
            record["Filtered_Sentiment"][pruning] = sentiment.user_balanced_score(cleaned)
        for alpha in soft_alphas:
            if record["Has_Events"]:
                user_weights = {
                    str(user): (1.0 - min(1.0, record["TGN_Prob"] * float(score)))
                    ** float(alpha)
                    for user, score in record["User_Scores"].items()
                }
            else:
                user_weights = {}
            record["Soft_Sentiment"][float(alpha)] = sentiment.user_balanced_score(
                day_frame, user_weights=user_weights
            )
        progress.update(
            detail=f"tarih {record['Tarih']}, şirket {record['Hisse_Kodu']}"
        )
    progress.finish(detail="duyarlılık hesaplaması tamamlandı")


def signals_from_records(
    records: Sequence[dict], pruning: int, sentiment_threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Duyarlılık kayıtlarını üç karşılaştırılabilir günlük işlem sinyaline çevirir.

    Ham strateji ham duyarlılığı, TGN-filtreli strateji seçilen budama skorunu aynı
    duyarlılık eşiğiyle değerlendirir. Karşılaştırma stratejisi her şirket-günde
    daima uzun sinyali üretir. Üç panel aynı tarih ve şirket anahtarlarını taşır.

    Çıktı: ``(raw_signals, filtered_signals, benchmark_signals)`` DataFrame üçlüsü.
    """
    raw_rows, filtered_rows, benchmark_rows = [], [], []
    for record in records:
        base = {"Tarih": record["Tarih"], "Hisse_Kodu": record["Hisse_Kodu"]}
        raw_rows.append(
            {**base, "Signal": int(record["Raw_Sentiment"] > sentiment_threshold)}
        )
        filtered_rows.append(
            {
                **base,
                "Signal": int(
                    record["Filtered_Sentiment"][pruning] > sentiment_threshold
                ),
            }
        )
        benchmark_rows.append({**base, "Signal": 1})
    return pd.DataFrame(raw_rows), pd.DataFrame(filtered_rows), pd.DataFrame(benchmark_rows)


def signals_from_records_v3(
    records: Sequence[dict],
    pruning: int,
    sentiment_threshold: float,
    soft_alphas: Sequence[float],
    top_fraction: float,
) -> Mapping[str, pd.DataFrame]:
    """Ham, sert/yumuşak filtreli ve kesitsel top-k sinyallerini birlikte üretir."""
    score_rows: MutableMapping[str, List[dict]] = defaultdict(list)
    for record in records:
        base = {"Tarih": record["Tarih"], "Hisse_Kodu": record["Hisse_Kodu"]}
        score_rows["Raw_LongCash"].append(
            {**base, "Score": record["Raw_Sentiment"]}
        )
        score_rows["Hard_Filter_LongCash"].append(
            {**base, "Score": record["Filtered_Sentiment"][pruning]}
        )
        for alpha in soft_alphas:
            score_rows[f"Soft_a{alpha:g}_LongCash"].append(
                {**base, "Score": record["Soft_Sentiment"][float(alpha)]}
            )

    output: Dict[str, pd.DataFrame] = {}
    for name, rows in score_rows.items():
        frame = pd.DataFrame(rows)
        frame["Signal"] = (frame["Score"] > sentiment_threshold).astype(int)
        output[name] = frame

        top_frame = frame.copy()
        top_frame["Signal"] = 0
        for _, indices in top_frame.groupby("Tarih").groups.items():
            ordered = top_frame.loc[list(indices)].sort_values("Score", ascending=False)
            count = max(1, int(np.ceil(len(ordered) * top_fraction)))
            selected = ordered.head(count).index
            top_frame.loc[selected, "Signal"] = 1
        output[name.replace("LongCash", "TopFraction")] = top_frame

    benchmark = pd.DataFrame(
        [
            {"Tarih": record["Tarih"], "Hisse_Kodu": record["Hisse_Kodu"], "Score": 1.0, "Signal": 1}
            for record in records
        ]
    )
    output["Intraday_Always_Long"] = benchmark
    return output


def run_backtest(
    signals: pd.DataFrame,
    market: pd.DataFrame,
    round_trip_cost: float,
) -> Tuple[dict, pd.DataFrame]:
    """Şirket-gün sinyallerini T+1 açılış--kapanış getirileriyle backtest eder.

    Sinyaller piyasa tablosuyla tarih ve şirket üzerinden iç birleşimle eşlenir.
    Pozisyon varsa gün içi brüt getiriden gidiş-dönüş maliyeti, pozisyon yoksa
    günlük risksiz getiri uygulanır. Aynı tarihteki şirket getirileri eşit
    ağırlıkla portföy getirisine dönüştürülür.

    Çıktılar: Toplam getiri, yıllık Sharpe, azami düşüş, işlem, maruziyet ve kapsam
    ölçülerini içeren sözlük ile günlük portföy/equity eğrisi. Eşleşme yoksa boş
    eğri ve tanımlı güvenli özet döndürür.
    """
    merged = signals.merge(market, on=["Tarih", "Hisse_Kodu"], how="inner")
    if merged.empty:
        return {
            "Return": np.nan,
            "Sharpe": np.nan,
            "MDD": np.nan,
            "Trades": 0,
            "Exposure": np.nan,
            "Coverage": 0.0,
            "Excess_Return": np.nan,
            "Information_Ratio": np.nan,
        }, pd.DataFrame()
    merged["Gross_Return"] = merged["T1_Kapanis"] / merged["T1_Acilis"] - 1.0
    merged["Net_Return"] = np.where(
        merged["Signal"] == 1,
        merged["Gross_Return"] - round_trip_cost,
        merged["Gunluk_Risksiz_Getiri"],
    )
    daily = (
        merged.groupby("Tarih")
        .agg(
            Net_Return=("Net_Return", "mean"),
            Risk_Free=("Gunluk_Risksiz_Getiri", "mean"),
            Universe_Return=("Gross_Return", "mean"),
        )
        .sort_index()
    )
    daily["Wealth"] = (1.0 + daily["Net_Return"]).cumprod()
    daily["Cum_Return"] = daily["Wealth"] - 1.0
    daily["Drawdown"] = daily["Wealth"] / daily["Wealth"].cummax() - 1.0
    daily["Daily_Excess"] = daily["Net_Return"] - daily["Universe_Return"]
    daily["Benchmark_Wealth"] = (1.0 + daily["Universe_Return"]).cumprod()
    daily["Cum_Benchmark_Return"] = daily["Benchmark_Wealth"] - 1.0
    # Günlük aktif getiriler bilgi oranında kullanılır. Kümülatif fazla getiri
    # ise günlük farkların bağımsız bir varlıkmış gibi bileşiklenmesiyle değil,
    # strateji ve karşılaştırma portföyünün kümülatif getirileri arasındaki
    # farkla hesaplanır.
    daily["Cum_Excess_Return"] = (
        daily["Cum_Return"] - daily["Cum_Benchmark_Return"]
    )
    excess = daily["Net_Return"] - daily["Risk_Free"]
    standard_deviation = excess.std(ddof=1)
    sharpe = (
        np.sqrt(252) * excess.mean() / standard_deviation
        if pd.notna(standard_deviation) and standard_deviation > 0
        else np.nan
    )
    excess_std = daily["Daily_Excess"].std(ddof=1)
    information_ratio = (
        np.sqrt(252) * daily["Daily_Excess"].mean() / excess_std
        if pd.notna(excess_std) and excess_std > 0
        else np.nan
    )
    metrics = {
        "Return": float(daily["Cum_Return"].iloc[-1]),
        "Sharpe": float(sharpe),
        "MDD": float(-daily["Drawdown"].min()),
        "Trades": int(merged["Signal"].sum()),
        "Exposure": float(merged["Signal"].mean()),
        "Coverage": float(len(merged) / len(signals)) if len(signals) else 0.0,
        "Excess_Return": float(daily["Cum_Excess_Return"].iloc[-1]),
        "Information_Ratio": float(information_ratio),
    }
    return metrics, daily.reset_index()


def signal_change_diagnostics(
    strategies: Mapping[str, pd.DataFrame], output_path: Path
) -> pd.DataFrame:
    """Her filtreli stratejinin ham sinyali gerçekten değiştirip değiştirmediğini kaydeder."""
    raw = strategies["Raw_LongCash"][["Tarih", "Hisse_Kodu", "Signal", "Score"]].rename(
        columns={"Signal": "Raw_Signal", "Score": "Raw_Score"}
    )
    rows: List[dict] = []
    for name, frame in strategies.items():
        if name in {"Raw_LongCash", "Intraday_Always_Long"} or "TopFraction" in name:
            continue
        merged = raw.merge(
            frame[["Tarih", "Hisse_Kodu", "Signal", "Score"]],
            on=["Tarih", "Hisse_Kodu"],
            how="inner",
        )
        rows.append(
            {
                "Strategy": name,
                "N": len(merged),
                "Signal_Flip_Count": int((merged["Raw_Signal"] != merged["Signal"]).sum()),
                "Signal_Flip_Rate": float((merged["Raw_Signal"] != merged["Signal"]).mean()),
                "Mean_Absolute_Score_Change": float(
                    (merged["Raw_Score"] - merged["Score"]).abs().mean()
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result


def expanding_folds(dates: Sequence[object], folds: int) -> List[Tuple[List[object], List[object]]]:
    """Benzersiz tarihleri genişleyen pencereli rolling-origin fold'larına ayırır.

    Scikit-learn ``TimeSeriesSplit`` kullanılarak her fold'da eğitim geçmişi
    büyür ve validasyon daima eğitimden sonra gelir. Rastgele karıştırma yapılmaz.
    Yetersiz tarih varsa anlamlı değerlendirme kurulamayacağı için hata yükselir.

    Çıktı: Her elemanı ``(train_dates, validation_dates)`` olan liste.
    """
    dates_array = np.asarray(sorted(dates), dtype=object)
    if len(dates_array) < folds + 2:
        raise ValueError("Rolling-origin için yeterli benzersiz tarih yok.")
    splitter = TimeSeriesSplit(n_splits=folds)
    return [
        (dates_array[train_indices].tolist(), dates_array[val_indices].tolist())
        for train_indices, val_indices in splitter.split(dates_array)
    ]


def positive_event_keys_for_dates(
    streams: Mapping[str, CompanyStream],
    dates: Iterable[object],
    excluded_company_days: Optional[Iterable[Tuple[str, object]]] = None,
) -> set:
    """Verilen tarihlerdeki pozitif şirket--Olay_ID anahtarlarını döndürür."""
    allowed = set(dates)
    excluded = {
        (str(company), day) for company, day in (excluded_company_days or [])
    }
    event_keys: set = set()
    for company, stream in streams.items():
        subset = stream.frame[
            stream.frame["Date"].isin(allowed) & (stream.frame["SPK_Etiketi"] == 1)
        ].copy()
        if excluded:
            subset = subset.loc[
                ~subset["Date"].map(lambda day: (str(company), day) in excluded)
            ]
        event_ids = subset["Olay_ID"].dropna().astype(str).str.strip()
        event_keys.update(
            f"{company}::{event_id}" for event_id in event_ids if event_id
        )
    return event_keys


def company_days_for_positive_events(
    streams: Mapping[str, CompanyStream],
    dates: Iterable[object],
    event_keys: Iterable[str],
) -> set:
    """Belirli pozitif şirket--olayların aday şirket-günlerini döndürür.

    İç rolling-origin sınırı uzun bir olayı ikiye böldüğünde, validasyon
    tarafındaki aynı olaya ait yalnız ilgili şirket-günler çıkarılır. Aynı
    takvim günündeki diğer şirketler korunur.
    """
    allowed_dates = set(dates)
    allowed_events = {str(event).strip() for event in event_keys} - {""}
    matched_company_days: set = set()
    if not allowed_dates or not allowed_events:
        return matched_company_days
    for company, stream in streams.items():
        event_values = stream.frame["Olay_ID"].fillna("").astype(str).str.strip()
        company_event_values = str(company) + "::" + event_values
        subset = stream.frame[
            stream.frame["Date"].isin(allowed_dates)
            & (stream.frame["SPK_Etiketi"] == 1)
            & company_event_values.isin(allowed_events)
        ]
        matched_company_days.update((str(company), day) for day in subset["Date"])
    return matched_company_days


def class_support_for_dates(
    streams: Mapping[str, CompanyStream],
    dates: Iterable[object],
    excluded_company_days: Optional[Iterable[Tuple[str, object]]] = None,
) -> Tuple[int, int]:
    """Tarih kümesindeki benzersiz şirket-gün pozitif ve negatif sayılarını verir."""
    allowed = set(dates)
    excluded = {
        (str(company), day) for company, day in (excluded_company_days or [])
    }
    positives = 0
    negatives = 0
    for company, stream in streams.items():
        subset = stream.frame.loc[
            stream.frame["Date"].isin(allowed), ["Date", "SPK_Etiketi"]
        ].drop_duplicates("Date")
        if excluded:
            subset = subset.loc[
                ~subset["Date"].map(lambda day: (str(company), day) in excluded)
            ]
        positives += int((subset["SPK_Etiketi"] == 1).sum())
        negatives += int((subset["SPK_Etiketi"] == 0).sum())
    return positives, negatives


def event_safe_expanding_folds(
    streams: Mapping[str, CompanyStream],
    dates: Sequence[object],
    folds: int,
) -> List[Tuple[List[object], List[object], set]]:
    """Olay çakışmasını validasyon tarafından purging ile gideren zaman fold'ları kurar.

    ``TimeSeriesSplit`` sınırı bir resmî olayı eğitim ve validasyon tarafına
    bölerse olay kimliği değiştirilmez. Bunun yerine aynı olaya ait validasyon
    şirket-günleri fold'dan çıkarılır; aynı takvim tarihindeki diğer şirketler
    korunur. Kalan validasyon diliminde iki sınıf desteği ve sıfır olay çakışması
    yeniden doğrulanır.
    """
    safe: List[Tuple[List[object], List[object], set]] = []
    for fold_number, (train_dates, validation_dates) in enumerate(
        expanding_folds(dates, folds), start=1
    ):
        overlap = positive_event_keys_for_dates(streams, train_dates) & positive_event_keys_for_dates(
            streams, validation_dates
        )
        purged_company_days: set = set()
        if overlap:
            purged_company_days = company_days_for_positive_events(
                streams, validation_dates, overlap
            )
            logging.warning(
                "İç rolling fold %d: %d ortak şirket--olay nedeniyle validasyondan "
                "%d şirket-gün purge edildi. Anahtarlar: %s",
                fold_number,
                len(overlap),
                len(purged_company_days),
                ", ".join(sorted(overlap)),
            )
        if not train_dates or not validation_dates:
            logging.warning("İç rolling fold %d purging sonrasında boş kaldı.", fold_number)
            continue
        remaining_overlap = positive_event_keys_for_dates(
            streams, train_dates
        ) & positive_event_keys_for_dates(
            streams, validation_dates, purged_company_days
        )
        if remaining_overlap:
            logging.warning(
                "İç rolling fold %d purging sonrasında hâlâ olay çakışması içeriyor: %s",
                fold_number,
                ", ".join(sorted(remaining_overlap)),
            )
            continue
        validation_positive, validation_negative = class_support_for_dates(
            streams, validation_dates, purged_company_days
        )
        if validation_positive == 0 or validation_negative == 0:
            logging.warning(
                "İç rolling fold %d purging sonrasında iki sınıf desteğini kaybetti: "
                "pozitif=%d, negatif=%d.",
                fold_number,
                validation_positive,
                validation_negative,
            )
            continue
        safe.append((train_dates, validation_dates, purged_company_days))
    if not safe:
        raise ValueError(
            "Purging sonrasında iki sınıflı ve olay çakışmasız iç rolling-origin "
            "fold üretilemedi. Logdaki olay kimliklerini ve kalan sınıf desteğini denetleyin."
        )
    return safe


@dataclass(frozen=True)
class OuterFold:
    """Bir ileri-zincirli dış test yılının eğitim ve test tarihlerini taşır."""

    test_year: int
    train_dates: Tuple[object, ...]
    test_dates: Tuple[object, ...]
    train_event_count: int
    test_event_count: int
    test_positive_days: int
    test_negative_days: int
    purged_overlap_event_count: int
    purged_test_company_day_count: int
    purged_test_company_days: Tuple[Tuple[str, object], ...]


def apply_data_variant(forum: pd.DataFrame, variant: str) -> pd.DataFrame:
    """Ana veri, kısa-ileti ve önceden tanımlı olay-dışlama profillerini üretir."""
    if variant in {
        "full_v4", "keep_all", "no_short_edges", "revised_tgn", "revised_tgn_long",
        "revised_no_activity", "revised_no_gate", "activity_controlled_tgn",
        "tgn_no_time_encoding", "tgn_daily_memory_reset",
    }:
        result = forum.copy()
    elif variant == "keep_all_drop_unknown":
        result = forum[~forum["Kullanici_Bilinmiyor"].astype(bool)].copy()
    elif variant == "min3_ablation":
        result = forum[forum["Kelime_Sayisi"] >= 3].copy()
    elif variant in DOMINANT_EVENT_EXCLUSIONS:
        company, event_id, _ = DOMINANT_EVENT_EXCLUSIONS[variant]
        event_values = forum["Olay_ID"].fillna("").astype(str).str.strip()
        exclusion_mask = (
            forum["Hisse_Kodu"].astype(str).str.upper().eq(company)
            & event_values.eq(event_id)
            & forum["SPK_Etiketi"].eq(1)
        )
        excluded_rows = int(exclusion_mask.sum())
        if excluded_rows == 0:
            raise ValueError(
                f"{variant} için beklenen {company}::{event_id} pozitif olayı bulunamadı."
            )
        result = forum.loc[~exclusion_mask].copy()
        logging.warning(
            "%s sağlamlık profilinde %s::%s olayına ait %d pozitif ileti çıkarıldı; "
            "bu iletiler negatif olarak yeniden etiketlenmedi.",
            variant,
            company,
            event_id,
            excluded_rows,
        )
    else:
        raise ValueError(f"Bilinmeyen veri varyantı: {variant}")
    if result.empty:
        raise ValueError(f"{variant} veri varyantı boş kaldı.")
    return result.reset_index(drop=True)


def write_variant_audit(frame: pd.DataFrame, variant: str, output_dir: Path) -> None:
    """Kısa ileti kararının sınıf, şirket ve yıl dağılımına etkisini kaydeder."""
    summary = (
        frame.assign(Yil=pd.to_datetime(frame["Date"]).dt.year)
        .groupby(["Yil", "SPK_Etiketi"], as_index=False)
        .agg(
            Mesaj=("Mesaj_ID", "size"),
            Kisa_Mesaj=("Kisa_Mesaj", "sum"),
            Bilinmeyen_Kullanici_Mesaji=("Kullanici_Bilinmiyor", "sum"),
            Sirket=("Hisse_Kodu", "nunique"),
            Sirket_Gun=("Date", "nunique"),
        )
    )
    summary.insert(0, "Data_Variant", variant)
    summary.to_csv(output_dir / f"variant_audit_{variant}.csv", index=False)


def build_outer_folds(forum: pd.DataFrame, config: ExperimentConfig) -> List[OuterFold]:
    """Takvim yılı bazlı, geçmişten eğitilen olay-çakışmasız fold'lar kurar.

    Bir pozitif ``Olay_ID`` yıl sınırını aşıp hem eğitim hem testte görünüyorsa
    kimlik yapay olarak bölünmez ve bütün test yılı da atılmaz. Çakışan olayın
    test tarafındaki yalnız ilgili şirket-günler purge edilir. Aynı tarihi
    paylaşan diğer şirketler korunur. Kalan test diliminde olay ayrıklığı ve
    önceden belirlenen iki sınıf desteği yeniden doğrulanır.
    """
    day_table = (
        forum.groupby(["Hisse_Kodu", "Date"], as_index=False)
        .agg(SPK_Etiketi=("SPK_Etiketi", "first"), Olay_ID=("Olay_ID", "first"))
    )
    day_table["Olay_ID"] = day_table["Olay_ID"].fillna("").astype(str).str.strip()
    day_table["Event_Key"] = np.where(
        day_table["SPK_Etiketi"] == 1,
        day_table["Hisse_Kodu"].astype(str) + "::" + day_table["Olay_ID"],
        "",
    )
    folds: List[OuterFold] = []
    for year in config.outer_test_years:
        train_days = day_table[pd.to_datetime(day_table["Date"]).dt.year < year]
        test_days = day_table[pd.to_datetime(day_table["Date"]).dt.year == year]
        if train_days.empty or test_days.empty:
            logging.warning("Dış fold %d boş eğitim veya test nedeniyle atlandı.", year)
            continue
        train_events = set(
            train_days.loc[train_days["SPK_Etiketi"] == 1, "Event_Key"]
        ) - {""}
        test_events = set(
            test_days.loc[test_days["SPK_Etiketi"] == 1, "Event_Key"]
        ) - {""}
        overlap = train_events & test_events
        purged_company_days: set = set()
        if overlap:
            purge_rows = test_days.loc[
                (test_days["SPK_Etiketi"] == 1)
                & test_days["Event_Key"].isin(overlap),
                ["Hisse_Kodu", "Date"],
            ]
            purged_company_days = set(
                (str(row.Hisse_Kodu), row.Date)
                for row in purge_rows.itertuples(index=False)
            )
            purge_index = pd.MultiIndex.from_tuples(
                purged_company_days, names=["Hisse_Kodu", "Date"]
            )
            test_index = pd.MultiIndex.from_frame(test_days[["Hisse_Kodu", "Date"]])
            test_days = test_days.loc[~test_index.isin(purge_index)].copy()
            logging.warning(
                "Dış fold %d: eğitim/test sınırını aşan %d şirket--olay nedeniyle "
                "testten %d şirket-gün purge edildi. Çakışan anahtarlar: %s",
                year,
                len(overlap),
                len(purged_company_days),
                ", ".join(sorted(overlap)),
            )
        if test_days.empty:
            logging.warning("Dış fold %d purging sonrasında boş kaldı.", year)
            continue
        remaining_test_events = set(
            test_days.loc[
                test_days["SPK_Etiketi"] == 1, "Event_Key"
            ].dropna().astype(str)
        ) - {""}
        remaining_overlap = train_events & remaining_test_events
        if remaining_overlap:
            raise RuntimeError(
                f"Dış fold {year} purging sonrasında olay çakışması koruyor: "
                + ", ".join(sorted(remaining_overlap))
            )
        positives = int((test_days["SPK_Etiketi"] == 1).sum())
        negatives = int((test_days["SPK_Etiketi"] == 0).sum())
        if (
            positives < config.minimum_test_positive_days
            or negatives < config.minimum_test_negative_days
        ):
            logging.warning(
                "Dış fold %d yetersiz destek nedeniyle atlandı: pozitif=%d negatif=%d.",
                year,
                positives,
                negatives,
            )
            continue
        folds.append(
            OuterFold(
                test_year=year,
                train_dates=tuple(sorted(train_days["Date"].unique())),
                test_dates=tuple(sorted(test_days["Date"].unique())),
                train_event_count=len(train_events),
                test_event_count=len(remaining_test_events),
                test_positive_days=positives,
                test_negative_days=negatives,
                purged_overlap_event_count=len(overlap),
                purged_test_company_day_count=len(purged_company_days),
                purged_test_company_days=tuple(sorted(purged_company_days)),
            )
        )
    if not folds:
        raise ValueError(
            "Geçerli ileri-zincirli dış fold üretilemedi. Olay_ID sınırlarını ve "
            "test sınıf desteklerini denetleyin."
        )
    return folds


def select_filter_parameters(
    validation_scores: Mapping[Tuple[int, float], List[dict]]
) -> Tuple[int, float]:
    """Validasyon backtestlerinden budama yüzdeliği ve duyarlılık eşiği seçer.

    Her parametre çifti için filtreli stratejinin mutlak validasyon Sharpe'ı
    birincil, ham stratejiye göre Sharpe farkı ikincil ölçüttür. Böylece mutlak
    olarak çok kötü bir strateji yalnız göreli farkı küçük ölçüde pozitif diye
    seçilmez. Test sonuçları bu seçime girmez.

    Çıktı: ``(pruning_percentile, sentiment_threshold)`` ikilisi.
    """
    ranked = []
    for parameters, values in validation_scores.items():
        delta_values = np.asarray([value["Delta_Sharpe"] for value in values], dtype=float)
        filtered_values = np.asarray([value["Filtered_Sharpe"] for value in values], dtype=float)
        delta_sharpe = (
            float(np.mean(delta_values[np.isfinite(delta_values)]))
            if np.isfinite(delta_values).any()
            else -np.inf
        )
        filtered_sharpe = (
            float(np.mean(filtered_values[np.isfinite(filtered_values)]))
            if np.isfinite(filtered_values).any()
            else -np.inf
        )
        ranked.append((filtered_sharpe, delta_sharpe, parameters))
        logging.info(
            "Validasyon parametresi prune=%d theta=%.2f: DeltaSharpe=%.4f, FilteredSharpe=%.4f",
            parameters[0],
            parameters[1],
            delta_sharpe,
            filtered_sharpe,
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not ranked or not np.isfinite(ranked[0][0]):
        raise ValueError(
            "Hiçbir validasyon parametresi için tanımlı Sharpe farkı hesaplanamadı. "
            "Piyasa eşleşmesi, günlük getiri varyansı ve sinyal kapsamı denetlenmelidir."
        )
    if ranked[0][0] <= 0:
        logging.warning(
            "Bütün filtre adaylarının ortalama validasyon Sharpe'ı sıfır veya altında. "
            "Seçilen ayar yalnız önceden tanımlı karşılaştırma için kullanılacak; "
            "ekonomik başarı iddiası kurulmayacak."
        )
    return ranked[0][2]


def rolling_origin_selection(
    streams: Mapping[str, CompanyStream],
    development_dates: Sequence[object],
    market: pd.DataFrame,
    sentiment: TurkishSentiment,
    device: torch.device,
    config: ExperimentConfig,
    loss_mode: str = "bce",
) -> Tuple[Tuple[int, float], pd.DataFrame]:
    """Geliştirme döneminde seed ve zaman fold'ları üzerinden model seçimi yürütür.

    Her seed/fold için modeller yalnız geçmiş eğitim tarihlerinde eğitilir,
    validasyon tarihlerinde değerlendirilir ve model-spesifik eşikler belirlenir.
    Ham/filtreli duyarlılık backtestleri parametre ızgarasının her elemanı için
    yalnız validasyonda hesaplanır. Böylece kilitli test seçim sürecine sızmaz.

    Çıktılar: Seçilen downstream parametre çifti ve bütün seed/fold/model
    sınıflandırma metriklerini içeren DataFrame. Uzun aşamalar ilerleme gösterir.
    """
    folds = event_safe_expanding_folds(streams, development_dates, config.rolling_folds)
    parameter_scores: MutableMapping[Tuple[int, float], List[dict]] = defaultdict(list)
    metric_rows: List[dict] = []
    pruning_values = sorted({parameters[0] for parameters in config.parameter_grid})
    fold_progress = ProgressReporter(
        "Rolling-origin model seçimi",
        len(config.seeds) * len(folds),
        log_interval_seconds=20,
    )

    for seed in config.seeds:
        for fold_number, (
            train_dates,
            validation_dates,
            excluded_validation_company_days,
        ) in enumerate(folds, start=1):
            logging.info(
                "Rolling-origin seed=%d fold=%d: train %s..%s, val %s..%s",
                seed,
                fold_number,
                train_dates[0],
                train_dates[-1],
                validation_dates[0],
                validation_dates[-1],
            )
            models, memory = train_models(
                streams, train_dates, seed, device, config, loss_mode=loss_mode
            )
            validation_records = evaluate_models(
                models,
                memory,
                streams,
                validation_dates,
                device,
                progress_label=f"Seed {seed} fold {fold_number} validasyonu",
                excluded_company_days=excluded_validation_company_days,
            )
            thresholds = model_thresholds(validation_records)
            labels = [record["Label"] for record in validation_records]
            for model_name, probability_key in (
                ("TGN", "TGN_Prob"),
                ("Text", "Text_Prob"),
                ("Graph", "Graph_Prob"),
            ):
                metrics = classification_metrics(
                    labels,
                    [record[probability_key] for record in validation_records],
                    thresholds[model_name],
                )
                metric_rows.append(
                    {
                        "Seed": seed,
                        "Fold": fold_number,
                        "Loss_Mode": loss_mode,
                        "Model": model_name,
                        **metrics,
                    }
                )

            attach_sentiment_scores(
                validation_records,
                streams,
                sentiment,
                thresholds["TGN"],
                pruning_values,
                progress_label=f"Seed {seed} fold {fold_number} duyarlılık",
            )
            for pruning, sentiment_threshold in config.parameter_grid:
                raw, filtered, _ = signals_from_records(
                    validation_records, pruning, sentiment_threshold
                )
                raw_metrics, _ = run_backtest(
                    raw, market, config.selection_round_trip_cost
                )
                filtered_metrics, _ = run_backtest(
                    filtered, market, config.selection_round_trip_cost
                )
                parameter_scores[(pruning, sentiment_threshold)].append(
                    {
                        "Delta_Sharpe": filtered_metrics["Sharpe"] - raw_metrics["Sharpe"],
                        "Filtered_Sharpe": filtered_metrics["Sharpe"],
                    }
                )
            fold_progress.update(detail=f"seed {seed}, fold {fold_number} tamamlandı")
    fold_progress.finish(detail="rolling-origin seçimi tamamlandı")
    selected = select_filter_parameters(parameter_scores)
    return selected, pd.DataFrame(metric_rows)


def summarize_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Seed düzeyindeki sınıflandırma ölçülerini model bazında özetler.

    AUPRC, AUROC, kesinlik, duyarlılık, F1 ve Brier için aritmetik ortalama ile
    örnek standart sapmayı hesaplar. Ham seed sonuçlarını değiştirmez.
    Çıktı, makale tablolarına uygun model-satırlı özet DataFrame'dir.
    """
    rows: List[dict] = []
    for model, model_frame in frame.groupby("Model"):
        row = {"Model": model}
        for metric in ("AUPRC", "AUROC", "Precision", "Recall", "F1", "Brier"):
            values = model_frame[metric].astype(float)
            row[f"{metric}_Mean"] = float(values.mean())
            row[f"{metric}_SD"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def weighted_auprc(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> float:
    """Bootstrap tekrarında gözlem ağırlıklarıyla AUPRC hesaplar.

    Sıfır ağırlıklı satırlar dışlanır. Etkin örneklem tek sınıfa düştüğünde AUPRC
    tanımsız olduğundan ``NaN`` döndürülür; aksi hâlde sklearn'in ağırlıklı
    average precision ölçüsü kullanılır.
    """
    mask = weights > 0
    if len(np.unique(labels[mask])) < 2:
        return np.nan
    return float(
        average_precision_score(labels[mask], probabilities[mask], sample_weight=weights[mask])
    )


def two_way_cluster_bootstrap_difference(
    frame: pd.DataFrame,
    first_probability: str,
    second_probability: str,
    repetitions: int,
    seed: int = 2024,
    progress_label: str = "İki yönlü bootstrap",
) -> dict:
    """İki modelin AUPRC farkı için tarih ve şirket kümeli bootstrap uygular.

    Her tekrarda tarihler ve şirketler bağımsız olarak yerine koymalı örneklenir;
    iki örnekleme frekansının çarpımı şirket-gün ağırlığını oluşturur. Aynı ağırlık
    iki model olasılığına uygulanarak eşleştirilmiş AUPRC farkı hesaplanır. Bu
    yaklaşım hem gün içi ortak şokları hem şirket içi bağımlılığı yaklaşık korur.

    Çıktı: Ortalama fark ile yüzde 95 yüzdelik güven aralığını içeren sözlük.
    Geçersiz tek-sınıflı tekrarlar atlanır ve ilerleme/ETA raporlanır.
    """
    generator = np.random.default_rng(seed)
    dates = frame["Tarih"].unique()
    companies = frame["Hisse_Kodu"].unique()
    labels = frame["Label"].to_numpy(dtype=int)
    first = frame[first_probability].to_numpy(dtype=float)
    second = frame[second_probability].to_numpy(dtype=float)
    differences: List[float] = []
    progress = ProgressReporter(progress_label, repetitions, log_interval_seconds=20)
    for _ in range(repetitions):
        sampled_dates = generator.choice(dates, size=len(dates), replace=True)
        sampled_companies = generator.choice(companies, size=len(companies), replace=True)
        date_counts = pd.Series(sampled_dates).value_counts()
        company_counts = pd.Series(sampled_companies).value_counts()
        weights = (
            frame["Tarih"].map(date_counts).fillna(0).to_numpy(dtype=float)
            * frame["Hisse_Kodu"].map(company_counts).fillna(0).to_numpy(dtype=float)
        )
        first_score = weighted_auprc(labels, first, weights)
        second_score = weighted_auprc(labels, second, weights)
        if np.isfinite(first_score) and np.isfinite(second_score):
            differences.append(first_score - second_score)
        progress.update()
    progress.finish(detail="bootstrap tamamlandı")
    if not differences:
        return {
            "Mean_Difference": np.nan,
            "CI_Low": np.nan,
            "CI_High": np.nan,
            "P_Value_Two_Sided": np.nan,
        }
    values = np.asarray(differences, dtype=float)
    lower_tail = (float(np.sum(values <= 0.0)) + 1.0) / (len(values) + 1.0)
    upper_tail = (float(np.sum(values >= 0.0)) + 1.0) / (len(values) + 1.0)
    return {
        "Mean_Difference": float(np.mean(values)),
        "CI_Low": float(np.percentile(values, 2.5)),
        "CI_High": float(np.percentile(values, 97.5)),
        "P_Value_Two_Sided": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
    }


def add_holm_adjustment(
    frame: pd.DataFrame, p_column: str = "P_Value_Two_Sided"
) -> pd.DataFrame:
    """Sonlu bootstrap p-değerlerine Holm düzeltmesi ekler."""
    result = frame.copy()
    result["P_Holm"] = np.nan
    if p_column not in result.columns:
        return result
    valid = result[p_column].notna() & np.isfinite(result[p_column])
    ordered = result.loc[valid, p_column].sort_values()
    running = 0.0
    count = len(ordered)
    for rank, (index, value) in enumerate(ordered.items()):
        running = max(running, min(1.0, float(value) * (count - rank)))
        result.loc[index, "P_Holm"] = running
    return result


def write_manifest(
    output_dir: Path,
    forum_path: Path,
    market_path: Path,
    locked_test_start: str,
    selected_parameters: Tuple[int, float],
    config: ExperimentConfig,
    confirmatory: bool,
) -> None:
    """Deneyin veri sürümü, seçim parametreleri ve test politikasını JSON'a yazar.

    Forum/piyasa SHA-256 karmaları, kilitli test başlangıcı, validasyonda seçilen
    budama ve duyarlılık eşikleri ile bütün ``ExperimentConfig`` alanları kayda
    alınır. Doğrulayıcı ve keşifsel koşular farklı dosya adı ve açık test politikası
    kullanır.

    Yan etki: ``locked_test_manifest.json`` veya ``exploratory_manifest.json``
    üretir. Model tahmini veya metrik hesaplaması yapmaz.
    """
    manifest = {
        "forum_sha256": file_sha256(forum_path),
        "market_sha256": file_sha256(market_path),
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "dominant_event_exclusions": DOMINANT_EVENT_EXCLUSIONS,
        "locked_test_start": locked_test_start,
        "selected_pruning_percentile": selected_parameters[0],
        "selected_sentiment_threshold": selected_parameters[1],
        "config": asdict(config),
        "test_policy": (
            "single locked holdout; no parameter selection on test"
            if confirmatory
            else "exploratory analysis; test period was previously inspected"
        ),
    }
    manifest_name = "locked_test_manifest.json" if confirmatory else "exploratory_manifest.json"
    (output_dir / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )




# -----------------------------------------------------------------------------
# V4 RESUME / ATOMİK KAYIT YARDIMCILARI
# -----------------------------------------------------------------------------
def _stable_payload_hash(payload: object) -> str:
    """JSON-uyumlu bir nesne için kararlı SHA-256 üretir."""
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    """Metni aynı dizindeki geçici dosyaya yazıp atomik olarak hedefe taşır."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
    )


def _atomic_write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    """CSV'yi yarım dosya bırakmayacak biçimde atomik kaydeder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _read_json_if_valid(path: Path) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _resume_identity(
    config_hash: str,
    forum_hash: str,
    market_hash: str,
    **extra: object,
) -> dict:
    identity = {
        "config_sha256": config_hash,
        "forum_sha256": forum_hash,
        "market_sha256": market_hash,
    }
    identity.update(extra)
    return identity


def _identity_matches(marker: Optional[dict], expected: Mapping[str, object]) -> bool:
    return bool(marker) and all(str(marker.get(key)) == str(value) for key, value in expected.items())


def _prepare_resume_manifest(
    output_dir: Path,
    manifest: dict,
    config_hash: str,
    resume_run: bool,
) -> None:
    """Aynı klasörde yalnız aynı veri ve config ile devam edilmesine izin verir."""
    path = output_dir / "v4_experiment_manifest.json"
    existing = _read_json_if_valid(path)
    if existing is not None:
        existing_config_hash = existing.get("resume_config_sha256")
        if not existing_config_hash:
            existing_config_hash = _stable_payload_hash(existing.get("config", {}))
        mismatches = []
        if existing.get("forum_sha256") != manifest.get("forum_sha256"):
            mismatches.append("forum SHA-256")
        if existing.get("market_sha256") != manifest.get("market_sha256"):
            mismatches.append("piyasa SHA-256")
        if existing_config_hash != config_hash:
            mismatches.append("deney config'i")
        if mismatches:
            raise RuntimeError(
                "Resume reddedildi; mevcut çıktı klasörü farklı bir koşuya ait: "
                + ", ".join(mismatches)
            )
        if not resume_run:
            raise RuntimeError(
                "Çıktı klasöründe daha önce başlatılmış bir V4 koşusu var. "
                "Devam etmek için RESUME_RUN=True yapın veya yeni OUTPUT_DIR kullanın."
            )
        manifest = dict(existing)
        manifest["last_resumed_at_utc"] = pd.Timestamp.utcnow().isoformat()
        manifest["resume_enabled"] = True
        manifest["resume_config_sha256"] = config_hash
        _atomic_write_json(path, manifest)
        logging.warning("RESUME etkin: mevcut V4 manifesti doğrulandı.")
        return

    manifest = dict(manifest)
    manifest["resume_enabled"] = bool(resume_run)
    manifest["resume_config_sha256"] = config_hash
    _atomic_write_json(path, manifest)


def _atomic_save_model_checkpoint(path: Path, *args: object, **kwargs: object) -> None:
    """Torch checkpointini önce geçici dosyaya yazar, sonra atomik taşır."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        save_model_checkpoint(temporary, *args, **kwargs)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _normalise_cached_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    if not frame.empty and "Olay_ID" in frame.columns:
        # CSV tür çıkarımı, yalnız sayısal olay kimliği içeren bir seed dosyasını
        # float olarak okuyup örn. ``3298563`` değerini ``3298563.0`` yapabilir.
        # Seed tahminleri birleştirilmeden önce bu gösterim farkını kaldırmak aynı
        # şirket--olayın olay-makro ve leave-one-event-out analizinde iki ayrı
        # olaymış gibi sayılmasını önler.
        frame = frame.copy()
        frame["Olay_ID"] = (
            frame["Olay_ID"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
    if not frame.empty and "Tarih" in frame.columns:
        parsed = pd.to_datetime(frame["Tarih"], errors="coerce")
        if parsed.notna().all():
            frame = frame.copy()
            frame["Tarih"] = parsed.dt.date
    return frame


def _load_seed_cache(
    experiment_dir: Path,
    seed: int,
    expected_identity: Mapping[str, object],
    require_checkpoint: bool,
) -> Optional[Tuple[List[dict], List[dict], List[dict]]]:
    seed_dir = experiment_dir / "seed_results"
    marker_path = seed_dir / f"seed_{seed}.complete.json"
    marker = _read_json_if_valid(marker_path)
    if not _identity_matches(marker, expected_identity):
        return None
    metrics_path = seed_dir / f"seed_{seed}_metrics.csv"
    predictions_path = seed_dir / f"seed_{seed}_predictions.csv"
    financial_path = seed_dir / f"seed_{seed}_financial.csv"
    checkpoint_path = experiment_dir / "saved_models" / f"seed_{seed}.pt"
    required = [metrics_path, predictions_path, financial_path]
    if require_checkpoint:
        required.append(checkpoint_path)
    if not all(path.exists() and path.stat().st_size > 0 for path in required):
        logging.warning("Seed %d işareti var ancak dosyaları eksik; seed yeniden çalışacak.", seed)
        return None
    try:
        metrics = pd.read_csv(metrics_path).to_dict("records")
        predictions_frame = _normalise_cached_predictions(pd.read_csv(predictions_path))
        predictions = predictions_frame.to_dict("records")
        financial = pd.read_csv(financial_path).to_dict("records")
    except Exception as exc:
        logging.warning("Seed %d önbelleği okunamadı (%s); yeniden çalışacak.", seed, exc)
        return None
    logging.warning("RESUME: seed %d tamamlanmış; eğitim ve değerlendirme atlandı.", seed)
    return metrics, predictions, financial


def _write_seed_cache(
    experiment_dir: Path,
    seed: int,
    metrics: Sequence[dict],
    predictions: Sequence[dict],
    financial: Sequence[dict],
    identity: Mapping[str, object],
) -> None:
    """Seed çıktıları tamamlandıktan sonra en son tamamlanma işaretini yazar."""
    seed_dir = experiment_dir / "seed_results"
    _atomic_write_dataframe(
        pd.DataFrame(metrics), seed_dir / f"seed_{seed}_metrics.csv"
    )
    _atomic_write_dataframe(
        pd.DataFrame(predictions), seed_dir / f"seed_{seed}_predictions.csv"
    )
    _atomic_write_dataframe(
        pd.DataFrame(financial), seed_dir / f"seed_{seed}_financial.csv"
    )
    marker = dict(identity)
    marker["completed_at_utc"] = pd.Timestamp.utcnow().isoformat()
    _atomic_write_json(seed_dir / f"seed_{seed}.complete.json", marker)


def _load_experiment_cache(
    experiment_dir: Path,
    expected_identity: Mapping[str, object],
) -> Optional[Tuple[List[dict], List[dict], List[dict]]]:
    marker = _read_json_if_valid(experiment_dir / "EXPERIMENT_COMPLETED.json")
    if not _identity_matches(marker, expected_identity):
        return None
    metrics_path = experiment_dir / "test_metrics_by_seed.csv"
    predictions_path = experiment_dir / "test_predictions.csv"
    financial_path = experiment_dir / "financial_results_by_seed.csv"
    if not all(path.exists() and path.stat().st_size > 0 for path in (metrics_path, predictions_path, financial_path)):
        return None
    try:
        metrics = pd.read_csv(metrics_path).to_dict("records")
        predictions = _normalise_cached_predictions(pd.read_csv(predictions_path)).to_dict("records")
        financial = pd.read_csv(financial_path).to_dict("records")
    except Exception as exc:
        logging.warning("Deney önbelleği okunamadı (%s); deney yeniden kurulacak.", exc)
        return None
    logging.warning("RESUME: tamamlanmış deney atlandı: %s", experiment_dir)
    return metrics, predictions, financial


def _write_shareable_progress_bundle(
    output_dir: Path,
    all_metrics: Sequence[Mapping[str, object]],
    all_financial: Sequence[Mapping[str, object]],
    all_fold_rows: Sequence[Mapping[str, object]],
    current: Mapping[str, object],
) -> None:
    """Devam eden koşudan küçük ve paylaşılabilir bir ara sonuç ZIP'i üretir.

    Büyük tahmin tabloları ve model ağırlıkları pakete alınmaz. Paket her seed
    ve deney tamamlandığında aynı dosyanın üzerine atomik biçimde yazılır;
    böylece çalışma durdurulmadan makale güncellemesi için paylaşılabilir.
    """
    staging = output_dir / "_shareable_progress"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    metrics_frame = pd.DataFrame(all_metrics)
    financial_frame = pd.DataFrame(all_financial)
    fold_frame = pd.DataFrame(all_fold_rows)
    _atomic_write_dataframe(metrics_frame, staging / "partial_test_metrics_by_seed.csv")
    _atomic_write_dataframe(fold_frame, staging / "partial_outer_fold_manifest.csv")
    if not metrics_frame.empty:
        metric_summary = (
            metrics_frame.groupby(
                ["Data_Variant", "Test_Year", "Loss_Mode", "Model"],
                as_index=False,
            )
            .agg(
                AUPRC_Mean=("AUPRC", "mean"),
                AUPRC_SD=("AUPRC", "std"),
                AUROC_Mean=("AUROC", "mean"),
                Brier_Mean=("Brier", "mean"),
                F1_Mean=("F1", "mean"),
                Seeds=("Seed", "nunique"),
            )
        )
        _atomic_write_dataframe(
            metric_summary, staging / "partial_test_metrics_summary.csv"
        )
    if not financial_frame.empty:
        _atomic_write_dataframe(
            financial_frame, staging / "partial_financial_results_by_seed.csv"
        )
        financial_summary = (
            financial_frame.groupby(
                ["Data_Variant", "Test_Year", "Loss_Mode", "Cost", "Strategy"],
                as_index=False,
            )
            .agg(
                Return_Mean=("Return", "mean"),
                Sharpe_Mean=("Sharpe", "mean"),
                MDD_Mean=("MDD", "mean"),
                Trades_Mean=("Trades", "mean"),
                Seeds=("Seed", "nunique"),
            )
        )
        _atomic_write_dataframe(
            financial_summary, staging / "partial_financial_results_summary.csv"
        )

    for source_name in ("v4_experiment_manifest.json", "v4_experiment_profiles.csv"):
        source = output_dir / source_name
        if source.exists():
            shutil.copy2(source, staging / source_name)

    macro_paths = sorted(output_dir.glob("*/test_*/*/generalization_macro_metrics.csv"))
    macro_frames = [pd.read_csv(path) for path in macro_paths if path.stat().st_size > 0]
    if macro_frames:
        _atomic_write_dataframe(
            pd.concat(macro_frames, ignore_index=True),
            staging / "partial_event_macro_metrics.csv",
        )
    loo_paths = sorted(output_dir.glob("*/test_*/*/leave_one_positive_event_out.csv"))
    loo_frames = [pd.read_csv(path) for path in loo_paths if path.stat().st_size > 0]
    loo_frames = [frame for frame in loo_frames if not frame.empty]
    if loo_frames:
        _atomic_write_dataframe(
            pd.concat(loo_frames, ignore_index=True),
            staging / "partial_leave_one_positive_event_out.csv",
        )

    completed_experiments = len(
        list(output_dir.glob("*/test_*/*/EXPERIMENT_COMPLETED.json"))
    )
    completed_seeds = len(list(output_dir.glob("*/test_*/*/seed_results/seed_*.complete.json")))
    status = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "experiment_scope": EXPERIMENT_SCOPE,
        "completed_experiments": completed_experiments,
        "completed_seeds": completed_seeds,
        "current": dict(current),
        "note": "Ara sonuçtur; yalnız tamamlanmış seed ve deneyleri içerir.",
    }
    _atomic_write_json(staging / "PROGRESS_STATUS.json", status)

    log_path = output_dir / "experiment.log"
    if log_path.exists():
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            log_tail = handle.readlines()[-500:]
        (staging / "experiment_log_tail.txt").write_text(
            "".join(log_tail), encoding="utf-8"
        )

    archive_base = output_dir / "SHAREABLE_PROGRESS.tmp"
    archive_path = Path(
        shutil.make_archive(str(archive_base), "zip", root_dir=staging)
    )
    archive_path.replace(output_dir / "SHAREABLE_PROGRESS.zip")
    shutil.rmtree(staging)


def run_v4_experiment_suite(args: argparse.Namespace) -> None:
    """V4 deney matrisini sırasıyla çalıştırır ve her ara sonucu diske yazar.

    Sıra: veri sözleşmesi ve kısa-ileti varyantı; olay çakışmasız yıllık dış
    fold; iç rolling-origin seçim; BCE ve nnPU eğitimleri; metin/statik/TGN
    metrikleri; sert ve yumuşak koordinasyon filtresi; uzun/nakit, kesitsel top-k
    ve her-zaman-uzun finansal değerlendirme; seed/fold toplulaştırması ve
    bootstrap. İncelenmiş tarihsel yılların tamamı keşifsel olarak işaretlenir.
    """
    output_dir = Path(args.output_dir)
    configure_logging(output_dir)
    started_at = time.perf_counter()
    forum_path = Path(args.forum_csv)
    market_path = Path(args.market_csv)

    if QUICK_SMOKE_TEST:
        config = ExperimentConfig(
            seeds=(42,),
            epochs=1,
            rolling_folds=2,
            bootstrap_repetitions=50,
            # Önce bütün aday yılların olay çakışması denetlenir. Aşağıda
            # bunlar arasından kronolojik olarak en son geçerli fold seçilir.
            # Yalnız takvimdeki son yılı vermek, o yılı aşan tek bir olay
            # bulunduğunda duman testini gereksiz biçimde boş bırakabilir.
            outer_test_years=OUTER_TEST_YEARS,
            data_variants=("keep_all", "revised_tgn"),
        )
        logging.warning("QUICK_SMOKE_TEST etkin: sonuçlar bilimsel rapora alınamaz.")
    else:
        configured_variants = DATA_VARIANTS
        if not RUN_REVISED_COMPONENT_ABLATIONS:
            configured_variants = tuple(
                value for value in DATA_VARIANTS
                if value not in {"revised_no_activity", "revised_no_gate"}
            )
        config = ExperimentConfig(
            epochs=args.epochs,
            rolling_folds=args.rolling_folds,
            bootstrap_repetitions=args.bootstrap_repetitions,
            data_variants=configured_variants,
            experiment_scope=EXPERIMENT_SCOPE,
        )

    forum_all = prepare_forum(forum_path)
    market = prepare_market(market_path, output_dir)
    write_unknown_user_audit(forum_all, output_dir)
    write_label_structure_audit(forum_all, output_dir)
    market_scale_suspicious = write_market_regime_audit(
        forum_all, market, output_dir, "preflight"
    )
    if not market_scale_suspicious.empty:
        examples = ", ".join(
            f"{row.Hisse_Kodu}/{int(row.Yil)}"
            for row in market_scale_suspicious.head(8).itertuples(index=False)
        )
        raise ValueError(
            "Piyasa fiyat ölçeği denetimi başarısız. En az 10 gözlemli bazı "
            "şirket-yıllarda mutlak ortalama veya medyan gün içi getiri %8'i "
            "aşıyor. T1_Acilis ve T1_Kapanis aynı gün ve aynı düzeltilmiş/"
            "düzeltilmemiş ölçekte olmalıdır. Şüpheli örnekler: " + examples
        )
    forum_hash = file_sha256(forum_path)
    market_hash = file_sha256(market_path)
    config_hash = _stable_payload_hash(asdict(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("V4 hesaplama cihazı: %s", device)

    root_manifest = {
        "pipeline_version": 4,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "status": "historical_exploratory",
        "forum_sha256": forum_hash,
        "market_sha256": market_hash,
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "purging_unit": "company_date_for_overlapping_company_event",
        "dominant_event_exclusions": DOMINANT_EVENT_EXCLUSIONS,
        "config": asdict(config),
        "experiment_order": [
            "data_audit",
            "event_disjoint_forward_folds",
            "bce_baselines",
            "hard_filter",
            "soft_filter",
            "long_cash_backtest",
            "cross_sectional_top_fraction_backtest",
            "aggregate_and_bootstrap",
        ] if EXPERIMENT_SCOPE == "full_v4" else [
            "data_audit",
            "short_message_ablation",
            "event_disjoint_forward_folds",
            "dominant_event_exclusion_ablations",
            "bce_baselines",
            "nnpu_models",
            "hard_filter",
            "soft_filter",
            "long_cash_backtest",
            "cross_sectional_top_fraction_backtest",
            "aggregate_and_bootstrap",
        ],
    }
    _prepare_resume_manifest(
        output_dir, root_manifest, config_hash, RESUME_RUN
    )
    experiment_profile_table(config).to_csv(
        output_dir / "v4_experiment_profiles.csv", index=False
    )
    if RESUME_RUN and (output_dir / "V4_COMPLETED").exists():
        logging.warning("V4_COMPLETED bulundu; tamamlanmış koşu yeniden çalıştırılmadı.")
        return
    (output_dir / "HISTORICAL_EXPLORATORY").write_text(
        "The 2021--2025 periods were used during method development. "
        "They are exploratory forward tests, not a new confirmatory holdout.\n",
        encoding="utf-8",
    )

    extractor = FeatureExtractors(device, config)
    sentiment = TurkishSentiment(device, config)
    all_metrics: List[dict] = []
    all_financial: List[dict] = []
    all_predictions: List[dict] = []
    all_fold_rows: List[dict] = []
    structural_stream_cache: Dict[str, Mapping[str, CompanyStream]] = {}

    for variant in config.data_variants:
        logging.info("[V4] Veri/model profili başlıyor: %s", variant)
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        forum = apply_data_variant(forum_all, variant)
        write_variant_audit(forum, variant, variant_dir)
        write_event_concentration_audit(forum, variant_dir, variant)
        write_market_regime_audit(forum, market, variant_dir, variant)
        structural_key = (
            "keep_all_edges" if variant in {
                "full_v4", "keep_all", "activity_controlled_tgn",
                "tgn_no_time_encoding", "tgn_daily_memory_reset",
            }
            else "drop_unknown_keep_all_edges" if variant == "keep_all_drop_unknown"
            else "min3_no_short_edges" if variant == "min3_ablation"
            else variant if variant in DOMINANT_EVENT_EXCLUSIONS
            else "keep_all_no_short_edges"
        )
        if structural_key in structural_stream_cache:
            streams = {
                company: replace(stream, experiment_profile=variant)
                for company, stream in structural_stream_cache[structural_key].items()
            }
        else:
            streams = build_streams(
                forum, extractor, config, experiment_profile=variant
            )
            structural_stream_cache[structural_key] = streams
        write_edge_audit(streams, variant_dir / "edge_relation_audit.csv")
        outer_folds = build_outer_folds(forum, config)
        if variant in DOMINANT_EVENT_EXCLUSIONS:
            _, _, event_year = DOMINANT_EVENT_EXCLUSIONS[variant]
            outer_folds = [fold for fold in outer_folds if fold.test_year >= event_year]
            logging.warning(
                "%s yalnız olay yılı ve sonraki dış foldlarda çalıştırılacak: %s",
                variant,
                [fold.test_year for fold in outer_folds],
            )
        if QUICK_SMOKE_TEST:
            outer_folds = [outer_folds[-1]]
            logging.warning(
                "QUICK_SMOKE_TEST için en son geçerli olay-çakışmasız dış fold seçildi: %d",
                outer_folds[0].test_year,
            )
        for fold in outer_folds:
            all_fold_rows.append({"Data_Variant": variant, **asdict(fold)})
            fold_dir = variant_dir / f"test_{fold.test_year}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            logging.info(
                "[V4] Dış fold %s/%d: train gün=%d, test gün=%d, test olay=%d",
                variant,
                fold.test_year,
                len(fold.train_dates),
                len(fold.test_dates),
                fold.test_event_count,
            )
            if len(fold.train_dates) <= config.rolling_folds + 1:
                logging.warning("%d fold'u yetersiz eğitim tarihi nedeniyle atlandı.", fold.test_year)
                continue

            loss_modes = profile_loss_modes(variant, config)
            for loss_mode in loss_modes:
                experiment_dir = fold_dir / loss_mode
                experiment_dir.mkdir(parents=True, exist_ok=True)
                experiment_identity = _resume_identity(
                    config_hash, forum_hash, market_hash,
                    data_variant=variant, test_year=fold.test_year, loss_mode=loss_mode,
                )
                cached_experiment = (
                    _load_experiment_cache(experiment_dir, experiment_identity)
                    if RESUME_RUN else None
                )
                if cached_experiment is not None:
                    cached_metrics, cached_predictions, cached_financial = cached_experiment
                    all_metrics.extend(cached_metrics)
                    all_predictions.extend(cached_predictions)
                    all_financial.extend(cached_financial)
                    continue

                rolling_state_path = experiment_dir / "rolling_selection_state.json"
                rolling_metrics_path = experiment_dir / "rolling_validation_metrics.csv"
                rolling_state = (
                    _read_json_if_valid(rolling_state_path) if RESUME_RUN else None
                )
                if (
                    _identity_matches(rolling_state, experiment_identity)
                    and rolling_metrics_path.exists()
                    and rolling_metrics_path.stat().st_size > 0
                ):
                    selected_parameters = (
                        int(rolling_state["pruning_percentile"]),
                        float(rolling_state["sentiment_threshold"]),
                    )
                    rolling_metrics = pd.read_csv(rolling_metrics_path)
                    logging.warning("RESUME: rolling-origin seçimi önbellekten yüklendi: %s", experiment_dir)
                else:
                    selected_parameters, rolling_metrics = rolling_origin_selection(
                    streams,
                    fold.train_dates,
                    market,
                    sentiment,
                    device,
                    config,
                    loss_mode=loss_mode,
                    )
                    _atomic_write_dataframe(rolling_metrics, rolling_metrics_path)
                    rolling_marker = dict(experiment_identity)
                    rolling_marker.update(
                        {
                            "pruning_percentile": int(selected_parameters[0]),
                            "sentiment_threshold": float(selected_parameters[1]),
                            "completed_at_utc": pd.Timestamp.utcnow().isoformat(),
                        }
                    )
                    _atomic_write_json(rolling_state_path, rolling_marker)
                pruning, sentiment_threshold = selected_parameters
                (
                    final_train_dates,
                    final_validation_dates,
                    final_validation_excluded_company_days,
                ) = event_safe_expanding_folds(
                    streams, fold.train_dates, config.rolling_folds
                )[-1]

                experiment_metrics: List[dict] = []
                experiment_predictions: List[dict] = []
                experiment_financial: List[dict] = []
                for seed in config.seeds:
                    seed_identity = dict(experiment_identity)
                    seed_identity["seed"] = int(seed)
                    cached_seed = (
                        _load_seed_cache(
                            experiment_dir, seed, seed_identity, bool(SAVE_MODELS)
                        )
                        if RESUME_RUN else None
                    )
                    if cached_seed is not None:
                        seed_metrics, seed_predictions, seed_financial = cached_seed
                        experiment_metrics.extend(seed_metrics)
                        experiment_predictions.extend(seed_predictions)
                        experiment_financial.extend(seed_financial)
                        all_metrics.extend(seed_metrics)
                        all_predictions.extend(seed_predictions)
                        all_financial.extend(seed_financial)
                        continue

                    models, memory = train_models(
                        streams,
                        final_train_dates,
                        seed,
                        device,
                        config,
                        loss_mode=loss_mode,
                    )
                    validation_records = evaluate_models(
                        models,
                        memory,
                        streams,
                            final_validation_dates,
                            device,
                            progress_label=(
                                f"{variant}/{fold.test_year}/{loss_mode}/seed{seed} validasyon"
                            ),
                            excluded_company_days=(
                                final_validation_excluded_company_days
                            ),
                        )
                    thresholds = model_thresholds(validation_records)
                    validation_labels = [record["Label"] for record in validation_records]
                    validation_auprc = (
                        float(
                            average_precision_score(
                                validation_labels,
                                [record["TGN_Prob"] for record in validation_records],
                            )
                        )
                        if len(set(validation_labels)) == 2
                        else float("nan")
                    )
                    test_records = evaluate_models(
                        models,
                        memory,
                        streams,
                        fold.test_dates,
                        device,
                        progress_label=(
                            f"{variant}/{fold.test_year}/{loss_mode}/seed{seed} test"
                        ),
                        excluded_company_days=fold.purged_test_company_days,
                    )
                    labels = [record["Label"] for record in test_records]
                    for model_name, probability_key in (
                        ("TGN", "TGN_Prob"),
                        ("Text", "Text_Prob"),
                        ("Graph", "Graph_Prob"),
                    ):
                        metrics = classification_metrics(
                            labels,
                            [record[probability_key] for record in test_records],
                            thresholds[model_name],
                        )
                        row = {
                            "Data_Variant": variant,
                            "Test_Year": fold.test_year,
                            "Loss_Mode": loss_mode,
                            "Seed": seed,
                            "Model": model_name,
                            "Validation_AUPRC": validation_auprc,
                            **metrics,
                        }
                        experiment_metrics.append(row)
                        all_metrics.append(row)

                    attach_sentiment_scores(
                        test_records,
                        streams,
                        sentiment,
                        thresholds["TGN"],
                        [pruning],
                        soft_alphas=config.soft_filter_alphas,
                        progress_label=(
                            f"{variant}/{fold.test_year}/{loss_mode}/seed{seed} duyarlılık"
                        ),
                    )
                    strategies = signals_from_records_v3(
                        test_records,
                        pruning,
                        sentiment_threshold,
                        config.soft_filter_alphas,
                        config.top_fraction,
                    )
                    signal_change_diagnostics(
                        strategies,
                        experiment_dir / f"signal_changes_seed_{seed}.csv",
                    )
                    for cost in config.round_trip_costs:
                        for strategy_name, signal_frame in strategies.items():
                            financial_metrics, curve = run_backtest(
                                signal_frame, market, cost
                            )
                            row = {
                                "Data_Variant": variant,
                                "Test_Year": fold.test_year,
                                "Loss_Mode": loss_mode,
                                "Seed": seed,
                                "Cost": cost,
                                "Strategy": strategy_name,
                                **financial_metrics,
                            }
                            experiment_financial.append(row)
                            all_financial.append(row)
                            if ARCHIVE_ALL_SEED_CURVES or seed == config.seeds[0]:
                                curve.to_csv(
                                    experiment_dir
                                    / (
                                        f"curve_seed_{seed}_{strategy_name}_"
                                        f"cost_{cost:.3f}.csv"
                                    ),
                                    index=False,
                                )

                    for record in test_records:
                        prediction = {
                            "Data_Variant": variant,
                            "Test_Year": fold.test_year,
                            "Loss_Mode": loss_mode,
                            "Seed": seed,
                            "Tarih": record["Tarih"],
                            "Hisse_Kodu": record["Hisse_Kodu"],
                            "Label": record["Label"],
                            "Olay_ID": record["Olay_ID"],
                            "TGN_Prob": record["TGN_Prob"],
                            "Text_Prob": record["Text_Prob"],
                            "Graph_Prob": record["Graph_Prob"],
                            "Has_Events": record["Has_Events"],
                        }
                        if ARCHIVE_USER_SCORES:
                            prediction["User_Scores_JSON"] = json.dumps(
                                record["User_Scores"],
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        experiment_predictions.append(prediction)
                        all_predictions.append(prediction)

                    if SAVE_MODELS:
                        _atomic_save_model_checkpoint(
                            experiment_dir / "saved_models" / f"seed_{seed}.pt",
                            models,
                            config,
                            thresholds,
                            selected_parameters,
                            seed,
                            validation_auprc,
                            forum_hash,
                            market_hash,
                            f"{fold.test_year}-01-01",
                            experiment_metadata={
                                "data_variant": variant,
                                "test_year": fold.test_year,
                                "loss_mode": loss_mode,
                                "historical_exploratory": True,
                            },
                        )

                    seed_metric_rows = [
                        row for row in experiment_metrics if int(row["Seed"]) == int(seed)
                    ]
                    seed_prediction_rows = [
                        row for row in experiment_predictions if int(row["Seed"]) == int(seed)
                    ]
                    seed_financial_rows = [
                        row for row in experiment_financial if int(row["Seed"]) == int(seed)
                    ]
                    _write_seed_cache(
                        experiment_dir,
                        seed,
                        seed_metric_rows,
                        seed_prediction_rows,
                        seed_financial_rows,
                        seed_identity,
                    )
                    _write_shareable_progress_bundle(
                        output_dir,
                        all_metrics,
                        all_financial,
                        all_fold_rows,
                        {
                            "stage": "seed_completed",
                            "data_variant": variant,
                            "test_year": fold.test_year,
                            "loss_mode": loss_mode,
                            "seed": int(seed),
                        },
                    )

                macro_rows: List[dict] = []
                leave_one_event_rows: List[pd.DataFrame] = []
                experiment_prediction_frame = pd.DataFrame(experiment_predictions)
                for seed, seed_frame in experiment_prediction_frame.groupby("Seed"):
                    for model_name, probability_column in (
                        ("TGN", "TGN_Prob"),
                        ("Text", "Text_Prob"),
                        ("Graph", "Graph_Prob"),
                    ):
                        macro_rows.append(
                            {
                                "Data_Variant": variant,
                                "Test_Year": fold.test_year,
                                "Loss_Mode": loss_mode,
                                "Seed": seed,
                                "Model": model_name,
                                **generalization_macro_metrics(
                                    seed_frame, probability_column
                                ),
                            }
                        )
                        loo = leave_one_positive_event_out_metrics(
                            seed_frame, probability_column
                        )
                        if not loo.empty:
                            loo.insert(0, "Seed", seed)
                            loo.insert(0, "Loss_Mode", loss_mode)
                            loo.insert(0, "Test_Year", fold.test_year)
                            loo.insert(0, "Data_Variant", variant)
                            leave_one_event_rows.append(loo)
                _atomic_write_dataframe(
                    pd.DataFrame(macro_rows),
                    experiment_dir / "generalization_macro_metrics.csv",
                )
                _atomic_write_dataframe(
                    pd.concat(leave_one_event_rows, ignore_index=True)
                    if leave_one_event_rows
                    else pd.DataFrame(),
                    experiment_dir / "leave_one_positive_event_out.csv",
                )
                _atomic_write_dataframe(
                    pd.DataFrame(experiment_metrics),
                    experiment_dir / "test_metrics_by_seed.csv",
                )
                _atomic_write_dataframe(
                    experiment_prediction_frame,
                    experiment_dir / "test_predictions.csv",
                )
                _atomic_write_dataframe(
                    pd.DataFrame(experiment_financial),
                    experiment_dir / "financial_results_by_seed.csv",
                )
                _atomic_write_json(
                    experiment_dir / "selected_parameters.json",
                    {
                        "pruning_percentile": pruning,
                        "sentiment_threshold": sentiment_threshold,
                        "tgn_thresholds_by_seed_are_in_checkpoints": True,
                    },
                )
                experiment_marker = dict(experiment_identity)
                experiment_marker["completed_at_utc"] = pd.Timestamp.utcnow().isoformat()
                _atomic_write_json(
                    experiment_dir / "EXPERIMENT_COMPLETED.json", experiment_marker
                )
                _write_shareable_progress_bundle(
                    output_dir,
                    all_metrics,
                    all_financial,
                    all_fold_rows,
                    {
                        "stage": "experiment_completed",
                        "data_variant": variant,
                        "test_year": fold.test_year,
                        "loss_mode": loss_mode,
                    },
                )

    fold_frame = pd.DataFrame(all_fold_rows)
    _atomic_write_dataframe(fold_frame, output_dir / "outer_fold_manifest.csv")
    metrics_frame = pd.DataFrame(all_metrics)
    financial_frame = pd.DataFrame(all_financial)
    predictions_frame = pd.DataFrame(all_predictions)
    _atomic_write_dataframe(metrics_frame, output_dir / "all_test_metrics_by_seed.csv")
    _atomic_write_dataframe(financial_frame, output_dir / "all_financial_results_by_seed.csv")
    _atomic_write_dataframe(predictions_frame, output_dir / "all_test_predictions.csv")

    if not metrics_frame.empty:
        metrics_summary = (
            metrics_frame.groupby(
                ["Data_Variant", "Test_Year", "Loss_Mode", "Model"], as_index=False
            )
            .agg(
                AUPRC_Mean=("AUPRC", "mean"),
                AUPRC_SD=("AUPRC", "std"),
                AUROC_Mean=("AUROC", "mean"),
                Brier_Mean=("Brier", "mean"),
                F1_Mean=("F1", "mean"),
            )
        )
        _atomic_write_dataframe(metrics_summary, output_dir / "all_test_metrics_summary.csv")
    if not financial_frame.empty:
        financial_summary = (
            financial_frame.groupby(
                ["Data_Variant", "Test_Year", "Loss_Mode", "Cost", "Strategy"],
                as_index=False,
            )
            .agg(
                Return_Mean=("Return", "mean"),
                Sharpe_Mean=("Sharpe", "mean"),
                MDD_Mean=("MDD", "mean"),
                Excess_Return_Mean=("Excess_Return", "mean"),
                Information_Ratio_Mean=("Information_Ratio", "mean"),
                Trades_Mean=("Trades", "mean"),
                Exposure_Mean=("Exposure", "mean"),
            )
        )
        _atomic_write_dataframe(
            financial_summary, output_dir / "all_financial_results_summary.csv"
        )

    aggregate_macro_rows: List[dict] = []
    aggregate_leave_one_event_rows: List[pd.DataFrame] = []
    bootstrap_rows: List[dict] = []
    if not predictions_frame.empty:
        group_columns = ["Data_Variant", "Test_Year", "Loss_Mode"]
        for group_key, group in predictions_frame.groupby(group_columns):
            group = group.copy()
            group["Olay_ID"] = (
                group["Olay_ID"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.replace(r"\.0$", "", regex=True)
            )
            ensemble = (
                group.groupby(
                    ["Tarih", "Hisse_Kodu", "Label", "Olay_ID"],
                    as_index=False,
                    dropna=False,
                )
                .agg(
                    TGN_Prob=("TGN_Prob", "mean"),
                    Text_Prob=("Text_Prob", "mean"),
                    Graph_Prob=("Graph_Prob", "mean"),
                )
            )
            for model_name, probability_column in (
                ("TGN", "TGN_Prob"),
                ("Text", "Text_Prob"),
                ("Graph", "Graph_Prob"),
            ):
                aggregate_macro_rows.append(
                    {
                        "Data_Variant": group_key[0],
                        "Test_Year": group_key[1],
                        "Loss_Mode": group_key[2],
                        "Model": model_name,
                        "Aggregation": "mean_across_seeds",
                        **generalization_macro_metrics(ensemble, probability_column),
                    }
                )
                loo = leave_one_positive_event_out_metrics(
                    ensemble, probability_column
                )
                if not loo.empty:
                    loo.insert(0, "Aggregation", "mean_across_seeds")
                    loo.insert(0, "Loss_Mode", group_key[2])
                    loo.insert(0, "Test_Year", group_key[1])
                    loo.insert(0, "Data_Variant", group_key[0])
                    aggregate_leave_one_event_rows.append(loo)
            for baseline in ("Text_Prob", "Graph_Prob"):
                result = two_way_cluster_bootstrap_difference(
                    ensemble,
                    "TGN_Prob",
                    baseline,
                    config.bootstrap_repetitions,
                    progress_label=f"V4 bootstrap {group_key} {baseline}",
                )
                bootstrap_rows.append(
                    {
                        "Data_Variant": group_key[0],
                        "Test_Year": group_key[1],
                        "Loss_Mode": group_key[2],
                        "Comparison": f"TGN-{baseline}",
                        **result,
                    }
                )
    _atomic_write_dataframe(
        pd.DataFrame(aggregate_macro_rows),
        output_dir / "all_event_macro_metrics.csv",
    )
    _atomic_write_dataframe(
        pd.concat(aggregate_leave_one_event_rows, ignore_index=True)
        if aggregate_leave_one_event_rows
        else pd.DataFrame(),
        output_dir / "all_leave_one_positive_event_out.csv",
    )
    _atomic_write_dataframe(
        add_holm_adjustment(pd.DataFrame(bootstrap_rows)),
        output_dir / "all_two_way_bootstrap_auprc.csv",
    )
    _atomic_write_text(
        output_dir / "V4_COMPLETED",
        f"Completed in {format_duration(time.perf_counter() - started_at)}\n",
    )
    logging.info(
        "V4 deney paketi tamamlandı. Süre: %s",
        format_duration(time.perf_counter() - started_at),
    )


def run_pipeline(args: argparse.Namespace) -> None:
    """Veri hazırlamadan kilitli teste kadar bütün akademik deney boru hattını yürütür.

    Ana aşamalar şunlardır: güvenlik işaretlerini ve veri sözleşmesini denetleme;
    forum/piyasa hazırlama; BERT/SBERT özellikleri ve sızıntısız olayları kurma;
    rolling-origin model/parametre seçimi; her seed için nihai eğitim, checkpoint,
    validasyon ve kilitli test; duyarlılık filtresi, backtest, bootstrap ve
    yeniden üretilebilirlik çıktıları. Uzman anotasyonu V4 kapsamı dışındadır.

    Doğrulayıcı modda daha önce açılmış test işaretleri veya açık kullanıcı onayı
    yoksa işlem durur. Test başlangıcından sonra hiçbir hiperparametre seçilmez.
    Fonksiyon çok sayıda CSV/JSON/PT/log çıktısı üretir ve tamamlanma işaretini
    yazar; esas dönüş değeri yoktur.
    """
    output_dir = Path(args.output_dir)
    exploratory = bool(args.exploratory_mode or args.exploratory_rerun)
    marker = output_dir / "LOCKED_TEST_COMPLETED"
    started_marker = output_dir / "LOCKED_TEST_STARTED"
    if (marker.exists() or started_marker.exists()) and not exploratory:
        raise RuntimeError(
            "Bu kilitli test daha önce açılmış veya tamamlanmış. Teste tekrar bakmak yasaktır. "
            "Yalnız keşifsel tekrar için --exploratory-rerun kullanılabilir."
        )
    if not exploratory and not args.confirm_unseen_test:
        raise RuntimeError(
            "--confirm-unseen-test zorunludur. Daha önce incelenmiş tarihler doğrulayıcı test olamaz."
        )
    if not args.locked_test_start:
        raise ValueError("--locked-test-start zorunludur.")

    configure_logging(output_dir)
    pipeline_started_at = time.perf_counter()
    logging.info("[AŞAMA 1/6] Veri hazırlama ve deney güvenlik kontrolleri")
    config = ExperimentConfig(
        rolling_folds=args.rolling_folds,
        epochs=args.epochs,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    forum_path = Path(args.forum_csv)
    market_path = Path(args.market_csv)
    forum = prepare_forum(forum_path)
    market = prepare_market(market_path, output_dir)
    write_unknown_user_audit(forum, output_dir)
    company_confounding = write_label_structure_audit(forum, output_dir)
    if company_confounding and not exploratory:
        raise RuntimeError(
            "Doğrulayıcı deney durduruldu: mevcut CSV'de etiket şirket kimliğiyle tamamen "
            "karışmış. Yaptırım şirketlerinin ihlal dışı forum günleri eklenmeden bu veri "
            "doğrulayıcı RQ1 testi için kullanılamaz. Yalnız açıkça keşifsel çalışma için "
            "--exploratory-mode kullanılabilir."
        )
    if exploratory:
        logging.warning(
            "KEŞİFSEL MOD: Bu sonuçlar doğrulayıcı test veya rebuttal kanıtı olarak sunulamaz."
        )
    locked_start = pd.Timestamp(args.locked_test_start).date()
    development_dates = sorted(day for day in forum["Date"].unique() if day < locked_start)
    test_dates = sorted(day for day in forum["Date"].unique() if day >= locked_start)
    if not development_dates or not test_dates:
        raise ValueError("Kilitli test tarihi boş development veya test bölümü oluşturdu.")
    logging.info(
        "Kilitli ayrım: development %s..%s, test %s..%s",
        development_dates[0],
        development_dates[-1],
        test_dates[0],
        test_dates[-1],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Hesaplama cihazı: %s", device)
    logging.info("[AŞAMA 2/6] BERT/SBERT özellik çıkarımı ve zamansal çizge kurulumu")
    extractor = FeatureExtractors(device, config)
    streams = build_streams(forum, extractor, config)
    sentiment = TurkishSentiment(device, config)

    logging.info("[AŞAMA 3/6] Rolling-origin model ve parametre seçimi")
    selected_parameters, rolling_metrics = rolling_origin_selection(
        streams, development_dates, market, sentiment, device, config
    )
    rolling_metrics.to_csv(output_dir / "rolling_validation_metrics.csv", index=False)
    pruning, sentiment_threshold = selected_parameters
    logging.info(
        "Validasyonda kilitlenen downstream parametreleri: pruning=%d, theta=%.2f",
        pruning,
        sentiment_threshold,
    )
    write_manifest(
        output_dir,
        forum_path,
        market_path,
        args.locked_test_start,
        selected_parameters,
        config,
        confirmatory=not exploratory,
    )
    if not exploratory:
        started_marker.write_text(
            "Locked test access started. A rerun cannot be treated as confirmatory.\n",
            encoding="utf-8",
        )

    logging.info("[AŞAMA 4/6] Nihai modellerin eğitimi, kaydı ve kilitli test")
    final_train_dates, final_validation_dates = expanding_folds(
        development_dates, config.rolling_folds
    )[-1]
    seed_metric_rows: List[dict] = []
    prediction_rows: List[dict] = []
    financial_rows: List[dict] = []
    forum_hash = file_sha256(forum_path)
    market_hash = file_sha256(market_path)
    model_dir = output_dir / "saved_models"
    best_validation_auprc = -np.inf
    best_seed: Optional[int] = None
    saved_model_index: List[dict] = []
    final_seed_progress = ProgressReporter(
        "Nihai seed çalışmaları", len(config.seeds), log_interval_seconds=20
    )

    for seed in config.seeds:
        logging.info("Final kilitli test modeli eğitiliyor: seed=%d", seed)
        models, memory = train_models(
            streams, final_train_dates, seed, device, config
        )
        validation_records = evaluate_models(
            models,
            memory,
            streams,
            final_validation_dates,
            device,
            progress_label=f"Seed {seed} nihai validasyon",
        )
        thresholds = model_thresholds(validation_records)
        validation_labels = [record["Label"] for record in validation_records]
        validation_probabilities = [record["TGN_Prob"] for record in validation_records]
        validation_auprc = (
            float(average_precision_score(validation_labels, validation_probabilities))
            if len(set(validation_labels)) == 2
            else float("nan")
        )
        if SAVE_MODELS:
            checkpoint_path = model_dir / f"seed_{seed}.pt"
            save_model_checkpoint(
                checkpoint_path,
                models,
                config,
                thresholds,
                selected_parameters,
                seed,
                validation_auprc,
                forum_hash,
                market_hash,
                args.locked_test_start,
            )
            saved_model_index.append(
                {
                    "seed": int(seed),
                    "checkpoint": checkpoint_path.name,
                    "validation_auprc": validation_auprc,
                    "thresholds": {
                        key: float(value) for key, value in thresholds.items()
                    },
                }
            )
            score_for_selection = (
                validation_auprc if np.isfinite(validation_auprc) else -np.inf
            )
            if score_for_selection > best_validation_auprc:
                best_validation_auprc = score_for_selection
                best_seed = seed
                shutil.copy2(checkpoint_path, model_dir / "best_model.pt")

        test_records = evaluate_models(
            models,
            memory,
            streams,
            test_dates,
            device,
            progress_label=f"Seed {seed} kilitli test",
        )
        labels = [record["Label"] for record in test_records]
        for model_name, probability_key in (
            ("TGN", "TGN_Prob"),
            ("Text", "Text_Prob"),
            ("Graph", "Graph_Prob"),
        ):
            metrics = classification_metrics(
                labels,
                [record[probability_key] for record in test_records],
                thresholds[model_name],
            )
            seed_metric_rows.append({"Seed": seed, "Model": model_name, **metrics})

        attach_sentiment_scores(
            test_records,
            streams,
            sentiment,
            thresholds["TGN"],
            [pruning],
            progress_label=f"Seed {seed} test duyarlılığı",
        )
        raw_signals, filtered_signals, benchmark_signals = signals_from_records(
            test_records, pruning, sentiment_threshold
        )
        for cost in config.round_trip_costs:
            for strategy, signals in (
                ("Raw", raw_signals),
                ("TGN_Filtered", filtered_signals),
                ("Intraday_Always_Long", benchmark_signals),
            ):
                metrics, curve = run_backtest(signals, market, cost)
                financial_rows.append(
                    {"Seed": seed, "Cost": cost, "Strategy": strategy, **metrics}
                )
                if seed == config.seeds[0]:
                    curve.to_csv(
                        output_dir / f"curve_{strategy}_cost_{cost:.3f}.csv", index=False
                    )

        for record in test_records:
            prediction_rows.append(
                {
                    "Seed": seed,
                    "Tarih": record["Tarih"],
                    "Hisse_Kodu": record["Hisse_Kodu"],
                    "Label": record["Label"],
                    "TGN_Prob": record["TGN_Prob"],
                    "Text_Prob": record["Text_Prob"],
                    "Graph_Prob": record["Graph_Prob"],
                }
            )
        final_seed_progress.update(detail=f"seed {seed} tamamlandı")

    final_seed_progress.finish(detail="bütün nihai seed'ler tamamlandı")
    if SAVE_MODELS:
        if best_seed is None and saved_model_index:
            # Tek sınıflı validasyonda AUPRC tanımsız olabilir; kullanılabilir bir
            # varsayılan bırakmak için ilk seed kayıtlı model olarak seçilir.
            best_seed = int(saved_model_index[0]["seed"])
            shutil.copy2(
                model_dir / saved_model_index[0]["checkpoint"],
                model_dir / "best_model.pt",
            )
            logging.warning(
                "Validasyon AUPRC tanımsızdı; best_model.pt için ilk seed (%d) kullanıldı.",
                best_seed,
            )
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "best_seed": best_seed,
                    "best_checkpoint": "best_model.pt" if best_seed is not None else None,
                    "models": saved_model_index,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (model_dir / "KULLANIM_TR.txt").write_text(
            "KAYITLI MODEL KULLANIMI\n"
            "=======================\n"
            "1. v3_punpAnddump.py dosyasının AYARLAR bölümünde RUN_MODEL=False yapın.\n"
            "2. RUN_SAVED_MODEL=True yapın.\n"
            "3. SAVED_MODEL_PATH değerini best_model.pt dosyasına yöneltin.\n"
            "4. INFERENCE_FORUM_CSV değerini yeni forum CSV dosyasına yöneltin.\n"
            "5. Model hücresini çalıştırın. Tahminler INFERENCE_OUTPUT_CSV yoluna yazılır.\n\n"
            "Not: Yeni veride SPK_Etiketi zorunlu değildir. Zamansal kullanıcı belleğinin "
            "oluşması için yalnız son günü değil, uygun bir tarihsel ısınma dönemini de "
            "girdi dosyasına ekleyin. BERT/SBERT modelleri checkpoint içine kopyalanmaz; "
            "kayıtlı model adlarından yerel önbellekten veya Hugging Face üzerinden yüklenir.\n",
            encoding="utf-8",
        )
        if best_seed is not None:
            best_score_text = (
                f"{best_validation_auprc:.6f}"
                if np.isfinite(best_validation_auprc)
                else "tanımsız (ilk seed varsayılanı)"
            )
            logging.info(
                "En iyi kayıtlı model: %s | seed=%d | validasyon AUPRC=%s",
                model_dir / "best_model.pt",
                best_seed,
                best_score_text,
            )

    logging.info("[AŞAMA 5/6] Metrikler, bootstrap ve finansal özetler")
    seed_metrics = pd.DataFrame(seed_metric_rows)
    seed_metrics.to_csv(output_dir / "locked_test_metrics_by_seed.csv", index=False)
    metric_summary = summarize_seed_metrics(seed_metrics)
    metric_summary.to_csv(output_dir / "locked_test_metrics_summary.csv", index=False)
    logging.info("\n%s", metric_summary.to_string(index=False))

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(output_dir / "locked_test_predictions.csv", index=False)
    ensemble = (
        predictions.groupby(["Tarih", "Hisse_Kodu", "Label"], as_index=False)
        .agg(TGN_Prob=("TGN_Prob", "mean"), Text_Prob=("Text_Prob", "mean"), Graph_Prob=("Graph_Prob", "mean"))
    )
    bootstrap_rows = []
    for baseline in ("Text_Prob", "Graph_Prob"):
        result = two_way_cluster_bootstrap_difference(
            ensemble,
            "TGN_Prob",
            baseline,
            config.bootstrap_repetitions,
            progress_label=f"TGN - {baseline} bootstrap",
        )
        bootstrap_rows.append({"Comparison": f"TGN-{baseline}", **result})
    pd.DataFrame(bootstrap_rows).to_csv(
        output_dir / "two_way_cluster_bootstrap_auprc.csv", index=False
    )

    financial = pd.DataFrame(financial_rows)
    financial.to_csv(output_dir / "financial_results_by_seed.csv", index=False)
    financial_summary = (
        financial.groupby(["Cost", "Strategy"])
        .agg(
            Return_Mean=("Return", "mean"),
            Return_SD=("Return", lambda values: values.std(ddof=1)),
            Sharpe_Mean=("Sharpe", "mean"),
            Sharpe_SD=("Sharpe", lambda values: values.std(ddof=1)),
            MDD_Mean=("MDD", "mean"),
            Trades_Mean=("Trades", "mean"),
            Exposure_Mean=("Exposure", "mean"),
            Coverage_Mean=("Coverage", "mean"),
        )
        .reset_index()
    )
    financial_summary.to_csv(output_dir / "financial_results_summary.csv", index=False)

    logging.info("[AŞAMA 6/6] Çıktılar tamamlandı")
    if exploratory:
        (output_dir / "EXPLORATORY_RUN").write_text(
            "This run is exploratory because the test period was previously inspected "
            "and company identity is confounded with the label.\n",
            encoding="utf-8",
        )
        logging.info("Keşifsel çalışma tamamlandı; doğrulayıcı kanıt sayılamaz.")
    else:
        marker.write_text(
            "Locked test completed. Subsequent analyses must be labelled exploratory.\n",
            encoding="utf-8",
        )
        logging.info("Kilitli test tamamlandı. Sonraki tekrarlar doğrulayıcı sayılamaz.")
    logging.info(
        "Toplam çalışma süresi: %s",
        format_duration(time.perf_counter() - pipeline_started_at),
    )


def parse_args() -> argparse.Namespace:
    """Notebook ayarlarını ``run_pipeline`` için argparse nesnesine dönüştürür.

    Dosya komut satırı betiği gibi kullanılabilse de bu sürümde değerler HÜCRE 1
    değişkenlerinden oluşturulur. Epoch, fold ve bootstrap sayıları parser
    varsayılanlarıyla tanımlanır; ``CONFIRM_UNSEEN_TEST`` doğruysa güvenlik bayrağı
    eklenir. Çıktı doğrulanmış ``argparse.Namespace`` nesnesidir.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--forum-csv")
    parser.add_argument("--market-csv")
    parser.add_argument("--output-dir")
    parser.add_argument("--locked-test-start")
    parser.add_argument("--confirm-unseen-test", action="store_true")
    parser.add_argument("--exploratory-rerun", action="store_true")
    parser.add_argument("--exploratory-mode", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rolling-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    values = [
        "--forum-csv", str(FORUM_CSV),
        "--market-csv", str(MARKET_CSV),
        "--output-dir", str(OUTPUT_DIR),
        "--locked-test-start", LOCKED_TEST_START,
    ]
    if CONFIRM_UNSEEN_TEST:
        values.append("--confirm-unseen-test")
    return parser.parse_args(values)


# %% HÜCRE 5 — ÇALIŞTIRMA
# HÜCRE 5/5 — HÜCRE 1--4 çalıştırıldıktan sonra yürütülmelidir.
# -----------------------------------------------------------------------------
# ANA ÇALIŞTIRMA YÖNLENDİRİCİSİ
# -----------------------------------------------------------------------------
# Bu son blok HÜCRE 1'deki iki anahtara göre üç çalışma biçiminden yalnız birini
# seçer:
#
# 1. RUN_MODEL=True:
#    Yeni modeli eğitir, rolling-origin seçim ve kilitli testi yürütür, sonuçları
#    OUTPUT_DIR altında saklar. Doğrulayıcı koşu için kullanıcı onayı zorunludur.
# 2. RUN_SAVED_MODEL=True:
#    Eğitim yapmadan best_model.pt dosyasını yükler ve yeni forum CSV'sinde tahmin
#    üretir. Yeni veride SPK_Etiketi bulunması gerekmez.
# 3. Her ikisi False:
#    Hiçbir ağır model işlemi başlatılmaz; yalnız açıklayıcı durum mesajı yazılır.
#
# İki modun aynı anda açılması çıktıların karışmasını önlemek için yasaktır.
def main() -> None:
    """Seçilen V4 çalışma kipini gerekli tanımlar yüklendikten sonra başlatır."""
    if RUN_MODEL and RUN_SAVED_MODEL:
        raise RuntimeError("RUN_MODEL ve RUN_SAVED_MODEL aynı anda True olamaz.")
    if RUN_MODEL:
        if "run_v4_experiment_suite" not in globals():
            raise RuntimeError(
                "Model fonksiyonları bellekte değil. Önce HÜCRE 4'ü çalıştırın "
                "veya çekirdeği yeniden başlatıp 'Run All' kullanın."
            )
        # 2021--2025 dönemleri daha önce incelendiğinden V4 bunları açıkça
        # keşifsel ileri-zincirli fold'lar olarak çalıştırır.
        run_v4_experiment_suite(parse_args())
        return
    if RUN_SAVED_MODEL:
        if "SavedModelPredictor" not in globals():
            raise RuntimeError(
                "Model sınıfları bellekte değil. Önce HÜCRE 4'ü çalıştırın "
                "veya çekirdeği yeniden başlatıp 'Run All' kullanın."
            )
        configure_logging(OUTPUT_DIR)
        saved_predictor = SavedModelPredictor(SAVED_MODEL_PATH)
        saved_predictions = saved_predictor.predict_csv(
            INFERENCE_FORUM_CSV, INFERENCE_OUTPUT_CSV
        )
        print("Kayıtlı model tahmini tamamlandı:", INFERENCE_OUTPUT_CSV.resolve())
        print(saved_predictions.head())
        return
    print(
        "RUN_MODEL=False ve RUN_SAVED_MODEL=False: model çalıştırılmadı. "
        "V4 deney paketini çalıştırmak için RUN_MODEL=True; kayıtlı model tahmini için "
        "RUN_SAVED_MODEL=True yapın."
    )


if __name__ == "__main__":
    main()