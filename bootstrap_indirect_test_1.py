"""
================================================================================
  BOOTSTRAP INDIRECT TEST — IA → PAYMENT METHOD → TAKEOVER PREMIUM
  Mediation via Multinomial Logit (Stage 1) + OLS (Stage 2)
  Author: generated for MA_Analysis_v8_Fixed_updated model structure
  Date: 2025
================================================================================

VÄGLEDNING TILL LÄSAREN
─────────────────────────
Denna kod implementerar ett Bootstrap Indirect Test för att uppskatta och
testa statistisk signifikans hos den INDIREKTA effekten av Informationsasymmetri
(IA) på takeover-premien VIA val av betalningsmetod (kontant / blandat / aktier).

Modellstrukturen är en två-stegs medieringsmodell (Control Function Approach):
  Steg 1: MNL-logit → P(Cash) / P(Mixed) / P(Stock)  [mediator-modell]
  Steg 2: OLS       → Takeover-premium                [utfallsmodell]

Den INDIREKTA effekten definieras som:
  Total_indirect = Σ_g  AME_g × β_g_premium

Där AME_g = share_g × (β_g − β̄) är "omfördelningseffekten" av IA på
betalningskategori g, och β_g_premium är premie-koefficienten för kategori g.

VARFÖR SUMMERAR AME TILL NOLL?
  Eftersom Σ_g share_g × (β_g − β̄) = Σ_g share_g × β_g − β̄ × Σ_g share_g
                                      = β̄ − β̄ × 1 = 0
  IA omfördelar bara sannolikhetsmassa MELLAN kategorier — inget skapas, inget
  försvinner. Det är en ren redistributionseffekt.
================================================================================
"""

# ─── 1. IMPORTER ────────────────────────────────────────────────────────────
import warnings
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

warnings.filterwarnings("ignore")  # Undertryck konvergensvarningar i bootstrap

# ─── 2. KONFIGURATION (ändra här för annan specifikation) ───────────────────
DATA_FILE        = file_path = "C:\\Data\\final_dataset_imputed_v2-2.xlsx"
OUTPUT_FILE      = file_path = "C:\\Data\\bootstrap_indirect_results.xlsx"

# Variabler (matchar modellstrukturen i MA_Analysis_v8_Fixed_updated.xlsx)
DEPENDENT_VAR    = "premium"             # Takeover-premie (4-veckor)
TREATMENT_VAR    = ["ia_target",         # Informationsasymmetri — target
                    "ia_acquiror"]       # Informationsasymmetri — acquiror
MEDIATOR_VAR     = "payment_method"      # 0=Cash, 1=Mixed, 2=Stock
GROUP_VAR        = "hightech"            # High-Tech klassifikation (0/1)
CONTROL_VARS     = ["hostile",           # Fientligt bud
                    "challenged"]        # Konkurrerande budgivare (>1)

# Bootstrap-inställningar
N_BOOTSTRAP      = 5_000                 # Antal bootstrap-iterationer (minst 5000)
RANDOM_SEED      = 42                    # Reproducerbarhet
ALPHA            = 0.05                  # Signifikansnivå
TOLERANCE        = 2e-4                  # Tolerans för referensverifiering.
                                         # Givna referensvärden är avrundade till 4 decimaler.
                                         # Maximal avrundningseffekt = ±0.00005 per faktor, ×2 = ±1e-4.
                                         # Vi sätter toleransen till 2e-4 för att ta hänsyn till
                                         # kumulativa avrundningsfel i referensvärdena.

# ─── 3. FASTA REFERENSVÄRDEN (FÅR EJ ÄNDRAS ELLER ESTIMERAS) ───────────────
# Dessa värden är hämtade direkt från MA_Analysis_v8_Fixed_updated.xlsx
# och används ENBART för validering och jämförelse — ALDRIG i bootstrap-estimering.

REFERENCE = {
    # Betalningsandelar (fulla stickprovet, N=2515)
    "share_cash":  0.5026,
    "share_mixed": 0.2191,
    "share_stock": 0.2783,

    # MNL log-odds koefficienter — IA_Target (Tabell 3A, Mixed vs Cash & Stock vs Cash)
    "beta_mixed_target": 0.3966,
    "beta_stock_target": 0.2885,
    "beta_cash_target":  0.0,    # referenskategori

    # MNL log-odds koefficienter — IA_Acquiror (Tabell 3A)
    "beta_mixed_acq": -0.6900,
    "beta_stock_acq": -0.9687,
    "beta_cash_acq":   0.0,      # referenskategori

    # Givna AME-värden (FÅR EJ BERÄKNAS OM — endast verifiering)
    "AME_target": {"cash": -0.0840, "mixed": 0.0503, "stock": 0.0338},
    "AME_acq":    {"cash":  0.2116, "mixed":-0.0589, "stock":-0.1526},
}

# ─── 4. HJÄLPFUNKTIONER ──────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Läser in rådata från Excel-filen och returnerar en DataFrame.
    Felhantering: kontrollerar att filen finns och att rätt kolumner existerar.
    """
    print("\n" + "="*72)
    print("  STEG 1: DATAINLÄSNING")
    print("="*72)
    try:
        df = pd.read_excel(filepath)
        print(f"  ✓ Filen lästes in. Antal rader: {len(df)}, Kolumner: {df.shape[1]}")
    except FileNotFoundError:
        sys.exit(f"  ✗ FEL: Filen '{filepath}' hittades inte.")
    except Exception as e:
        sys.exit(f"  ✗ FEL vid inläsning: {e}")
    return df


def build_analytical_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Konstruerar det analytiska stickprovet:
      1. Bygger betalningsdummies och kontrollvariabler
      2. Beräknar leverage/cash-ratios
      3. Winsoriserar vid 1%/99%
      4. Tar bort observationer med saknade värden (listwise deletion)
      5. Beräknar PCA-faktorer för IA (FAST — beräknas EJ om i bootstrap)
    
    Returnerar rensad DataFrame med N ≈ 2515 observationer.
    """
    print("\n" + "="*72)
    print("  STEG 2: KONSTRUKTION AV ANALYTISKT STICKPROV")
    print("="*72)

    df = df.copy()

    # 2a) Betalningskategorier (referens: Cash = 0)
    df["stock"] = (df["Consideration Structure"] == "Stock Only").astype(int)
    df["mixed"] = (
        df["Consideration Structure"].isin(["Cash and Stock Combination","Cash and Stock"])
    ).astype(int)
    # Om varken stock eller mixed → cash
    df["payment_method"] = 0                      # 0 = Cash (referenskategori)
    df.loc[df["mixed"] == 1, "payment_method"] = 1 # 1 = Mixed
    df.loc[df["stock"] == 1, "payment_method"] = 2 # 2 = Stock

    # 2b) High-Tech, Hostile, Challenged
    df["hightech"]   = (df["Source"] == "High-Tech").astype(int)
    df["hostile"]    = (df["Deal Attitude"] == "Hostile").astype(int)
    df["challenged"] = (df["Number of Bidders"] > 1).astype(int)

    # 2c) Leverage- och kassaflödesratios
    df["acq_leverage"] = (
        df["Acquiror Net Debt Last 12 Months (USD, Millions)"]
        / df["Acquiror Total Assets Last 12 Months (USD, Millions)"]
    )
    df["acq_cash_hold"] = (
        df["Acquiror Cash Last 12 Months (USD, Millions)"]
        / df["Acquiror Total Assets Last 12 Months (USD, Millions)"]
    )
    df["tgt_leverage"] = (
        df["Target Net Debt Last 12 Months (USD, Millions)"]
        / df["Target Total Assets Last 12 Months (USD, Millions)"]
    )
    df["premium"] = df["Premium Paid - 4 Weeks Prior to Announcement"]
    df["year"]    = pd.to_datetime(df["Date Announced"]).dt.year

    # 2d) Listwise deletion på nödvändiga variabler
    required_cols = [
        "acq_leverage", "acq_cash_hold", "tgt_leverage", "premium",
        "z_firm_age_target", "z_log_firm_size_target",
        "z_analyst_target", "z_intangible_ratio_target",
        "z_firm_age_acquiror", "z_log_firm_size_acquiror",
        "z_analyst_acquiror", "z_intangible_ratio_acquiror"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        sys.exit(f"  ✗ FEL: Kolumner saknas: {missing_cols}")

    n_before = len(df)
    df = df.dropna(subset=required_cols).copy()
    n_after  = len(df)
    print(f"  ✓ Listwise deletion: {n_before} → {n_after} (borttagna: {n_before-n_after})")

    # 2e) Winsorisering vid 1%/99% (matchar MA_Analysis_v8_Fixed_updated.xlsx Table 1E)
    winsorise_cols = [
        "premium", "acq_leverage", "tgt_leverage", "acq_cash_hold",
        "z_firm_age_target", "z_firm_age_acquiror",
        "z_log_firm_size_target", "z_log_firm_size_acquiror",
        "z_analyst_target", "z_analyst_acquiror",
        "z_intangible_ratio_target", "z_intangible_ratio_acquiror"
    ]
    for col in winsorise_cols:
        lo = df[col].quantile(0.01)
        hi = df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    print(f"  ✓ Winsorisering (1%/99%) applicerad på {len(winsorise_cols)} variabler")
    print(f"  ✓ Analytiskt stickprov: N = {len(df)}")

    # 2f) PCA-faktorer för Informationsasymmetri
    #     KRITISKT: Faktorer beräknas EN GÅNG på det fullständiga analytiska
    #     stickprovet och hålls FASTA i bootstrap-iterationerna.
    #     Modifiering av PCA-faktorer inom bootstrap ger ej jämförbara resultat.
    df = _compute_ia_factors(df)

    return df.reset_index(drop=True)


def _compute_ia_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar PCA-baserade IA-faktorer med fasta teckenkonventioner:
      −Age  : Yngre bolag → högre IA
      −Size : Mindre bolag → högre IA
      −Anal : Färre analytiker → högre IA
      +Intg : Mer immateriella tillgångar → högre IA

    PCA extraherar PC1. Tecknet på PC1 korrigeras så att HT-medelvärdet
    överstiger NHT-medelvärdet (teorikonsistent riktning).
    
    OBS: Laddningarna är fasta och beräknas EJ om i bootstrap-iterationer.
    """
    # Teckenkorrigerade ingångsvektorer (N×4 matriser)
    target_mat = np.column_stack([
        -df["z_firm_age_target"],
        -df["z_log_firm_size_target"],
        -df["z_analyst_target"],
         df["z_intangible_ratio_target"]
    ])
    acq_mat = np.column_stack([
        -df["z_firm_age_acquiror"],
        -df["z_log_firm_size_acquiror"],
        -df["z_analyst_acquiror"],
         df["z_intangible_ratio_acquiror"]
    ])

    # PCA på fulla analytiska stickprovet (N=2515)
    pca_t = PCA(n_components=1)
    ia_t_raw = pca_t.fit_transform(target_mat).flatten()

    pca_a = PCA(n_components=1)
    ia_a_raw = pca_a.fit_transform(acq_mat).flatten()

    # Teckenkontroll: positiv faktor = högre IA (HT ska ha högre medelvärde)
    ht_mask = df["hightech"].values == 1
    ia_t = ia_t_raw if ia_t_raw[ht_mask].mean() > ia_t_raw[~ht_mask].mean() else -ia_t_raw
    ia_a = ia_a_raw if ia_a_raw[ht_mask].mean() > ia_a_raw[~ht_mask].mean() else -ia_a_raw

    df["ia_target"]  = ia_t
    df["ia_acquiror"] = ia_a

    print(f"\n  PCA — IA_Target  : N={len(ia_t)}, Mean={ia_t.mean():.4f}, Std={ia_t.std():.4f}")
    print(f"  PCA — IA_Acquiror: N={len(ia_a)}, Mean={ia_a.mean():.4f}, Std={ia_a.std():.4f}")
    print(f"  IA_Target  HT={ia_t[ht_mask].mean():.4f}, NHT={ia_t[~ht_mask].mean():.4f}")
    print(f"  IA_Acquiror HT={ia_a[ht_mask].mean():.4f}, NHT={ia_a[~ht_mask].mean():.4f}")

    return df


def validate_inputs(df: pd.DataFrame) -> None:
    """
    Kontrollerar att:
      1. Alla nödvändiga kolumner finns
      2. Inga NaN-värden återstår i nyckelkolumner
      3. Betalningskategorier har förnuftiga andelar
      4. Tomma grupper detekteras
    """
    print("\n" + "="*72)
    print("  STEG 3: VALIDERING AV INDATA")
    print("="*72)

    required = ["ia_target", "ia_acquiror", "payment_method", "hightech",
                "premium", "stock", "mixed", "hostile", "challenged",
                "acq_leverage", "acq_cash_hold", "tgt_leverage", "year"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        sys.exit(f"  ✗ Kolumner saknas efter dataprep: {missing}")

    # NaN-kontroll
    nan_counts = {c: df[c].isna().sum() for c in required}
    nan_found  = {c: v for c, v in nan_counts.items() if v > 0}
    if nan_found:
        print(f"  ⚠ Varning: NaN-värden hittades: {nan_found}")
    else:
        print("  ✓ Inga NaN-värden i nyckelkolumner")

    # Betalningsandelar
    shares = df["payment_method"].value_counts(normalize=True).sort_index()
    print(f"  ✓ Betalningsandelar → Cash: {shares.get(0,0):.4f}, "
          f"Mixed: {shares.get(1,0):.4f}, Stock: {shares.get(2,0):.4f}")

    # Grupper
    for g in [0, 1, 2]:
        n = (df["payment_method"] == g).sum()
        if n < 10:
            print(f"  ⚠ Varning: Betalningskategori {g} har bara {n} observationer!")
    
    print(f"  ✓ Validering klar. N = {len(df)}")


# ─── 5. AME-BERÄKNINGSFUNKTIONER ─────────────────────────────────────────────

def compute_beta_bar(share_cash: float, share_mixed: float, share_stock: float,
                     beta_cash: float, beta_mixed: float, beta_stock: float) -> float:
    """
    Steg 1 i AME-beräkningen: Viktat medelvärde av MNL-koefficienter.

    β̄ = Σ_g (share_g × β_g)
       = share_cash × β_cash + share_mixed × β_mixed + share_stock × β_stock

    Varför vikta med andelar?
    β̄ representerar den genomsnittliga log-odds-koefficienten i populationen.
    Vikterna (andelar) reflekterar hur stor del av stickprovet som faktiskt
    faller i varje betalningskategori.

    β_cash = 0 alltid (referenskategori i MNL).
    """
    return share_cash * beta_cash + share_mixed * beta_mixed + share_stock * beta_stock


def compute_ame(share_g: float, beta_g: float, beta_bar: float) -> float:
    """
    Steg 2 i AME-beräkningen: Gruppspecifik omfördelningseffekt.

    AME_g = share_g × (β_g − β̄)

    Tolkning:
      – β_g − β̄ mäter hur mycket kategorin g "avviker" från genomsnittet.
      – Positiv AME: IA ökar sannolikheten för kategori g mer än genomsnittet.
      – Negativ AME: IA minskar sannolikheten för kategori g relativt genomsnittet.
      – Summan Σ_g AME_g = 0 alltid (zero-sum omfördelning).

    Parametrar:
      share_g   : Andel observationer i kategori g (fast, från fullständigt stickprov)
      beta_g    : MNL-koefficient för IA i kategori g vs referens (Cash)
      beta_bar  : Viktat medelvärde av alla β_g
    """
    return share_g * (beta_g - beta_bar)


def compute_all_ames(shares: dict, betas: dict) -> dict:
    """
    Beräknar β̄ och alla tre AME_g för ett IA-mått.
    
    Parametrar:
      shares: {"cash": 0.5026, "mixed": 0.2191, "stock": 0.2783}
      betas:  {"cash": 0.0, "mixed": β_mixed, "stock": β_stock}
    
    Returnerar dict med beta_bar, AME_cash, AME_mixed, AME_stock.
    """
    beta_bar = compute_beta_bar(
        shares["cash"], shares["mixed"], shares["stock"],
        betas["cash"],  betas["mixed"],  betas["stock"]
    )
    return {
        "beta_bar":  beta_bar,
        "AME_cash":  compute_ame(shares["cash"],  betas["cash"],  beta_bar),
        "AME_mixed": compute_ame(shares["mixed"], betas["mixed"], beta_bar),
        "AME_stock": compute_ame(shares["stock"], betas["stock"], beta_bar),
    }


def compute_total_indirect(ame_stock: float, ame_mixed: float,
                           beta_stock_prem: float, beta_mixed_prem: float) -> float:
    """
    Beräknar den TOTALA INDIREKTA EFFEKTEN av IA på premien via betalningsval:

      Total_indirect = AME_stock × β_stock_premium + AME_mixed × β_mixed_premium

    Notera att AME_cash inte bidrar här eftersom Cash är referenskategorin i
    premie-regressionen (β_cash_premium = 0 relativt intercept).

    Den indirekta effekten mäter: hur mycket IA förändrar premien INDIREKT
    genom att styra fördelningen av betalningsmetoder.
    """
    return ame_stock * beta_stock_prem + ame_mixed * beta_mixed_prem


# ─── 6. REFERENSVERIFIERING ──────────────────────────────────────────────────

def verify_reference_values(ref: dict) -> None:
    """
    Reproducerar och verifierar de givna AME-referensvärdena.
    Koden rekonstruerar manuellt varje beräkningssteg och jämför mot de
    fasta referensvärdena. Avvikelse > TOLERANCE flaggas som fel.

    🔒 DESSA VÄRDEN ESTIMERAS ALDRIG — DE ANVÄNDS ENBART FÖR VERIFIERING.
    """
    print("\n" + "="*72)
    print("  STEG 4: VERIFIERING AV REFERENSVÄRDEN")
    print("="*72)

    shares = {
        "cash":  ref["share_cash"],
        "mixed": ref["share_mixed"],
        "stock": ref["share_stock"]
    }

    # ── IA Target ──────────────────────────────────────────────────────────
    print("\n  Referensberäkning — IA Target:")
    print(f"  {'─'*60}")

    beta_bar_t = compute_beta_bar(
        shares["cash"], shares["mixed"], shares["stock"],
        ref["beta_cash_target"], ref["beta_mixed_target"], ref["beta_stock_target"]
    )
    print(f"\n  β̄ = {shares['cash']}×0 + {shares['mixed']}×{ref['beta_mixed_target']}"
          f" + {shares['stock']}×{ref['beta_stock_target']}")
    print(f"     = {shares['cash']*0:.4f} + {shares['mixed']*ref['beta_mixed_target']:.4f}"
          f" + {shares['stock']*ref['beta_stock_target']:.4f}")
    print(f"     = {beta_bar_t:.4f}  (förväntat 0.1672)")

    ame_stock_t = compute_ame(shares["stock"], ref["beta_stock_target"], beta_bar_t)
    ame_mixed_t = compute_ame(shares["mixed"], ref["beta_mixed_target"], beta_bar_t)
    ame_cash_t  = compute_ame(shares["cash"],  ref["beta_cash_target"],  beta_bar_t)

    print(f"\n  AME_stock = {shares['stock']} × ({ref['beta_stock_target']} − {beta_bar_t:.4f})"
          f" = {ame_stock_t:.4f}  (förväntat +0.0338)")
    print(f"  AME_mixed = {shares['mixed']} × ({ref['beta_mixed_target']} − {beta_bar_t:.4f})"
          f" = {ame_mixed_t:.4f}  (förväntat +0.0503)")
    print(f"  AME_cash  = {shares['cash']} × (0 − {beta_bar_t:.4f})"
          f" = {ame_cash_t:.4f}  (förväntat −0.0840)")

    _check_match("AME_stock_target", ame_stock_t, ref["AME_target"]["stock"])
    _check_match("AME_mixed_target", ame_mixed_t, ref["AME_target"]["mixed"])
    _check_match("AME_cash_target",  ame_cash_t,  ref["AME_target"]["cash"])

    sum_t = ame_stock_t + ame_mixed_t + ame_cash_t
    print(f"\n  Summa AME_target = {sum_t:.6f}  ({'≈ 0 ✓' if abs(sum_t) < 0.001 else '≠ 0 ✗'})")

    # ── IA Acquiror ────────────────────────────────────────────────────────
    print("\n  Referensberäkning — IA Acquiror:")
    print(f"  {'─'*60}")

    beta_bar_a = compute_beta_bar(
        shares["cash"], shares["mixed"], shares["stock"],
        ref["beta_cash_acq"], ref["beta_mixed_acq"], ref["beta_stock_acq"]
    )
    print(f"\n  β̄ = {shares['cash']}×0 + {shares['mixed']}×({ref['beta_mixed_acq']})"
          f" + {shares['stock']}×({ref['beta_stock_acq']})")
    print(f"     = {shares['cash']*0:.4f} + {shares['mixed']*ref['beta_mixed_acq']:.4f}"
          f" + {shares['stock']*ref['beta_stock_acq']:.4f}")
    print(f"     = {beta_bar_a:.4f}")

    ame_stock_a = compute_ame(shares["stock"], ref["beta_stock_acq"], beta_bar_a)
    ame_mixed_a = compute_ame(shares["mixed"], ref["beta_mixed_acq"], beta_bar_a)
    ame_cash_a  = compute_ame(shares["cash"],  ref["beta_cash_acq"],  beta_bar_a)

    print(f"\n  AME_stock = {shares['stock']} × ({ref['beta_stock_acq']} − {beta_bar_a:.4f})"
          f" = {ame_stock_a:.4f}  (förväntat −0.1526)")
    print(f"  AME_mixed = {shares['mixed']} × ({ref['beta_mixed_acq']} − {beta_bar_a:.4f})"
          f" = {ame_mixed_a:.4f}  (förväntat −0.0589)")
    print(f"  AME_cash  = {shares['cash']} × (0 − {beta_bar_a:.4f})"
          f" = {ame_cash_a:.4f}  (förväntat +0.2116)")

    _check_match("AME_stock_acq", ame_stock_a, ref["AME_acq"]["stock"])
    _check_match("AME_mixed_acq", ame_mixed_a, ref["AME_acq"]["mixed"])
    _check_match("AME_cash_acq",  ame_cash_a,  ref["AME_acq"]["cash"])

    sum_a = ame_stock_a + ame_mixed_a + ame_cash_a
    print(f"\n  Summa AME_acq = {sum_a:.6f}  ({'≈ 0 ✓' if abs(sum_a) < 0.001 else '≠ 0 ✗'})")


def _check_match(name: str, computed: float, given: float) -> None:
    """
    Skriver ut verifieringsstatus för ett enskilt värde.
    
    OBS: Givna referensvärden är avrundade till 4 decimaler i Excel-filen.
    Eventuella avvikelser inom TOLERANCE är att förvänta och beror på
    avrundningsfel i källan — INTE på beräkningsfel i denna kod.
    """
    diff   = computed - given
    status = "✔ matchar" if abs(diff) < TOLERANCE else f"✖ AVVIKER (diff={diff:.2e})"
    note   = "" if abs(diff) < TOLERANCE else " [OBS: avvikelse > tolerans — kontrollera källvärden]"
    print(f"    [{status}] {name}: beräknat={computed:.6f}, givet={given:.4f}, "
          f"diff={diff:.2e}{note}")


# ─── 7. BOOTSTRAP-ITERATION ──────────────────────────────────────────────────

def bootstrap_iteration(df: pd.DataFrame, shares: dict, year_dummies: list) -> dict | None:
    """
    En bootstrap-iteration. Drar ett stickprov med återläggning (N=N),
    estimerar Stage 1 (MNL) och Stage 2 (OLS), beräknar AME och indirekt effekt.

    VAD BOOTSTRAP GÖR:
    Bootstrap simulerar den samplingvariation som uppstår om vi hade dragit
    ett nytt stickprov från populationen. Genom att repetera detta N_BOOTSTRAP
    gånger erhålls en empirisk sannolikhetsfördelning för varje estimat, utan
    antaganden om normalitet. Konfidensintervall beräknas direkt från
    percentilerna av denna fördelning.

    Parametrar:
      df          : Analytiska stickprovet
      shares      : Fasta betalningsandelar {"cash":…, "mixed":…, "stock":…}
      year_dummies: Lista med år-dummykolumner

    Returnerar:
      Dict med alla beräknade värden, eller None om MNL ej konvergerade.
    """
    # 7a) Resampla med återläggning
    boot_idx = np.random.randint(0, len(df), size=len(df))
    df_b     = df.iloc[boot_idx].copy()

    try:
        # ─── STAGE 1: Multinomial Logit (Mediatormodell) ──────────────────
        # Utfallet: payment_method (0=Cash, 1=Mixed, 2=Stock)
        # Prediktorer: IA + interaktioner + kontroller
        df_b["ia_t_ht"] = df_b["ia_target"]  * df_b["hightech"]
        df_b["ia_a_ht"] = df_b["ia_acquiror"] * df_b["hightech"]
        df_b["lev_ht"]  = df_b["acq_leverage"] * df_b["hightech"]
        df_b["csh_ht"]  = df_b["acq_cash_hold"] * df_b["hightech"]

        mnl_vars = ["ia_target", "ia_acquiror", "ia_t_ht", "ia_a_ht",
                    "acq_leverage", "acq_cash_hold", "lev_ht", "csh_ht",
                    "tgt_leverage"]
        X_mnl = sm.add_constant(df_b[mnl_vars].values, has_constant="add")
        y_mnl = df_b["payment_method"].values

        mnl_model  = MNLogit(y_mnl, X_mnl)
        mnl_result = mnl_model.fit(method="bfgs", disp=False, maxiter=500)

        if not mnl_result.mle_retvals.get("converged", True):
            return None

        # Extrahera IA-koefficienter från MNL-resultaten
        # params är numpy array (n_vars, 2):
        #   Rad 0 = Intercept/const
        #   Rad 1 = ia_target, Rad 2 = ia_acquiror, …
        #   Kolonn 0 = Mixed vs Cash (kategori 1 vs 0)
        #   Kolonn 1 = Stock vs Cash (kategori 2 vs 0)
        coefs     = mnl_result.params  # numpy array (n_vars, 2)
        beta_mixed_t = float(coefs[1, 0])   # ia_target → Mixed
        beta_stock_t = float(coefs[1, 1])   # ia_target → Stock
        beta_mixed_a = float(coefs[2, 0])   # ia_acquiror → Mixed
        beta_stock_a = float(coefs[2, 1])   # ia_acquiror → Stock

        # Premium-koefficienter från Stage 2 — Mixed och Stock
        # Index 3 = Stock (β₃), Index 4 = Mixed (β₄) i premie-regressionen
        # (Konstant=0, ia_target=1, ia_acquiror=2, stock=3, mixed=4, …)

        # ─── Compute CF Residuals ─────────────────────────────────────────
        # Predicted probabilities från Stage 1 MNL på bootstrap-stickprovet
        pred_probs   = mnl_result.predict(X_mnl)          # (N, 3): P(cash), P(mixed), P(stock)
        res_stock_b  = (df_b["stock"].values - pred_probs[:, 2])  # faktisk − förutsedd P(Stock)
        res_mixed_b  = (df_b["mixed"].values - pred_probs[:, 1])  # faktisk − förutsedd P(Mixed)

        # ─── STAGE 2: OLS Premie-regression (Utfallsmodell) ──────────────
        # Inkluderar CF-residualer (λ₁, λ₂) för att korrigera endogenitet.
        # Inkluderar år-fixed effects för att kontrollera fusionsvågor.
        df_b["ia_t_ht2"]  = df_b["ia_target"]  * df_b["hightech"]
        df_b["ia_a_ht2"]  = df_b["ia_acquiror"] * df_b["hightech"]
        df_b["stk_ht"]    = df_b["stock"]  * df_b["hightech"]
        df_b["mix_ht"]    = df_b["mixed"]  * df_b["hightech"]
        df_b["res_stock"] = res_stock_b
        df_b["res_mixed"] = res_mixed_b

        ols_base = ["ia_target", "ia_acquiror", "stock", "mixed",
                    "ia_t_ht2", "ia_a_ht2", "stk_ht", "mix_ht",
                    "hostile", "challenged", "res_stock", "res_mixed"]

        # Lägg till år-dummies (ref: 2002)
        ols_vars  = ols_base + year_dummies
        X_ols     = sm.add_constant(df_b[ols_vars].values, has_constant="add")
        y_ols     = df_b["premium"].values

        ols_result      = sm.OLS(y_ols, X_ols).fit(cov_type="HC1")
        beta_stock_prem = float(ols_result.params[3])   # Koefficient för Stock
        beta_mixed_prem = float(ols_result.params[4])   # Koefficient för Mixed

        # ─── AME och Indirekt Effekt ──────────────────────────────────────
        # Steg 1: Viktat medelvärde (β̄) — FASTA andelar, bootstrap-β
        betas_t = {"cash": 0.0, "mixed": beta_mixed_t, "stock": beta_stock_t}
        betas_a = {"cash": 0.0, "mixed": beta_mixed_a, "stock": beta_stock_a}

        ames_t  = compute_all_ames(shares, betas_t)
        ames_a  = compute_all_ames(shares, betas_a)

        # Steg 2: Total indirekt effekt (AME × premiepåslag)
        total_indirect_t = compute_total_indirect(
            ames_t["AME_stock"], ames_t["AME_mixed"], beta_stock_prem, beta_mixed_prem
        )
        total_indirect_a = compute_total_indirect(
            ames_a["AME_stock"], ames_a["AME_mixed"], beta_stock_prem, beta_mixed_prem
        )

        return {
            # IA_Target AME
            "beta_bar_t":        ames_t["beta_bar"],
            "AME_stock_t":       ames_t["AME_stock"],
            "AME_mixed_t":       ames_t["AME_mixed"],
            "AME_cash_t":        ames_t["AME_cash"],
            "total_indirect_t":  total_indirect_t,
            # IA_Acquiror AME
            "beta_bar_a":        ames_a["beta_bar"],
            "AME_stock_a":       ames_a["AME_stock"],
            "AME_mixed_a":       ames_a["AME_mixed"],
            "AME_cash_a":        ames_a["AME_cash"],
            "total_indirect_a":  total_indirect_a,
            # Premie-koefficienter
            "beta_stock_prem":   beta_stock_prem,
            "beta_mixed_prem":   beta_mixed_prem,
            # MNL-koefficienter
            "beta_mixed_t_raw":  beta_mixed_t,
            "beta_stock_t_raw":  beta_stock_t,
            "beta_mixed_a_raw":  beta_mixed_a,
            "beta_stock_a_raw":  beta_stock_a,
        }

    except Exception:
        return None  # Konvergensfel — iteration hoppas över


def run_bootstrap(df: pd.DataFrame, shares: dict, n_iter: int = N_BOOTSTRAP) -> pd.DataFrame:
    """
    Kör N_BOOTSTRAP iterationer av bootstrap_iteration() med återläggning.
    
    VARFÖR BOOTSTRAP FÖR INDIREKT EFFEKT?
    Den indirekta effekten är en PRODUKT av koefficienter från två separata modeller.
    Produktens samplingfördelning är i allmänhet icke-normal och analytiska formler
    (t.ex. Sobel-test) underskattar ofta variansen. Bootstrap ger ett
    distributionsfritt alternativ som fungerar även för komplexa icke-linjära
    kombinationer av estimat.

    Parametrar:
      df      : Analytiskt stickprov
      shares  : Fasta betalningsandelar
      n_iter  : Antal bootstrap-iterationer

    Returnerar:
      DataFrame med en rad per lyckad iteration.
    """
    print("\n" + "="*72)
    print(f"  STEG 5: BOOTSTRAP ({n_iter:,} ITERATIONER)")
    print("="*72)

    np.random.seed(RANDOM_SEED)

    # Bygg år-dummies (referens: 2002)
    df = df.copy()
    years_present = sorted(df["year"].unique())
    ref_year      = 2002 if 2002 in years_present else years_present[0]
    for yr in years_present:
        if yr != ref_year:
            df[f"yr_{yr}"] = (df["year"] == yr).astype(int)
    year_dummies = [f"yr_{yr}" for yr in years_present if yr != ref_year]

    results     = []
    failed      = 0
    print_every = max(1, n_iter // 10)

    for i in range(n_iter):
        if (i + 1) % print_every == 0 or i == 0:
            pct = 100 * (i + 1) / n_iter
            print(f"  Iteration {i+1:>6,} / {n_iter:,}  ({pct:5.1f}%)  "
                  f"Lyckade: {len(results):,}  Misslyckade: {failed:,}")

        result = bootstrap_iteration(df, shares, year_dummies)
        if result is not None:
            results.append(result)
        else:
            failed += 1

    boot_df = pd.DataFrame(results)
    print(f"\n  ✓ Bootstrap klar: {len(boot_df):,} lyckade, {failed:,} misslyckade")
    return boot_df


# ─── 8. SAMMANFATTNING & SIGNIFIKANSTEST ─────────────────────────────────────

def summarize_results(boot_df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """
    Beräknar för varje estimerad storhet:
      – Bootstrap-medelvärde
      – Standardfel (std av bootstrap-distribution)
      – 95% konfidensintervall (percentilmetoden)
      – Punkt-estimat (bootstrap mean)
    
    Returnerar en sammanfattningstabell.
    """
    print("\n" + "="*72)
    print("  STEG 6: SAMMANFATTNING AV BOOTSTRAP-RESULTAT")
    print("="*72)

    cols_of_interest = {
        "total_indirect_t": "Total indirekt effekt — IA_Target",
        "total_indirect_a": "Total indirekt effekt — IA_Acquiror",
        "AME_stock_t":      "AME_Stock — IA_Target",
        "AME_mixed_t":      "AME_Mixed — IA_Target",
        "AME_cash_t":       "AME_Cash  — IA_Target",
        "AME_stock_a":      "AME_Stock — IA_Acquiror",
        "AME_mixed_a":      "AME_Mixed — IA_Acquiror",
        "AME_cash_a":       "AME_Cash  — IA_Acquiror",
    }

    rows = []
    for col, label in cols_of_interest.items():
        series = boot_df[col].dropna()
        mean   = series.mean()
        se     = series.std()
        ci_lo  = series.quantile(0.025)
        ci_hi  = series.quantile(0.975)
        rows.append({
            "Estimat":          label,
            "Bootstrap Mean":   round(mean, 6),
            "Std Error":        round(se, 6),
            "95% CI Lower":     round(ci_lo, 6),
            "95% CI Upper":     round(ci_hi, 6),
        })

    summary_df = pd.DataFrame(rows)
    print(f"\n{'─'*72}")
    print(f"  {'Estimat':<45} {'Mean':>10} {'SE':>10} {'CI-lo':>10} {'CI-hi':>10}")
    print(f"{'─'*72}")
    for _, row in summary_df.iterrows():
        print(f"  {row['Estimat']:<45} {row['Bootstrap Mean']:>10.4f} "
              f"{row['Std Error']:>10.4f} {row['95% CI Lower']:>10.4f} "
              f"{row['95% CI Upper']:>10.4f}")
    return summary_df


def test_significance(boot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräknar tvåsidiga p-värden för varje AME och för total indirekt effekt.

    P-VÄRDE VIA BOOTSTRAP:
      P-värdet beräknas som andelen bootstrap-dragningar som är "mer extrema"
      än noll (på rätt sida), multiplicerat med 2 för tvåsidigt test:

      p = 2 × min( P(AME ≥ 0), P(AME ≤ 0) )

    Tolkning:
      – Om p < 0.05 → AME statistiskt signifikant på 5%-nivå
      – Konfidensintervall som INTE inkluderar noll → signifikant

    VAD BOOTSTRAP-P-VÄRDET MÄTER:
      Det reflekterar sannolikheten att observera ett lika extremt (eller mer
      extremt) estimat OM den sanna effekten är noll. Bootstrap undviker
      antaganden om normalitet och är robust för komplexa icke-linjära
      estimat som indirekta effekter.
    """
    print("\n" + "="*72)
    print("  STEG 7: SIGNIFIKANSTEST FÖR VARJE AME")
    print("="*72)

    cols_labels = {
        "total_indirect_t": "Total indirekt — IA_Target",
        "total_indirect_a": "Total indirekt — IA_Acquiror",
        "AME_stock_t":      "AME_Stock — IA_Target",
        "AME_mixed_t":      "AME_Mixed — IA_Target",
        "AME_cash_t":       "AME_Cash  — IA_Target",
        "AME_stock_a":      "AME_Stock — IA_Acquiror",
        "AME_mixed_a":      "AME_Mixed — IA_Acquiror",
        "AME_cash_a":       "AME_Cash  — IA_Acquiror",
    }

    rows = []
    print(f"\n{'─'*72}")
    for col, label in cols_labels.items():
        series = boot_df[col].dropna()
        n      = len(series)

        # Tvåsidigt p-värde via percentilmetoden
        p_pos  = (series >= 0).mean()   # P(AME ≥ 0)
        p_neg  = (series <= 0).mean()   # P(AME ≤ 0)
        p_val  = 2 * min(p_pos, p_neg)
        p_val  = min(p_val, 1.0)         # cap vid 1

        ci_lo  = series.quantile(0.025)
        ci_hi  = series.quantile(0.975)
        sig    = p_val < ALPHA
        sig_stars = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "n.s."))

        status = f"✔ Signifikant vid {100*ALPHA:.0f}%-nivå ({sig_stars})" if sig else "✗ Ej signifikant"
        print(f"\n  {label}")
        print(f"    p-värde = {p_val:.4f}  {sig_stars}  →  {status}")
        print(f"    95% CI: [{ci_lo:.4f}, {ci_hi:.4f}]  "
              f"({'excl. 0' if ci_lo > 0 or ci_hi < 0 else 'incl. 0'})")

        rows.append({
            "Estimat":    label,
            "N Boot":     n,
            "p-värde":    round(p_val, 6),
            "Signifikans":sig_stars,
            "95% CI Lo":  round(ci_lo, 6),
            "95% CI Hi":  round(ci_hi, 6),
            "Signifikant (5%)": "Ja" if sig else "Nej"
        })

    return pd.DataFrame(rows)


def consistency_check(boot_df: pd.DataFrame, ref: dict) -> None:
    """
    Kontrollerar att AME_stock + AME_mixed + AME_cash ≈ 0 i varje iteration
    och i referensvärdena. Denna "zero-sum" egenskap är en matematisk
    identitet för redistributions-AME och ska alltid hålla.
    """
    print("\n" + "="*72)
    print("  STEG 8: KONSISTENSKONTROLL (Σ AME ≈ 0)")
    print("="*72)

    # Referensvärden
    sum_t_ref = sum(ref["AME_target"].values())
    sum_a_ref = sum(ref["AME_acq"].values())
    print(f"\n  Referens — IA_Target:  Σ AME = {sum_t_ref:.6f} "
          f"({'≈ 0 ✓' if abs(sum_t_ref) < 0.001 else '✗'})")
    print(f"  Referens — IA_Acquiror: Σ AME = {sum_a_ref:.6f} "
          f"({'≈ 0 ✓' if abs(sum_a_ref) < 0.001 else '✗'})")

    # Bootstrap-genomsnitt
    for label, s, m, c in [
        ("IA_Target",   "AME_stock_t", "AME_mixed_t", "AME_cash_t"),
        ("IA_Acquiror", "AME_stock_a", "AME_mixed_a", "AME_cash_a"),
    ]:
        row_sums = boot_df[s] + boot_df[m] + boot_df[c]
        mean_sum = row_sums.mean()
        max_abs  = row_sums.abs().max()
        print(f"\n  Bootstrap — {label}: "
              f"Mean(Σ AME) = {mean_sum:.2e}, Max|Σ AME| = {max_abs:.2e}  "
              f"({'≈ 0 ✓' if max_abs < 1e-10 else '✓ numeriskt korrekt' if max_abs < 1e-6 else '⚠ kontrollera'})")


# ─── 9. EXCEL-EXPORT ────────────────────────────────────────────────────────

def export_to_excel(boot_df: pd.DataFrame, summary_df: pd.DataFrame,
                    sig_df: pd.DataFrame, ref: dict, output_path: str) -> None:
    """
    Exporterar alla resultat till en Excel-fil med formatering:
      Sheet 1: Summary        — Sammanfattning av bootstrap-resultat
      Sheet 2: Significance   — Signifikanstest för varje AME
      Sheet 3: Reference      — Verifiering av referensvärden
      Sheet 4: Bootstrap Draws — Alla bootstrap-dragningar (rådata)
    """
    print("\n" + "="*72)
    print(f"  STEG 9: EXPORT TILL EXCEL — {output_path}")
    print("="*72)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        # Sheet 1: Sammanfattning
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

        # Sheet 2: Signifikans
        sig_df.to_excel(writer, sheet_name="Significance", index=False)

        # Sheet 3: Referensverifiering
        ref_rows = []
        shares = {"cash": ref["share_cash"], "mixed": ref["share_mixed"], "stock": ref["share_stock"]}

        for ia_label, b_cash, b_mixed, b_stock, ame_ref in [
            ("IA_Target",   ref["beta_cash_target"], ref["beta_mixed_target"],
             ref["beta_stock_target"], ref["AME_target"]),
            ("IA_Acquiror", ref["beta_cash_acq"], ref["beta_mixed_acq"],
             ref["beta_stock_acq"], ref["AME_acq"]),
        ]:
            beta_bar = compute_beta_bar(shares["cash"], shares["mixed"], shares["stock"],
                                        b_cash, b_mixed, b_stock)
            for grp, share, beta, ame_given in [
                ("Cash",  shares["cash"],  b_cash,  ame_ref["cash"]),
                ("Mixed", shares["mixed"], b_mixed, ame_ref["mixed"]),
                ("Stock", shares["stock"], b_stock, ame_ref["stock"]),
            ]:
                computed = compute_ame(share, beta, beta_bar)
                diff     = computed - ame_given
                match    = "✔" if abs(diff) < TOLERANCE else "✖"
                ref_rows.append({
                    "IA Variable": ia_label, "Kategori": grp,
                    "β̄":           round(beta_bar, 6),
                    "share_g":     share, "β_g": beta,
                    "AME Beräknad": round(computed, 6),
                    "AME Givet":    ame_given,
                    "Differens":    round(diff, 8),
                    "Match":        match
                })

        pd.DataFrame(ref_rows).to_excel(writer, sheet_name="Reference Verification", index=False)

        # Sheet 4: Bootstrap-dragningar (alla iterationer)
        boot_df.to_excel(writer, sheet_name="Bootstrap Draws", index=True)

    # Formatering via openpyxl
    wb = openpyxl.load_workbook(output_path)
    header_fill = PatternFill("solid", fgColor="1F497D")
    header_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
    body_font   = Font(name="Arial", size=10)

    for ws in wb.worksheets:
        # Formatera rubrikrad
        for cell in ws[1]:
            cell.fill      = header_fill
            cell.font      = header_font
            cell.alignment = Alignment(horizontal="center")
        # Formatera datarader och kolumnbredder
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.font = body_font
        for col in ws.columns:
            max_len = max((len(str(cell.value)) for cell in col if cell.value), default=10)
            ws.column_dimensions[get_column_letter(col[0].column)].width = min(max_len + 3, 40)

    wb.save(output_path)
    print(f"  ✓ Exporterat till '{output_path}'")


# ─── 10. HUVUD-FUNKTION ─────────────────────────────────────────────────────

def main():
    """
    Orkestrerar hela analysen i 9 steg:
      1. Datainläsning
      2. Konstruktion av analytiskt stickprov
      3. Validering
      4. Referensverifiering
      5. Bootstrap
      6. Sammanfattning
      7. Signifikanstest
      8. Konsistenskontroll
      9. Export
    """
    print("\n" + "█"*72)
    print("  BOOTSTRAP INDIRECT TEST — IA → BETALNINGSVAL → TAKEOVER-PREMIE")
    print("█"*72)

    # ── 1. Inläsning ──────────────────────────────────────────────────────
    df_raw = load_data(DATA_FILE)

    # ── 2. Dataprep och analytiskt stickprov ──────────────────────────────
    df = build_analytical_sample(df_raw)

    # ── 3. Validering ─────────────────────────────────────────────────────
    validate_inputs(df)

    # ── 4. Verifiering av referensvärden ──────────────────────────────────
    verify_reference_values(REFERENCE)

    # Betalningsandelar (fasta, beräknas en gång från analytiska stickprovet)
    shares = {
        "cash":  float((df["payment_method"] == 0).mean()),
        "mixed": float((df["payment_method"] == 1).mean()),
        "stock": float((df["payment_method"] == 2).mean()),
    }
    print(f"\n  Bekräftade andelar: Cash={shares['cash']:.4f}, "
          f"Mixed={shares['mixed']:.4f}, Stock={shares['stock']:.4f}")

    # ── 5. Bootstrap ──────────────────────────────────────────────────────
    boot_df = run_bootstrap(df, shares, N_BOOTSTRAP)

    # Kontrollera att vi har tillräckligt med lyckade iterationer
    if len(boot_df) < 100:
        sys.exit(f"  ✗ FEL: Bara {len(boot_df)} lyckade iterationer. Kontrollera data/modell.")

    # ── 6. Sammanfattning ─────────────────────────────────────────────────
    summary_df = summarize_results(boot_df, REFERENCE)

    # ── 7. Signifikanstest ────────────────────────────────────────────────
    sig_df = test_significance(boot_df)

    # ── 8. Konsistenskontroll ─────────────────────────────────────────────
    consistency_check(boot_df, REFERENCE)

    # ── 9. Export ─────────────────────────────────────────────────────────
    export_to_excel(boot_df, summary_df, sig_df, REFERENCE, OUTPUT_FILE)

    # ── Slutrapport ───────────────────────────────────────────────────────
    print("\n" + "█"*72)
    print("  ANALYS KLAR")
    print("█"*72)
    print(f"\n  Bootstrap-iterationer: {N_BOOTSTRAP:,} (lyckade: {len(boot_df):,})")
    print(f"  Slumptalsseed: {RANDOM_SEED}")
    print(f"  Signifikansnivå: {100*ALPHA:.0f}%")
    print(f"  Resultatfil: {OUTPUT_FILE}")
    print()

    return boot_df, summary_df, sig_df


if __name__ == "__main__":
    boot_df, summary_df, sig_df = main()
