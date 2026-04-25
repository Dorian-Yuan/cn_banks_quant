import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "ashare")
REPORT_DIR = os.path.join(PROJECT_ROOT, "research")

BANKS = {
    "601398": "工商银行",
    "601288": "农业银行",
    "601988": "中国银行",
    "601939": "建设银行",
    "601328": "交通银行",
    "601658": "邮储银行",
}

SIX_BANK_CODES = list(BANKS.keys())
FIVE_BANK_CODES = ["601398", "601288", "601988", "601939", "601328"]
BANK_INDEX_CODE = "399986"

PSBC_START = "2019-12-10"
DATA_START = "2015-01-05"

CATEGORY_RANGES = {
    "short": (1, 120),
    "medium": (121, 240),
    "medium_long": (241, 480),
    "long": (481, 99999),
}

CATEGORY_LABELS = {
    "short": "短期(0~6个月)",
    "medium": "中期(7~12个月)",
    "medium_long": "中长期(12~24个月)",
    "long": "长期(>24个月)",
}

DEFAULT_N_SAMPLES = 1000
RANDOM_SEED = 42


def get_banks_for_period(start_date):
    if start_date < PSBC_START:
        return FIVE_BANK_CODES
    return SIX_BANK_CODES
