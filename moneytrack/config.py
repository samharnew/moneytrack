class Config:
    CSV_EXT = ".csv"
    EXTERNAL_ACCOUNT_KEY = "EXT"

    class FieldNames:
        # Core Dataset Fields
        ACCOUNT_KEY = "ACCOUNT_KEY"
        DATE = "DATE"
        BALANCE = "BALANCE"
        FROM_ACCOUNT_KEY = "FROM_ACCOUNT_KEY"
        TO_ACCOUNT_KEY = "TO_ACCOUNT_KEY"
        AMOUNT = "AMOUNT"
        DESCRIPTION = "DESCRIPTION"

        # Account Details
        ACCOUNT_NBR = "ACCOUNT_NBR"
        SORT_CODE = "SORT_CODE"
        COMPANY = "COMPANY"
        ACCOUNT_TYP = "ACCOUNT_TYP"
        ISA = "ISA"

        # Daily Summary Fields
        INTEREST = "INTEREST"
        TRANSFER = "TRANSFER"

        # Others
        PREV_BALANCE = "PREV_BALANCE"
        PREV_DATE = "PREV_DATE"
        INTEREST_RATE = "INTEREST_RATE"
        CUM_INTEREST_RATE = "CUM_INTEREST_RATE"
        START_DATE = "START_DATE"
        END_DATE = "END_DATE"
        START_BALANCE = "START_BALANCE"
        END_BALANCE = "END_BALANCE"
