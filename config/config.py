column_mapping = {
    'fact_expense_tb': {
        'date': 'transaction_date',
        'amount': 'transaction_amount',
        'description': 'transaction_description',
        'owner': 'transaction_owner',
        'filename': 'filename'
    }
}

currency_map = {
'EURO': 'EUR',
'STERLING': 'GBP',
'US_DOLLAR': 'USD',
}

mortgage_monthly_instalment = 1347.45

ASSET_LABELS = {
"acc_bk_sm_5081": "Bankinter Sara",
"acc_bk_co_4264": "Bankinter Common",
"acc_bk_cc_4257": "Bankinter Carlos",
"acc_lloyds": "Lloyds Carlos",
"indexa_cc": "Indexa Investments (Carlos)",
"indexa_sm": "Indexa Investments (Sara)",
"allianz_inv_cc": "Allianz Investments (Carlos)",
"allianz_inv_sm": "Allianz Investments (Sara)",
"fidelity_ap": "Fidelity Pension",
"indexa_pension_cc": "Indexa Pension (Carlos)",
"allianz_pension_sm": "Allianz Pension (Sara",
"mutua_pension": "Mutua Pension",
"generali_pension": "Generali Pension",
"azcona_43_house": "Azcona 43 (House Value)",
"mini_car": "Mini Cooper S",
"ducati_bike": "Ducati Monster",
"watch_collection": "Watch Collection",
"azcona_43_parking": "Azcona 43 (Parking)",
"mortgage_azcona_43": "Azcona 43 (Mortgage)",
"azcona_43_house_owned": "Azcona 43 (Net Equity)"
}

ACTIVE_DB_ENV = "dev"
