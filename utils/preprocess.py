import pandas as pd

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-zA-Z0-9_]', '', regex=True)
    return df

def load_data():
    survey_full = pd.read_excel("data/230807_Survey.xlsx")
    survey_2023 = pd.read_excel("data/Kundenmonitor_GKV_2023.xlsx")
    survey_2024 = pd.read_excel("data/Kundenmonitor_GKV_2024.xlsx")
    market_share = pd.read_excel("data/Marktanteile je Kasse.xlsx")
    morbidity = pd.read_excel("data/Morbidity_Region.xlsx")
    additional_contrib = pd.read_excel("data/Zusatzbeitrag_je Kasse je Quartal.xlsx")

    return survey_full, survey_2023, survey_2024, market_share, morbidity, additional_contrib

def reshape_survey(df, year):
    df = df.rename(columns={df.columns[0]: "Krankenkasse"})
    df = df.melt(id_vars=["Krankenkasse"], var_name="Question", value_name="Response")
    df["Jahr"] = year
    return df

def preprocess():
    survey_full, survey_2023, survey_2024, market_share, morbidity, additional_contrib = load_data()

    # Clean column names
    market_share = clean_column_names(market_share)
    morbidity = clean_column_names(morbidity)
    additional_contrib = clean_column_names(additional_contrib)

    # Reshape surveys
    survey_2023_long = reshape_survey(survey_2023, 2023)
    survey_2024_long = reshape_survey(survey_2024, 2024)

    # Combine surveys
    combined_survey = pd.concat([survey_2023_long, survey_2024_long], ignore_index=True)
    combined_survey= clean_column_names(combined_survey)

    # Merge additional contrib by average per Kasse per year
    avg_contrib = additional_contrib.groupby(['krankenkasse', 'jahr'])['zusatzbeitrag'].mean().reset_index()
    avg_contrib.rename(columns={'zusatzbeitrag': 'avg_zusatzbeitrag'}, inplace=True)

    # Merge all together
    merged = combined_survey.merge(market_share, on=['krankenkasse', 'jahr'], how='left')
    merged = merged.merge(morbidity, on=['krankenkasse', 'jahr'], how='left')
    merged = merged.merge(avg_contrib, on=['krankenkasse', 'jahr'], how='left')

    return merged

