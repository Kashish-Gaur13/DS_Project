import pandas as pd

def calculate_churn(marktanteile: pd.DataFrame) -> pd.DataFrame:
    # Copy relevant columns
    df = marktanteile[["Krankenkasse", "Marktanteil Mitglieder", "Jahr"]].copy()

    # Ensure column is string to strip % if present
    df["Marktanteil Mitglieder"] = df["Marktanteil Mitglieder"].astype(str).str.rstrip("%")

    # Convert cleaned values to float and scale by 1/100
    df["Marktanteil Mitglieder"] = df["Marktanteil Mitglieder"].astype(float) / 100

    # Pivot to have years as columns and Krankenkasse as rows
    pivot = df.pivot(index="Krankenkasse", columns="Jahr", values="Marktanteil Mitglieder")
    pivot = pivot.sort_index(axis=1)

    # Calculate churn as negative year-over-year percentage change
    churn = pivot.pct_change(axis=1) * -1

    # Convert back to long format
    churn = churn.reset_index().melt(id_vars="Krankenkasse", var_name="Jahr", value_name="ChurnRate")

    # Drop NaN churn values (e.g. first year where no previous data to compare)
    churn = churn.dropna(subset=["ChurnRate"])

    return churn

def extract_satisfaction_2023(kundenmonitor2023: pd.DataFrame) -> pd.DataFrame:
    # Rename the first column to "Metric" for clarity
    kundenmonitor2023 = kundenmonitor2023.rename(columns={kundenmonitor2023.columns[0]: "Metric"})

    # Filter only 'Globalzufriedenheit (Schulnote)'
    satisfaction = kundenmonitor2023[kundenmonitor2023["Metric"] == "Globalzufriedenheit (Schulnote)"].drop("Metric", axis=1)

    # Convert from wide to long format
    satisfaction_long = satisfaction.melt(var_name="Krankenkasse", value_name="Satisfaction")

    # Convert to numeric, coerce errors
    satisfaction_long["Satisfaction"] = pd.to_numeric(satisfaction_long["Satisfaction"], errors="coerce")

    # Add year
    satisfaction_long["Jahr"] = 2023

    # Remove rows with missing satisfaction values
    satisfaction_long = satisfaction_long.dropna(subset=["Satisfaction"])

    return satisfaction_long

def extract_satisfaction(kundenmonitor: pd.DataFrame) -> pd.DataFrame:
    kundenmonitor = kundenmonitor.rename(columns={kundenmonitor.columns[0]: "Metric"})

    # Filter only 'Globalzufriedenheit' row
    satisfaction = kundenmonitor[kundenmonitor["Metric"] == "Globalzufriedenheit"].drop("Metric", axis=1)

    # Convert wide format (funds as columns) to long format
    satisfaction_long = satisfaction.melt(var_name="Krankenkasse", value_name="Satisfaction")

    satisfaction_long["Satisfaction"] = pd.to_numeric(satisfaction_long["Satisfaction"], errors="coerce")
    satisfaction_long["Jahr"] = 2025  # Add year column (adjust if needed)

    # Remove NaNs
    satisfaction_long = satisfaction_long.dropna(subset=["Satisfaction"])

    return satisfaction_long

def compute_competitor_satisfaction(df: pd.DataFrame) -> pd.DataFrame:
    def competitor_avg_satisfaction(group):
        result = []
        for kasse in group["Krankenkasse"]:
            others = group[group["Krankenkasse"] != kasse]
            avg = others["Satisfaction"].mean()
            result.append(avg)
        group["CompetitorSatisfactionAvg"] = result
        return group

    df = df.groupby("Jahr").apply(competitor_avg_satisfaction).reset_index(drop=True)
    return df

def create_features(marktanteile: pd.DataFrame, kundenmonitor: pd.DataFrame, kundenmonitor2023: pd.DataFrame) -> pd.DataFrame:
    churn = calculate_churn(marktanteile)

    # Extract satisfaction from both datasets and combine
    satisfaction_2024 = extract_satisfaction(kundenmonitor)
    satisfaction_2023 = extract_satisfaction_2023(kundenmonitor2023)

    satisfaction = pd.concat([satisfaction_2023, satisfaction_2024], ignore_index=True)

    # Merge churn and satisfaction on Krankenkasse and Jahr
    df = churn.merge(satisfaction, on=["Krankenkasse", "Jahr"], how="left")

    # Calculate competitor average satisfaction
    df = compute_competitor_satisfaction(df)

    # Add placeholder for contribution increase (all zeros)
    df["Beitragserhöhung"] = 0

    # Drop rows missing critical data
    df = df.dropna(subset=["ChurnRate", "Satisfaction"])

    return df
