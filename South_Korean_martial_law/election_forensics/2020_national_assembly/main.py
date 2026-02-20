import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sqlite3
import os
import re
from scipy.stats import chisquare, pearsonr

# ==========================================
# 1. DEMOGRAPHIC CENSUS EXTRACTOR
# ==========================================
def fetch_kosis_demographics_csv(csv_path="data.csv"):
    print(f"\n--- [1/3] Loading Demographic Census Data ({csv_path}) ---")
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)

        propensity_cols = [f"2020년04월_계_{age}세" for age in range(40, 60)]
        voting_age_cols = [f"2020년04월_계_{age}세" for age in range(18, 100)] + ['2020년04월_계_100세 이상']

        for col in propensity_cols + voting_age_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

        df['target_pop'] = df[propensity_cols].sum(axis=1)
        df['total_voting_pop'] = df[voting_age_cols].sum(axis=1)
        df = df[df['total_voting_pop'] > 0].copy() 
        df['demographic_propensity'] = df['target_pop'] / df['total_voting_pop']
        
        def extract_si_gun_gu_dong(name):
            if not isinstance(name, str): return "", ""
            name = re.sub(r'\(.*\)', '', name).strip()
            parts = name.split()
            if len(parts) >= 3:
                si_gun_gu, dong = parts[-2], parts[-1]
            elif len(parts) == 2:
                si_gun_gu, dong = parts[-2], parts[-1]
            else:
                si_gun_gu, dong = "", name
            si_gun_gu = re.sub(r'(시|군|구)$', '', si_gun_gu).strip()
            return si_gun_gu, dong

        df['base_si_gun_gu'], df['dong_name'] = zip(*df['행정구역'].apply(extract_si_gun_gu_dong))
        return df[['base_si_gun_gu', 'dong_name', '행정구역', 'demographic_propensity']]
    except Exception as e:
        print(f"[!] Error processing CSV: {e}")
        return pd.DataFrame()

# ==========================================
# 2. ELECTION DATABASE EXTRACTORS
# ==========================================
def fetch_election_data():
    """Extracts BOTH Dong-level (area3) and Station-level (area4) data."""
    db_file = "korea_election_regional_21_kor.sqlite"
    print(f"--- [2/3] Extracting Election Data from SQLite ---")
    
    conn = sqlite3.connect(db_file)
    area2 = pd.read_sql_query("SELECT * FROM area2", conn)
    area3 = pd.read_sql_query("SELECT * FROM area3", conn)
    area4 = pd.read_sql_query("SELECT * FROM area4", conn)
    party = pd.read_sql_query("SELECT * FROM party", conn)
    candidate = pd.read_sql_query("SELECT * FROM candidate", conn)
    vote = pd.read_sql_query("SELECT * FROM vote", conn)
    
    # Map candidates to parties
    cand_to_party = dict(zip(candidate['uid'], candidate['party']))
    vote['party'] = vote['candidate'].map(cand_to_party).map(dict(zip(party['uid'], party['name'])))
    vote['is_dem'] = vote['party'].str.contains('민주당', case=False, na=False)
    vote['is_con'] = vote['party'].str.contains('통합당|한국당', case=False, na=False)

    # ---------- A. STATION LEVEL (For Metro 63:36 & Invalid Vote Tests) ----------
    area4['is_early'] = area4['name'].str.contains('prevote', case=False, na=False)
    a3_to_a2 = dict(zip(area3['uid'], area3['area2']))
    area4['area2_uid'] = area4['area3'].map(a3_to_a2)
    area4['region'] = area4['area2_uid'].map(dict(zip(area2['uid'], area2['name'])))
    
    def categorize_metro(name):
        if not isinstance(name, str): return "Other"
        if name.startswith('종로') or name.startswith('강남') or '구' in name:
            if '인천' in name or '계양' in name or '연수' in name: return "Incheon"
            elif '수원' in name or '고양' in name or '성남' in name or '용인' in name: return "Gyeonggi"
            else: return "Seoul"
        return "Other"
        
    area4['metro_zone'] = area4['region'].apply(categorize_metro)
    area4['invalid_rate'] = area4['sum_invalid'] / area4['sum_vote'].replace(0, np.nan)
    
    dem_votes_a4 = vote[vote['is_dem']].groupby('area')['vote'].sum()
    con_votes_a4 = vote[vote['is_con']].groupby('area')['vote'].sum()
    area4['dem_votes'] = area4['uid'].map(dem_votes_a4).fillna(0)
    area4['con_votes'] = area4['uid'].map(con_votes_a4).fillna(0)
    
    # ---------- B. DONG LEVEL (For Benford, Variance, and Gap Tests) ----------
    area3_early = area3[area3['name'].str.contains('prevote|ship|disabled|abroad', case=False, na=False)]['uid']
    area4_early = area4[area4['name'].str.contains('prevote', case=False, na=False)]['uid']
    
    vote_dong = vote.copy()
    vote_dong['vote_type'] = 'same_day'
    vote_dong.loc[vote_dong['area'].isin(area3_early) | vote_dong['area'].isin(area4_early), 'vote_type'] = 'early'
    
    a4_to_a3 = dict(zip(area4['uid'], area4['area3']))
    vote_dong['area3_uid'] = vote_dong['area'].map(lambda x: a4_to_a3.get(x, x))

    total_votes = vote_dong.groupby(['area3_uid', 'vote_type'])['vote'].sum().unstack(fill_value=0).reset_index()
    total_votes = total_votes.rename(columns={'early': 'total_early', 'same_day': 'total_same_day'})

    party_agg = vote_dong[vote_dong['is_dem']].groupby(['area3_uid', 'vote_type'])['vote'].sum().unstack(fill_value=0).reset_index()
    party_agg = party_agg.rename(columns={'early': 'early_votes', 'same_day': 'same_day_votes'})

    df_dong = pd.merge(party_agg, total_votes, on='area3_uid')
    
    area3['area2_name'] = area3['area2'].map(dict(zip(area2['uid'], area2['name'])))
    def get_base_municipality(name):
        if not isinstance(name, str): return ""
        if '_' in name: name = name.split('_')[-1]
        name = re.sub(r'(갑|을|병|정|무)$', '', name).strip()
        return name.split()[-1]

    area3['base_si_gun_gu'] = area3['area2_name'].apply(get_base_municipality)
    area3['match_key_si'] = area3['base_si_gun_gu'].apply(lambda x: re.sub(r'(시|군|구)$', '', x).strip())
    area3['dong_name'] = area3['name'].str.strip()

    df_dong = pd.merge(df_dong, area3[['uid', 'match_key_si', 'dong_name', 'sum_people', 'sum_vote']], left_on='area3_uid', right_on='uid')
    
    conn.close()
    return df_dong, area4

# ==========================================
# 3. THE FORENSICS ENGINE (All Tests Combined)
# ==========================================
def run_comprehensive_forensics_suite(df_dong, df_station, df_demo):
    print(f"\n--- [3/3] Executing Comprehensive Forensics Suite ---")
    
    # 1. Merge Census Data
    left_merged = pd.merge(df_dong, df_demo, left_on=['match_key_si', 'dong_name'], right_on=['base_si_gun_gu', 'dong_name'], how='left', indicator=True)
    df_merged = left_merged[left_merged['_merge'] == 'both'].drop(columns=['_merge']).copy()
    
    # Clean up outliers (areas with < 50 votes to prevent noise)
    df_merged = df_merged[(df_merged['total_early'] > 50) & (df_merged['total_same_day'] > 50)].copy()

    # Pre-calculate Demographics & Gaps
    df_merged['early_pct'] = df_merged['early_votes'] / df_merged['total_early']
    df_merged['same_day_pct'] = df_merged['same_day_votes'] / df_merged['total_same_day']
    df_merged['raw_vote_gap'] = df_merged['early_pct'] - df_merged['same_day_pct']
    df_merged['weighted_vote_gap'] = df_merged['raw_vote_gap'] / df_merged['demographic_propensity']
    df_merged['vote_share'] = (df_merged['early_votes'] + df_merged['same_day_votes']) / (df_merged['total_early'] + df_merged['total_same_day'])
    df_merged['turnout'] = df_merged['sum_vote'] / df_merged['sum_people'].replace(0, np.nan)

    print("\n" + "="*50)
    print("   PART 1: STANDARD FORENSICS (DIGITS & DISPERSION)")
    print("="*50)

    # Test A: 2BL (Second-Digit Benford's Law)
    valid_votes = df_merged['early_votes'].astype(str)
    second_digits = valid_votes.str[1].astype(int)
    obs_2bl = second_digits.value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_probs_2bl = [sum(math.log10(1 + (1 / (10 * k + d))) for k in range(1, 10)) for d in range(10)]
    exp_2bl = [p * len(valid_votes) for p in exp_probs_2bl]
    chi_stat_2bl, p_val_2bl = chisquare(f_obs=obs_2bl, f_exp=exp_2bl)
    print(f"[*] 2BL Test -> Chi-Square: {chi_stat_2bl:.2f} | P-Value: {p_val_2bl:.4f}")

    # Test B: Last Digit Uniformity
    last_digits = valid_votes.str[-1].astype(int)
    obs_ld = last_digits.value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_ld = [len(valid_votes) / 10] * 10
    chi_stat_ld, p_val_ld = chisquare(f_obs=obs_ld, f_exp=exp_ld)
    print(f"[*] Last Digit Uniformity -> Chi-Square: {chi_stat_ld:.2f} | P-Value: {p_val_ld:.4f}")

    # Test C: Variance & Dispersion
    variances = df_merged.groupby('match_key_si')['vote_share'].std().dropna() * 100
    print(f"[*] Variance Test -> Avg StDev between neighborhoods: {variances.mean():.2f}%")

    print("\n" + "="*50)
    print("   PART 2: TESTING CONSPIRACY THEORIES")
    print("="*50)

    # Test D: The Constant Gap / Fixed Padding Theory
    mean_gap, std_gap = df_merged['raw_vote_gap'].mean(), df_merged['raw_vote_gap'].std()
    print(f"[*] 'Constant Gap' Theory -> Mean Gap: +{mean_gap*100:.2f}% | StDev: {std_gap*100:.2f}%")
    print(f"    (High StDev disproves algorithmic fixed padding)")

    # Test E: The Algorithmic Ratio Theory
    corr, _ = pearsonr(df_merged['same_day_pct'], df_merged['early_pct'])
    r2 = corr ** 2
    print(f"[*] 'Algorithmic Ratio' Theory -> R-Squared: {r2:.4f}")
    print(f"    ({(1-r2)*100:.2f}% unexplained variance disproves a strict mathematical formula)")

    # Test F: The 63:36 Metropolitan Aggregate Theory
    early_stations = df_station[df_station['is_early'] == True].copy()
    metro_agg = early_stations.groupby('metro_zone')[['dem_votes', 'con_votes']].sum()
    metro_agg['total'] = metro_agg['dem_votes'] + metro_agg['con_votes']
    metro_agg['dem_ratio'] = (metro_agg['dem_votes'] / metro_agg['total']) * 100
    
    print(f"\n[*] '63:36' Theory -> Macroscopic Aggregates:")
    for zone, row in metro_agg.iterrows():
        if zone != "Other" and row['total'] > 0:
            print(f"    - {zone}: Dem {row['dem_ratio']:.2f}% | Con {100-row['dem_ratio']:.2f}%")
    
    early_stations['micro_dem_ratio'] = (early_stations['dem_votes'] / (early_stations['dem_votes'] + early_stations['con_votes']).replace(0, np.nan)) * 100
    micro_std = early_stations['micro_dem_ratio'].std()
    print(f"    - Microscopic Station Variance (StDev): {micro_std:.2f}%")
    print(f"    (High microscopic variance proves the 63:36 aggregate is just the Law of Large Numbers)")

    # Test G: The Invalid Vote Anomaly Theory
    early_clean = early_stations.dropna(subset=['invalid_rate', 'micro_dem_ratio'])
    invalid_corr, invalid_pval = pearsonr(early_clean['invalid_rate'], early_clean['micro_dem_ratio'])
    print(f"\n[*] 'Invalid Vote' Theory -> Correlation: {invalid_corr:.4f} (P-Val: {invalid_pval:.4f})")
    print(f"    (Correlation near zero proves invalid votes were not weaponized)")

    return df_merged, obs_ld, exp_ld, variances, r2

# ==========================================
# 4. MEGA DASHBOARD VISUALIZATION
# ==========================================
def plot_mega_dashboard(df, obs_ld, exp_ld, variances, r2):
    print("\nGenerating Mega Dashboard Image...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Comprehensive Election Forensics Dashboard", fontsize=20, fontweight='bold', y=0.98)

    # 1. Raw vs Age-Adjusted Gap (Top Row)
    axes[0, 0].hist(df['raw_vote_gap'] * 100, bins=50, color='red', alpha=0.6, edgecolor='black')
    axes[0, 0].axvline(df['raw_vote_gap'].mean() * 100, color='black', linestyle='dashed', linewidth=2)
    axes[0, 0].set_title('1. Raw Early vs Same-Day Gap')
    axes[0, 0].set_xlabel('Difference (%)')

    axes[0, 1].hist(df['weighted_vote_gap'] * 100, bins=50, color='green', alpha=0.6, edgecolor='black')
    axes[0, 1].axvline(df['weighted_vote_gap'].mean() * 100, color='black', linestyle='dashed', linewidth=2)
    axes[0, 1].set_title('2. Gap Adjusted by Neighborhood Demographics')
    axes[0, 1].set_xlabel('Weighted Difference Score')

    # 2. Algorithmic Check & Fingerprint (Middle Row)
    min_v = min(df['same_day_pct'].min(), df['early_pct'].min()) * 100
    max_v = max(df['same_day_pct'].max(), df['early_pct'].max()) * 100
    axes[1, 0].scatter(df['same_day_pct']*100, df['early_pct']*100, alpha=0.3, color='blue', edgecolor='k', s=20)
    axes[1, 0].plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect 1:1 Correlation')
    axes[1, 0].set_title(f'3. Algorithmic Ratio Check (R² = {r2:.4f})')
    axes[1, 0].set_xlabel('Same-Day Democratic Share (%)')
    axes[1, 0].set_ylabel('Early Democratic Share (%)')
    axes[1, 0].legend()

    fingerprint_df = df[(df['turnout'] <= 1.0) & (df['turnout'] > 0)]
    hb = axes[1, 1].hexbin(fingerprint_df['turnout'] * 100, fingerprint_df['vote_share'] * 100, gridsize=30, cmap='inferno', mincnt=1)
    fig.colorbar(hb, ax=axes[1, 1], label='Number of Districts')
    axes[1, 1].set_title('4. Election Fingerprint (Turnout vs Vote Share)')
    axes[1, 1].set_xlabel('Voter Turnout (%)')
    axes[1, 1].set_ylabel('Democratic Vote Share (%)')

    # 3. Digits & Variance (Bottom Row)
    axes[2, 0].bar(range(10), obs_ld, color='salmon', alpha=0.8, edgecolor='black', label='Observed')
    axes[2, 0].plot(range(10), exp_ld, color='black', linestyle='dashed', linewidth=2, label='Expected Uniform')
    axes[2, 0].set_title('5. Last Digit Uniformity Test')
    axes[2, 0].set_xlabel('Last Digit (0-9)')
    axes[2, 0].set_xticks(range(10))
    axes[2, 0].legend()

    axes[2, 1].hist(variances, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[2, 1].set_title('6. Dispersion Test (Vote Share Variance by City)')
    axes[2, 1].set_xlabel('Standard Deviation of Vote Share (%)')
    axes[2, 1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('mega_forensics_dashboard.png', dpi=300)
    print("Saved mega dashboard to 'mega_forensics_dashboard.png'")

# ==========================================
# EXECUTION TRIGGER
# ==========================================
if __name__ == "__main__":
    df_demo = fetch_kosis_demographics_csv()
    df_dong, df_station = fetch_election_data()
    
    if not df_dong.empty and not df_station.empty:
        df_merged, obs_ld, exp_ld, variances, r2 = run_comprehensive_forensics_suite(df_dong, df_station, df_demo)
        plot_mega_dashboard(df_merged, obs_ld, exp_ld, variances, r2)
