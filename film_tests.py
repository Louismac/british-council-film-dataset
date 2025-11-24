#!/usr/bin/env python3
"""
Statistical Analysis of Film Festival Data

Valid statistical tests given data limitations:
- We only know films that GOT INTO festivals
- We don't know which festivals films applied to
- Missing festival data means "no festivals mentioned" not "rejected"
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.multitest import multipletests
import json
from collections import Counter, defaultdict
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


def load_data(json_file='films_data.json'):
    """Load and prepare data"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create flat structure: one row per film-festival pair
    rows = []
    for film in data:
        if film['festivals']:
            for fest in film['festivals']:
                #remove festival names that were incorrectly scraped
                if "festival" in fest['festival'] or len(fest['festival'].split())<7:
                    rows.append({
                        'title': film['title'],
                        'year': film['year'],
                        'type': film['type'],
                        'genre': film['genre'],
                        'categories': film['categories'],
                        'director': film['director'],
                        'production_company': film['production_company'],
                        'festival': fest['festival'],
                        'festival_status': fest['status'],
                        'has_festival': True
                    })
        else:
            # Films with no festival mentions
            rows.append({
                'title': film['title'],
                'year': film['year'],
                'type': film['type'],
                'genre': film['genre'],
                'categories': film['categories'],
                'director': film['director'],
                'production_company': film['production_company'],
                'festival': None,
                'festival_status': None,
                'has_festival': False
            })
    df = pd.DataFrame(rows)


    return df, data

def genre_representation(df):
    """
    
    Question: Are certain genres over/under-represented at specific festivals?
    Note: Films can have multiple genres (comma-separated), so we count each genre tag
    
    Approach: Each film contributes to multiple genre counts
    - "Drama, Thriller" counts as +1 for Drama AND +1 for Thriller
    - Compare genre tag frequencies at festivals vs overall
    """
    print("\n" + "="*80)
    print("TEST 1B: GENRE REPRESENTATION AT FESTIVALS")
    print("="*80)
    print("\nNull Hypothesis: Genre tag distribution at a festival matches")
    print("the overall distribution of genre tags across festival-accepted films")
    print("\nNote: Films can have multiple genres, each counted separately\n")
    
    # Only look at films that got into festivals
    festival_films = df[df['has_festival']].copy()

    tag = "genre"
    
    # Explode genres: convert comma-separated genres to individual rows
    def explode_genres(df_input):
        """Convert comma-separated genre string to list of individual genres"""
        rows = []
        for _, row in df_input.iterrows():
            if pd.notna(row[tag]) and row[tag]:
                # Split on comma, strip whitespace
                genres = [g.strip() for g in str(row[tag]).split(',') if g.strip()]
                for genre in genres:
                    row_copy = row.copy()
                    row_copy['genre_single'] = genre
                    rows.append(row_copy)
            else:
                # No genre info
                row_copy = row.copy()
                row_copy['genre_single'] = 'Unknown'
                rows.append(row_copy)
        return pd.DataFrame(rows)
    
    # Get unique films first (to avoid counting same film multiple times if at multiple festivals)
    all_festival_films = festival_films.drop_duplicates('title')
    all_genres_exploded = explode_genres(all_festival_films)
    
    # Overall genre distribution (across all festival films)
    overall_genre_dist = all_genres_exploded['genre_single'].value_counts()
    total_genre_tags = len(all_genres_exploded)
    
    print(f"Overall genre distribution across {len(all_festival_films)} films")
    print(f"({all_genres_exploded["genre_single"].nunique()} total genre tags):")
    for genre, count in overall_genre_dist.head(15).items():
        print(f"  {genre}: {count} tags ({count/total_genre_tags:.1%})")
    
    results = []
    
    # For each major festival
    festival_counts = festival_films['festival'].value_counts()
    major_festivals = festival_counts[festival_counts >= 20].index
    
    for festival in major_festivals:
        fest_films = festival_films[festival_films['festival'] == festival]
        fest_titles = fest_films['title'].unique()
        
        # Get films at this festival (unique)
        fest_film_data = festival_films[festival_films['title'].isin(fest_titles)]
        fest_unique_films = fest_film_data.drop_duplicates('title')
        
        # Explode genres for this festival
        fest_genres_exploded = explode_genres(fest_unique_films)
        fest_genre_dist = fest_genres_exploded['genre_single'].value_counts()
        
        # Total genre tags at this festival
        n_fest_genre_tags = len(fest_genres_exploded)
        n_fest_films = len(fest_unique_films)
        
        # Only use genres that appear in both overall and festival data
        # and have sufficient expected frequency
        all_genres = sorted(overall_genre_dist.index.union(fest_genre_dist.index))
        
        # First pass: identify which genres meet the expected >= 5 requirement
        valid_genres = []
        # print(festival)
        for genre in all_genres:
            exp = (overall_genre_dist.get(genre, 0) / total_genre_tags) * n_fest_genre_tags
            # print(genre, exp)
            if exp >= 5:
                valid_genres.append(genre)
        
        # Need at least 2 genres for chi-square
        if len(valid_genres) < 2:
            print(f"\nSkipping {festival}: Insufficient genres with expected frequency >= 5")
            continue
        
        # Second pass: calculate observed and expected only for valid genres
        observed = []
        expected = []
        
        for genre in valid_genres:
            obs = fest_genre_dist.get(genre, 0)
            observed.append(obs)
        
        # Calculate expected proportions based ONLY on valid genres
        # (renormalize to exclude rare genres)
        valid_genre_overall_count = sum(overall_genre_dist.get(g, 0) for g in valid_genres)
        valid_genre_fest_count = sum(observed)
        
        for genre in valid_genres:
            # Expected = (proportion among valid genres) × (total valid genre tags at festival)
            exp = (overall_genre_dist.get(genre, 0) / valid_genre_overall_count) * valid_genre_fest_count
            expected.append(exp)
        
        # Now sums should match exactly
        if abs(sum(observed) - sum(expected)) > 0.01:
            print(f"\nWARNING: {festival} - observed sum {sum(observed):.1f} != expected sum {sum(expected):.1f}")
            print(f"  Valid genres: {len(valid_genres)}, Obs tags: {valid_genre_fest_count}, Exp tags: {sum(expected):.1f}")
            continue
        
        chi2, p_value = stats.chisquare(observed, expected)
        
        # Calculate effect size (Cramér's V)
        # Use number of films, not tags, for effect size
        cramers_v = np.sqrt(chi2 / n_fest_films)
        
        # Which genres are over/under represented?
        genre_effects = {}
        for i, genre in enumerate(valid_genres):
            if expected[i] > 0:
                ratio = observed[i] / expected[i]
                genre_effects[genre] = ratio
        
        results.append({
            'festival': festival,
            'n_films': n_fest_films,
            'n_genre_tags': n_fest_genre_tags,
            'n_genres_tested': len(valid_genres),
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cramers_v': cramers_v,
            'genre_effects': genre_effects,
            'top_overrep': max(genre_effects.items(), key=lambda x: x[1]) if genre_effects else None,
            'top_underrep': min(genre_effects.items(), key=lambda x: x[1]) if genre_effects else None
        })
    
    if not results:
        print("\nNo festivals had sufficient data for chi-square test")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results).sort_values('p_value')
    
    print("\n" + "="*80)
    print("Festivals with significant genre bias (p < 0.05):")
    print("="*80)
    
    for _, row in results_df[results_df['significant']].iterrows():
        print(f"\n{row['festival']} (n={row['n_films']} films, {row['n_genre_tags']} genre tags)")
        print(f"  χ² = {row['chi2']:.2f}, p = {row['p_value']:.4f}, Cramér's V = {row['cramers_v']:.3f}")
        print(f"  Tested {row['n_genres_tested']} genres with sufficient frequency")
        
        # Show genre effects
        print(f"  Genre representation (observed/expected):")
        for genre, ratio in sorted(row['genre_effects'].items(), key=lambda x: x[1], reverse=True):
            if ratio > 1.2 or ratio < 0.8:  # Only show notable differences
                direction = "over" if ratio > 1 else "under"
                print(f"    {genre}: {ratio:.2f}x ({direction}-represented)")
    
    print("\n" + "="*80)
    print("Festivals with no significant genre bias (p >= 0.05):")
    print("="*80)
    for _, row in results_df[~results_df['significant']].iterrows():
        print(f"  {row['festival']}: p = {row['p_value']:.3f}")
    
    return results_df


def festival_prestige_hierarchy(df):
    """
    Festival Prestige Hierarchy via Co-occurrence
    
    Question: Is there a hierarchy among festivals? (measured by co-occurrence patterns)
    Method: Jaccard similarity and conditional probabilities
    """
    print("\n" + "="*80)
    print("TEST 2: FESTIVAL HIERARCHY VIA CO-OCCURRENCE ANALYSIS")
    print("="*80)
    print("\nAnalysis: Which festivals commonly appear together on the same film?\n")
    
    festival_films = df[df['has_festival']].copy()
    
    # Get festival sets for each film
    film_festivals = festival_films.groupby('title')['festival'].apply(set).to_dict()
    
    # Only films with multiple festivals
    multi_fest_films = {title: fests for title, fests in film_festivals.items() if len(fests) > 1}
    
    print(f"Films with multiple festivals: {len(multi_fest_films)}")
    print(f"Films with single festival: {len(film_festivals) - len(multi_fest_films)}\n")
    
    # Calculate co-occurrence matrix
    festivals = festival_films['festival'].value_counts()
    major_festivals = festivals[festivals >= 10].index.tolist()
    
    cooccurrence = pd.DataFrame(0, index=major_festivals, columns=major_festivals, dtype=float)
    
    for title, fests in multi_fest_films.items():
        for f1, f2 in combinations(fests, 2):
            if f1 in major_festivals and f2 in major_festivals:
                cooccurrence.loc[f1, f2] += 1
                cooccurrence.loc[f2, f1] += 1
    
    # Calculate conditional probabilities P(Festival B | Festival A)
    conditional_probs = pd.DataFrame(index=major_festivals, columns=major_festivals, dtype=float)
    
    festival_counts = {fest: festivals[fest] for fest in major_festivals}
    
    for fest_a in major_festivals:
        for fest_b in major_festivals:
            if fest_a != fest_b:
                # P(B appears | A appears)
                p_b_given_a = cooccurrence.loc[fest_a, fest_b] / festival_counts[fest_a]
                conditional_probs.loc[fest_a, fest_b] = p_b_given_a
    
    # Find strongest relationships
    print("Strongest festival co-occurrence patterns:")
    print("(If film gets into Festival A, probability it also gets into Festival B)\n")
    
    results = []
    for fest_a in major_festivals:
        for fest_b in major_festivals:
            if fest_a != fest_b:
                prob = conditional_probs.loc[fest_a, fest_b]
                count = cooccurrence.loc[fest_a, fest_b]
                if count >= 3:  # Minimum threshold
                    results.append({
                        'festival_A': fest_a,
                        'festival_B': fest_b,
                        'co_occurrences': count,
                        'P(B|A)': prob,
                        'n_A': festival_counts[fest_a]
                    })
    
    results_df = pd.DataFrame(results).sort_values('P(B|A)', ascending=False)
    
    for i, row in results_df.head(15).iterrows():
        print(f"{row['festival_A']} → {row['festival_B']}")
        print(f"  P(B|A) = {row['P(B|A)']:.2%} ({row['co_occurrences']:.0f}/{row['n_A']:.0f} films)")
    
    return results_df, conditional_probs


def festival_selectivity_comparison(df):
    """
    
    Question: Do some festivals have more "selective" profiles than others?
    Method: Compare the festival portfolio sizes (films accepted per festival)
    """
    print("\n" + "="*80)
    print("TEST 3: FESTIVAL SELECTIVITY COMPARISON")
    print("="*80)
    print("\nAnalysis: Festivals that accept fewer films are considered more selective\n")
    
    festival_films = df[df['has_festival']].copy()
    
    # Count unique films per festival
    festival_stats = festival_films.groupby('festival').agg({
        'title': 'nunique',
        'year': lambda x: len(x.unique())
    }).rename(columns={'title': 'n_films', 'year': 'n_years'})
    
    festival_stats['films_per_year'] = festival_stats['n_films'] / festival_stats['n_years']
    festival_stats = festival_stats.sort_values('n_films')
    
    # Statistical test: Are film counts significantly different?
    # Use bootstrap to get confidence intervals
    
    print("Most selective festivals (fewest films in database):")
    for fest, row in festival_stats.head(10).iterrows():
        print(f"  {fest}: {row['n_films']} films ({row['films_per_year']:.1f}/year)")
    
    print("\nLeast selective festivals (most films in database):")
    for fest, row in festival_stats.tail(10).iterrows():
        print(f"  {fest}: {row['n_films']} films ({row['films_per_year']:.1f}/year)")
    
    return festival_stats

def genre_festival_affinity(df):
    """
    
    Question: For each genre, which festivals show statistical affinity?
    Method: 2x2 contingency table + Fisher's exact test
    Note: Films can have multiple genres - we test if genre TAG appears at festival
    """
    print("\n" + "="*80)
    print("TEST 4B: GENRE-FESTIVAL AFFINITY (FISHER'S EXACT TEST)")
    print("="*80)
    print("\nQuestion: Are certain genres more likely to appear at certain festivals?")
    print("Note: Films can have multiple genres, testing at genre tag level\n")
    
    festival_films = df[df['has_festival']].copy()

    tag = 'genre'
    
    # Explode genres
    def explode_genres(df_input):
        """Convert comma-separated genre string to individual rows"""
        rows = []
        for _, row in df_input.iterrows():
            if pd.notna(row[tag]) and row[tag]:
                genres = [g.strip() for g in str(row[tag]).split(',') if g.strip()]
                for genre in genres:
                    row_copy = row.copy()
                    row_copy['genre_single'] = genre
                    rows.append(row_copy)
        return pd.DataFrame(rows)
    
    # Get all unique films with their genres (avoid counting same film twice)
    all_festival_films = festival_films.drop_duplicates('title')
    all_genres_exploded = explode_genres(all_festival_films)
    
    # Get films that have each genre (as sets for efficient operations)
    genre_film_sets = {}
    for genre in all_genres_exploded['genre_single'].unique():
        genre_films = all_genres_exploded[all_genres_exploded['genre_single'] == genre]['title'].unique()
        genre_film_sets[genre] = set(genre_films)
    
    # Major festivals
    festival_counts = festival_films['festival'].value_counts()
    major_festivals = festival_counts[festival_counts >= 15].index
    
    # Major genres (appear in at least 10 films)
    genre_counts = all_genres_exploded.groupby('genre_single')['title'].nunique()
    major_genres = genre_counts[genre_counts >= 10].index
    
    print(f"Testing {len(major_festivals)} festivals × {len(major_genres)} genres")
    print(f"Total tests: {len(major_festivals) * len(major_genres)}\n")
    
    results = []
    
    for festival in major_festivals:
        # Films at this festival
        fest_films = festival_films[festival_films['festival'] == festival]['title'].unique()
        fest_film_set = set(fest_films)
        
        for genre in major_genres:
            # Build 2x2 contingency table
            # Rows: Has Genre? (Yes/No)
            # Cols: At Festival? (Yes/No)
            
            genre_film_set = genre_film_sets[genre]
            all_film_set = set(all_festival_films['title'].unique())
            
            a = len(fest_film_set & genre_film_set)  # Genre AND Festival
            b = len(genre_film_set - fest_film_set)  # Genre but NOT Festival
            c = len(fest_film_set - genre_film_set)  # Festival but NOT Genre
            d = len(all_film_set - fest_film_set - genre_film_set)  # Neither
            
            # Only test if we have data in all cells
            if a > 0 and b > 0 and c > 0 and d > 0:
                # Fisher's exact test
                table = [[a, b], [c, d]]
                oddsratio, p_value = fisher_exact(table)
                
                # Calculate rates for interpretation
                rate_at_fest = a / (a + c)
                rate_overall = (a + b) / (a + b + c + d)
                
                results.append({
                    'festival': festival,
                    tag: genre,
                    'n_genre_at_fest': a,
                    'n_fest_films': len(fest_film_set),
                    'n_genre_overall': len(genre_film_set),
                    'odds_ratio': oddsratio,
                    'p_value': p_value,
                    'rate_at_fest': rate_at_fest,
                    'rate_overall': rate_overall
                })
    
    if len(results) == 0:
        print("No valid festival-genre pairs to test")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    # Multiple testing correction (Bonferroni)
    results_df['p_adjusted'] = multipletests(results_df['p_value'], method='bonferroni')[1]
    results_df['significant'] = results_df['p_adjusted'] < 0.05
    
    results_df = results_df.sort_values('odds_ratio', ascending=False)
    
    print("="*80)
    print("Significant genre-festival affinities (after Bonferroni correction):")
    print("="*80)
    
    sig_results = results_df[results_df['significant']]
    
    if len(sig_results) == 0:
        print("\nNo significant affinities after multiple testing correction")
        print("(This is common with many tests and strict Bonferroni correction)\n")
    else:
        # Strong positive affinities (OR > 1)
        positive = sig_results[sig_results['odds_ratio'] > 1]
        if len(positive) > 0:
            print("\nPOSITIVE AFFINITIES (over-represented):")
            for _, row in positive.head(20).iterrows():
                print(f"\n{row[tag]} at {row['festival']}")
                print(f"  Odds Ratio: {row['odds_ratio']:.2f}")
                print(f"  {row['n_genre_at_fest']} films with {row[tag]} at festival")
                print(f"  {row['rate_at_fest']:.1%} of festival vs {row['rate_overall']:.1%} overall")
                print(f"  p = {row['p_value']:.6f}, p_adj = {row['p_adjusted']:.6f}")
        
        # Strong negative affinities (OR < 1)
        negative = sig_results[sig_results['odds_ratio'] < 1]
        if len(negative) > 0:
            print("\n\nNEGATIVE AFFINITIES (under-represented):")
            for _, row in negative.head(15).iterrows():
                print(f"\n{row[tag]} at {row['festival']}")
                print(f"  Odds Ratio: {row['odds_ratio']:.2f}")
                print(f"  {row['n_genre_at_fest']} films with {row[tag]} at festival")
                print(f"  {row['rate_at_fest']:.1%} of festival vs {row['rate_overall']:.1%} overall")
                print(f"  p = {row['p_value']:.6f}, p_adj = {row['p_adjusted']:.6f}")
    
    # Show top affinities even if not significant after correction
    print("\n" + "="*80)
    print("Top 10 strongest affinities (regardless of significance):")
    print("="*80)
    for _, row in results_df.head(10).iterrows():
        sig_marker = "***" if row['significant'] else ""
        print(f"\n{row[tag]} at {row['festival']} {sig_marker}")
        print(f"  OR: {row['odds_ratio']:.2f}, p: {row['p_value']:.6f}, p_adj: {row['p_adjusted']:.6f}")
        print(f"  {row['rate_at_fest']:.1%} vs {row['rate_overall']:.1%}")
    
    return results_df

def genre_temporal_trends(df):
    """
    
    Question: Are certain genres becoming more/less common at festivals over time?
    Method: Spearman correlation on genre tag proportions over time
    Note: Films can have multiple genres, each counted separately
    """
    print("\n" + "="*80)
    print("TEST 6B: TEMPORAL TRENDS IN GENRE POPULARITY")
    print("="*80)
    print("\nQuestion: How have genre preferences changed over time at festivals?")
    print("Note: Films can have multiple genres, each counted separately\n")
    
    festival_films = df[df['has_festival']].copy()
    
    # Convert year to numeric
    festival_films['year_numeric'] = pd.to_numeric(festival_films['year'], errors='coerce')
    festival_films = festival_films.dropna(subset=['year_numeric'])
    tag = "categories"
    # Explode genres
    def explode_genres(df_input):
        """Convert comma-separated genre string to individual rows"""
        rows = []
        for _, row in df_input.iterrows():
            if pd.notna(row[tag]) and row[tag]:
                genres = [g.strip() for g in str(row[tag]).split(',') if g.strip()]
                for genre in genres:
                    row_copy = row.copy()
                    row_copy['genre_single'] = genre
                    rows.append(row_copy)
            else:
                row_copy = row.copy()
                row_copy['genre_single'] = 'Unknown'
                rows.append(row_copy)
        return pd.DataFrame(rows)
    
    # Get unique films per year (avoid double-counting films at multiple festivals)
    unique_films_per_year = festival_films.drop_duplicates(['title', 'year_numeric'])
    
    # Explode genres
    exploded = explode_genres(unique_films_per_year)
    
    # Only years with sufficient data
    year_counts = exploded.groupby('year_numeric')['title'].nunique()
    valid_years = year_counts[year_counts >= 10].index
    exploded = exploded[exploded['year_numeric'].isin(valid_years)]
    
    if len(valid_years) <= 3:
        print("Insufficient temporal data for trend analysis (need >3 years)")
        return None
    
    print(f"Analyzing trends from {int(valid_years.min())} to {int(valid_years.max())}")
    print(f"Across {len(valid_years)} years with sufficient data\n")
    
    # Get genre tag counts by year
    yearly_genres = exploded.groupby(['year_numeric', 'genre_single']).size().unstack(fill_value=0)
    
    # Convert to proportions
    yearly_proportions = yearly_genres.div(yearly_genres.sum(axis=1), axis=0)
    
    # Only analyze genres that appear in at least 3 years with >1% share
    genres_to_analyze = []
    for genre in yearly_proportions.columns:
        # Check if genre appears substantially in multiple years
        years_above_threshold = (yearly_proportions[genre] > 0.01).sum()
        if years_above_threshold >= 3:
            genres_to_analyze.append(genre)
    
    print(f"Analyzing {len(genres_to_analyze)} genres that appear consistently\n")
    
    # Test for trends
    trends = []
    for genre in genres_to_analyze:
        years = yearly_proportions.index.values
        proportions = yearly_proportions[genre].values
        
        # Spearman correlation
        rho, p_value = stats.spearmanr(years, proportions)
        
        # Get actual tag counts for context
        start_count = yearly_genres.loc[years[0], genre]
        end_count = yearly_genres.loc[years[-1], genre]
        
        trends.append({
            tag: genre,
            'spearman_rho': rho,
            'p_value': p_value,
            'trend': 'increasing' if rho > 0 else 'decreasing',
            'start_prop': proportions[0],
            'end_prop': proportions[-1],
            'change': proportions[-1] - proportions[0],
            'start_count': start_count,
            'end_count': end_count
        })
    
    trends_df = pd.DataFrame(trends).sort_values('p_value')
    
    # Report significant trends
    print("="*80)
    print("Significant temporal trends (p < 0.05):")
    print("="*80)
    
    sig_trends = trends_df[trends_df['p_value'] < 0.05]
    
    if len(sig_trends) == 0:
        print("\nNo significant genre trends detected")
    else:
        # Separate increasing and decreasing
        increasing = sig_trends[sig_trends['spearman_rho'] > 0].sort_values('spearman_rho', ascending=False)
        decreasing = sig_trends[sig_trends['spearman_rho'] < 0].sort_values('spearman_rho')
        
        if len(increasing) > 0:
            print("\nINCREASING GENRES:")
            for _, row in increasing.iterrows():
                print(f"\n{row[tag]}")
                print(f"  ρ = {row['spearman_rho']:.3f}, p = {row['p_value']:.4f}")
                print(f"  Proportion: {row['start_prop']:.1%} → {row['end_prop']:.1%} ({row['change']:+.1%})")
                print(f"  Tag count: {row['start_count']:.0f} → {row['end_count']:.0f}")
        
        if len(decreasing) > 0:
            print("\nDECREASING GENRES:")
            for _, row in decreasing.iterrows():
                print(f"\n{row[tag]}")
                print(f"  ρ = {row['spearman_rho']:.3f}, p = {row['p_value']:.4f}")
                print(f"  Proportion: {row['start_prop']:.1%} → {row['end_prop']:.1%} ({row['change']:+.1%})")
                print(f"  Tag count: {row['start_count']:.0f} → {row['end_count']:.0f}")
    
    # Report non-significant trends
    print("\n" + "="*80)
    print("Genres with no significant trend (p >= 0.05):")
    print("="*80)
    non_sig = trends_df[trends_df['p_value'] >= 0.05].sort_values(tag)
    for _, row in non_sig.iterrows():
        print(f"  {row[tag]}: p = {row['p_value']:.3f}")
    
    return trends_df

def festival_circuit_clustering(df):
    """
    
    Question: Are there distinct "festival circuits" that films tend to follow?
    Method: Identify common festival combinations
    """
    print("\n" + "="*80)
    print("TEST 8: COMMON FESTIVAL CIRCUITS")
    print("="*80)
    print("\nQuestion: What are the most common festival combinations?\n")
    
    festival_films = df[df['has_festival']].copy()
    
    # Get festival sets for each film
    film_festivals = festival_films.groupby('title')['festival'].apply(lambda x: frozenset(x)).to_dict()
    
    # Count combinations
    circuit_counts = Counter(film_festivals.values())
    
    # Circuits with 2+ festivals
    multi_fest_circuits = {circuit: count for circuit, count in circuit_counts.items() 
                           if len(circuit) >= 2 and count >= 2}
    
    print(f"Found {len(multi_fest_circuits)} distinct multi-festival circuits\n")
    print("Most common festival circuits:\n")
    
    for circuit, count in sorted(multi_fest_circuits.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{count} films: {', '.join(sorted(circuit)[:5])}")
        if len(circuit) > 5:
            print(f"         ... and {len(circuit) - 5} more")
    
    return circuit_counts


def main():
    """Run all statistical tests"""
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'films_data.json'
    
    print("Loading data...")
    df, raw_data = load_data(json_file)
    
    print(f"\nDataset summary:")
    print(f"  Total films: {df['title'].nunique()}")
    print(f"  Films with festival mentions: {df[df['has_festival']]['title'].nunique()}")
    print(f"  Total festival mentions: {len(df[df['has_festival']])}")
    print(f"  Unique festivals: {df['festival'].nunique()}\n")
    
    


if __name__ == "__main__":
    main()