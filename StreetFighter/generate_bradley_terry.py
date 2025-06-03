#!/usr/bin/env python3
"""
Bradley-Terry Model for AI Model Battle Analysis
Calculate model strength ratings based on pairwise battle results using the Bradley-Terry model.

The Bradley-Terry model is a probabilistic model for pairwise comparisons that estimates
the relative strength of competitors based on their win/loss records against each other.

Mathematical Foundation:
- P(i beats j) = exp(θᵢ) / (exp(θᵢ) + exp(θⱼ))
- Where θᵢ is the strength parameter for model i
- Solved using maximum likelihood estimation via logistic regression

Input Format: battle_summary.csv with format:
Model,14B,14B_8bit,4B,4B_8bit,8B,8B_8bit
14B,-,15,24,0,19,0
...

Output: Rankings with Bradley-Terry scores and win probability predictions
"""

import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import argparse
import os
import warnings
warnings.filterwarnings('ignore')


def load_battle_summary(csv_file):
    """
    Load battle summary data from CSV file
    
    Args:
        csv_file: Path to battle summary CSV file
    
    Returns:
        tuple: (battle_matrix_dict, model_names_list)
    """
    battle_matrix = {}
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            model_names = header[1:]  # Skip 'Model' column
            
            for row in reader:
                model_name = row[0]
                battle_matrix[model_name] = {}
                
                for i, opponent in enumerate(model_names):
                    wins = int(row[i + 1]) if row[i + 1] != '-' else 0
                    battle_matrix[model_name][opponent] = wins
        
        print(f"Loaded battle data for {len(model_names)} models")
        return battle_matrix, model_names
        
    except Exception as e:
        print(f"Error loading battle data: {e}")
        return None, None


def prepare_training_data(battle_matrix, model_names, min_battles=0):
    """
    Convert battle matrix to training data for Bradley-Terry model
    
    Args:
        battle_matrix: Dictionary of win counts between models
        model_names: List of model names
        min_battles: Minimum total battles required to include a matchup
    
    Returns:
        tuple: (feature_matrix, outcomes, matchup_info)
    """
    features = []
    outcomes = []
    matchup_info = []
    
    total_pairs = 0
    included_pairs = 0
    total_battles = 0
    
    # Process each unique pair of models
    for i, model_a in enumerate(model_names):
        for j, model_b in enumerate(model_names):
            if i < j:  # Process each pair only once
                wins_a = battle_matrix[model_a][model_b]
                wins_b = battle_matrix[model_b][model_a]
                total_games = wins_a + wins_b
                
                total_pairs += 1
                
                if total_games >= min_battles:
                    included_pairs += 1
                    total_battles += total_games
                    
                    # Create feature vectors for each individual battle
                    # Feature vector: +1 for winner, -1 for loser, 0 for others
                    
                    # Add wins for model_a
                    for _ in range(wins_a):
                        feature_vec = np.zeros(len(model_names))
                        feature_vec[i] = 1   # model_a position
                        feature_vec[j] = -1  # model_b position
                        features.append(feature_vec)
                        outcomes.append(1)   # model_a wins
                    
                    # Add wins for model_b
                    for _ in range(wins_b):
                        feature_vec = np.zeros(len(model_names))
                        feature_vec[i] = 1   # model_a position
                        feature_vec[j] = -1  # model_b position
                        features.append(feature_vec)
                        outcomes.append(0)   # model_b wins
                    
                    matchup_info.append({
                        'model_a': model_a,
                        'model_b': model_b,
                        'wins_a': wins_a,
                        'wins_b': wins_b,
                        'total_battles': total_games,
                        'win_rate_a': wins_a / total_games if total_games > 0 else 0
                    })
    
    features = np.array(features)
    outcomes = np.array(outcomes)
    
    print(f"Training data preparation:")
    print(f"  Model pairs: {total_pairs} total, {included_pairs} included")
    print(f"  Battle samples: {len(features)} from {total_battles} total battles")
    print(f"  Win distribution: {np.mean(outcomes):.3f} (balanced = 0.5)")
    
    return features, outcomes, matchup_info


def fit_bradley_terry(features, outcomes, regularization=0.01, max_iterations=2000):
    """
    Fit Bradley-Terry model using logistic regression
    
    Args:
        features: Feature matrix (n_battles x n_models)
        outcomes: Battle outcomes (1 = first model wins, 0 = second model wins)
        regularization: L2 regularization strength (higher = more conservative)
        max_iterations: Maximum optimization iterations
    
    Returns:
        LogisticRegression: Fitted Bradley-Terry model
    """
    print(f"Fitting Bradley-Terry model:")
    print(f"  Regularization strength: {regularization}")
    print(f"  Maximum iterations: {max_iterations}")
    
    # Convert regularization parameter to sklearn format
    C_parameter = 1.0 / regularization if regularization > 0 else 1e10
    
    model = LogisticRegression(
        fit_intercept=False,    # Bradley-Terry model has no intercept term
        solver='lbfgs',         # Good for small datasets
        max_iter=max_iterations,
        C=C_parameter,
        random_state=42
    )
    
    try:
        model.fit(features, outcomes)
        print(f"  Optimization converged in {model.n_iter_} iterations")
        return model
    except Exception as e:
        print(f"Error fitting Bradley-Terry model: {e}")
        return None


def extract_model_strengths(bt_model, model_names):
    """
    Extract model strength parameters and create rankings
    
    Args:
        bt_model: Fitted Bradley-Terry model
        model_names: List of model names
    
    Returns:
        pd.DataFrame: Model rankings with strength scores
    """
    # Extract strength parameters (log-odds ratios)
    strength_params = bt_model.coef_[0]
    
    # Create results dataframe
    results = pd.DataFrame({
        'model': model_names,
        'strength_score': strength_params
    })
    
    # Sort by strength (higher = stronger)
    results = results.sort_values('strength_score', ascending=False).reset_index(drop=True)
    results['rank'] = range(1, len(results) + 1)
    
    # Calculate expected win rate against average opponent
    avg_strength = np.mean(strength_params)
    results['expected_win_rate'] = 1 / (1 + np.exp(-(results['strength_score'] - avg_strength)))
    
    # Add relative strength (difference from average)
    results['relative_strength'] = results['strength_score'] - avg_strength
    
    return results


def compute_win_probabilities(bt_model, model_names):
    """
    Compute pairwise win probabilities between all models
    
    Args:
        bt_model: Fitted Bradley-Terry model
        model_names: List of model names
    
    Returns:
        pd.DataFrame: Win probability matrix
    """
    n_models = len(model_names)
    prob_matrix = np.zeros((n_models, n_models))
    
    for i in range(n_models):
        for j in range(n_models):
            if i != j:
                # Create feature vector for model i vs model j
                feature_vec = np.zeros(n_models)
                feature_vec[i] = 1
                feature_vec[j] = -1
                
                # Predict probability that model i beats model j
                win_prob = bt_model.predict_proba([feature_vec])[0][1]
                prob_matrix[i][j] = win_prob
            else:
                prob_matrix[i][j] = 0.5  # Self-match
    
    # Create DataFrame with model names as indices
    prob_df = pd.DataFrame(prob_matrix, index=model_names, columns=model_names)
    return prob_df


def save_results(rankings, output_file):
    """
    Save Bradley-Terry rankings to CSV file
    
    Args:
        rankings: Model rankings DataFrame
        output_file: Output file path
    """
    output_columns = ['rank', 'model', 'strength_score', 'relative_strength', 'expected_win_rate']
    rankings[output_columns].to_csv(output_file, index=False, float_format='%.4f')


def print_analysis(rankings, battle_matrix, model_names, win_probabilities):
    """
    Print comprehensive Bradley-Terry analysis
    
    Args:
        rankings: Model rankings DataFrame
        battle_matrix: Original battle data
        model_names: List of model names
        win_probabilities: Win probability matrix
    """
    print("\n" + "="*80)
    print("BRADLEY-TERRY MODEL ANALYSIS")
    print("="*80)
    
    # Main rankings table
    print(f"\n{'Rank':<4} {'Model':<12} {'Strength':<10} {'Relative':<10} {'Expected WR':<12} {'Actual WR':<10} {'Battles':<8}")
    print("-" * 80)
    
    for _, row in rankings.iterrows():
        model = row['model']
        
        # Calculate actual performance from battle data
        total_wins = sum(battle_matrix[model][opp] for opp in model_names if opp != model)
        total_losses = sum(battle_matrix[opp][model] for opp in model_names if opp != model)
        total_battles = total_wins + total_losses
        actual_wr = total_wins / total_battles if total_battles > 0 else 0
        
        print(f"{row['rank']:<4} {model:<12} {row['strength_score']:<10.3f} "
              f"{row['relative_strength']:<10.3f} {row['expected_win_rate']:<12.1%} "
              f"{actual_wr:<10.1%} {total_battles:<8}")
    
    # Model performance summary
    print(f"\n{'MODEL PERFORMANCE SUMMARY':<50}")
    print("-" * 50)
    for model in model_names:
        total_wins = sum(battle_matrix[model][opp] for opp in model_names if opp != model)
        total_losses = sum(battle_matrix[opp][model] for opp in model_names if opp != model)
        strength = rankings[rankings['model'] == model]['strength_score'].iloc[0]
        
        print(f"{model}: {total_wins}W-{total_losses}L, Strength: {strength:.3f}")
    
    # Interesting matchup predictions
    print(f"\n{'MATCHUP PREDICTIONS':<50}")
    print("-" * 50)
    
    sorted_models = rankings['model'].tolist()
    
    print("Champion vs challengers:")
    champion = sorted_models[0]
    for challenger in sorted_models[1:4]:
        prob = win_probabilities.loc[champion, challenger]
        print(f"  {champion} vs {challenger}: {prob:.1%}")
    
    print("\nMost competitive matchups:")
    competitive_matches = []
    for i, model1 in enumerate(sorted_models):
        for model2 in sorted_models[i+1:]:
            prob = win_probabilities.loc[model1, model2]
            competitiveness = 1 - abs(prob - 0.5) * 2  # Higher = more competitive
            competitive_matches.append((model1, model2, prob, competitiveness))
    
    competitive_matches.sort(key=lambda x: x[3], reverse=True)
    for model1, model2, prob, comp in competitive_matches[:3]:
        print(f"  {model1} vs {model2}: {prob:.1%} (competitiveness: {comp:.3f})")


def main():
    parser = argparse.ArgumentParser(
        description='Bradley-Terry model analysis for AI battle results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_bradley_terry.py
  python generate_bradley_terry.py --min_battles 20 --regularization 0.005
  python generate_bradley_terry.py --save_probabilities --verbose
        """
    )
    
    parser.add_argument('--input', default='battle_summary.csv',
                       help='Input battle summary CSV file (default: battle_summary.csv)')
    parser.add_argument('--output', default='bradley_terry_rankings.csv',
                       help='Output rankings CSV file (default: bradley_terry_rankings.csv)')
    parser.add_argument('--min_battles', type=int, default=0,
                       help='Minimum battles required for model pair inclusion (default: 0)')
    parser.add_argument('--regularization', type=float, default=0.01,
                       help='L2 regularization strength, 0.001-0.1 (default: 0.01)')
    parser.add_argument('--max_iter', type=int, default=2000,
                       help='Maximum optimization iterations (default: 2000)')
    parser.add_argument('--save_probabilities', action='store_true',
                       help='Save win probability matrix to separate CSV file')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed analysis and diagnostics')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        print("Expected format: CSV with model names as columns and win counts as values")
        return 1
    
    print("Bradley-Terry Model for AI Battle Analysis")
    print(f"Input file: {args.input}")
    
    # Load battle data
    battle_matrix, model_names = load_battle_summary(args.input)
    if battle_matrix is None:
        return 1
    
    if args.verbose:
        print(f"\nConfiguration:")
        print(f"  Minimum battles per pair: {args.min_battles}")
        print(f"  Regularization strength: {args.regularization}")
        print(f"  Maximum iterations: {args.max_iter}")
    
    # Prepare training data
    features, outcomes, matchup_info = prepare_training_data(
        battle_matrix, model_names, args.min_battles
    )
    
    if len(features) == 0:
        print("Error: No training data available after filtering")
        print(f"Try reducing --min_battles (currently {args.min_battles})")
        return 1
    
    # Fit Bradley-Terry model
    bt_model = fit_bradley_terry(features, outcomes, args.regularization, args.max_iter)
    if bt_model is None:
        return 1
    
    # Extract results
    rankings = extract_model_strengths(bt_model, model_names)
    win_probabilities = compute_win_probabilities(bt_model, model_names)
    
    # Save results
    save_results(rankings, args.output)
    print(f"\nRankings saved to: {args.output}")
    
    if args.save_probabilities:
        prob_file = args.output.replace('.csv', '_win_probabilities.csv')
        win_probabilities.to_csv(prob_file, float_format='%.3f')
        print(f"Win probabilities saved to: {prob_file}")
    
    # Print analysis
    print_analysis(rankings, battle_matrix, model_names, win_probabilities)
    
    print(f"\n{'BRADLEY-TERRY MODEL SUMMARY':<50}")
    print("-" * 50)
    print("✓ Mathematically optimal rankings via maximum likelihood estimation")
    print("✓ Probabilistic win predictions for any model matchup")
    print("✓ Accounts for strength of schedule and opponent quality")
    print(f"✓ Trained on {len(features)} individual battle outcomes")
    
    return 0


if __name__ == "__main__":
    exit(main()) 