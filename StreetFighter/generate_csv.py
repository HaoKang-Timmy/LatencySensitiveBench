#!/usr/bin/env python3
"""
Battle Win/Loss Statistics Script
Parse battle log files, count win/loss records for each model (including bitwidth), and generate CSV tables.

File naming convention:
- avsb: a and b are model parameter sizes (e.g., 4vs8 means 4B model vs 8B model)
- Default bitwidth is 16-bit
- Special notation like 4vs4_8bit means fp16 vs fp8 battle

Win/Loss record format:
- Agent1 wins: Player1 ... Agent1 won!
- Agent2 wins: Player2 ... Agent2 won!
"""

import os
import re
import csv
import argparse
from collections import defaultdict
from pathlib import Path


def parse_filename(filename):
    """
    Parse filename to extract model information
    
    Args:
        filename: Battle log filename (e.g., "4vs8.log", "4vs4_8bit.log")
    
    Returns:
        tuple: (model1_info, model2_info) each info contains (size, bitwidth)
    """
    # Remove .log suffix
    name = filename.replace('.log', '')
    
    # Check for special bitwidth notation
    if '_8bit' in name:
        # Handle cases like "4vs4_8bit"
        base_name = name.replace('_8bit', '')
        parts = base_name.split('vs')
        if len(parts) == 2:
            model1 = (parts[0], '16')  # First model defaults to 16-bit
            model2 = (parts[1], '8')   # Second model is 8-bit
            return model1, model2
    
    # Handle standard format like "4vs8"
    parts = name.split('vs')
    if len(parts) == 2:
        model1 = (parts[0], '16')  # Default 16-bit
        model2 = (parts[1], '16')  # Default 16-bit
        return model1, model2
    
    raise ValueError(f"Cannot parse filename: {filename}")


def parse_log_file(filepath):
    """
    Parse a single battle log file
    
    Args:
        filepath: Log file path
    
    Returns:
        tuple: (agent1_wins, agent2_wins) win counts
    """
    agent1_wins = 0
    agent2_wins = 0
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                # Look for win records
                if 'won!' in line:
                    if 'Agent1 won!' in line:
                        agent1_wins += 1
                    elif 'Agent2 won!' in line:
                        agent2_wins += 1
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return 0, 0
    
    return agent1_wins, agent2_wins


def create_model_key(size, bitwidth):
    """Create model identifier"""
    if bitwidth == '16':
        return f"{size}B"
    else:
        return f"{size}B_{bitwidth}bit"


def generate_battle_stats(log_directory):
    """
    Generate battle statistics
    
    Args:
        log_directory: Directory containing battle log files
    
    Returns:
        dict: Battle statistics data
    """
    # Store all battle results
    battle_results = defaultdict(lambda: defaultdict(lambda: {'wins': 0, 'losses': 0}))
    all_models = set()
    
    # Iterate through all .log files
    log_files = [f for f in os.listdir(log_directory) if f.endswith('.log')]
    
    for log_file in log_files:
        try:
            # Parse filename
            model1_info, model2_info = parse_filename(log_file)
            model1_key = create_model_key(*model1_info)
            model2_key = create_model_key(*model2_info)
            
            all_models.add(model1_key)
            all_models.add(model2_key)
            
            # Parse battle results
            filepath = os.path.join(log_directory, log_file)
            agent1_wins, agent2_wins = parse_log_file(filepath)
            
            # Record battle results
            # model1 (Agent1) vs model2 (Agent2)
            battle_results[model1_key][model2_key]['wins'] += agent1_wins
            battle_results[model1_key][model2_key]['losses'] += agent2_wins
            
            # model2 (Agent2) vs model1 (Agent1) - reverse record
            battle_results[model2_key][model1_key]['wins'] += agent2_wins
            battle_results[model2_key][model1_key]['losses'] += agent1_wins
            
            print(f"Processing file {log_file}: {model1_key} vs {model2_key}")
            print(f"  {model1_key} wins: {agent1_wins}, {model2_key} wins: {agent2_wins}")
            
        except Exception as e:
            print(f"Error processing file {log_file}: {e}")
            continue
    
    return battle_results, sorted(all_models)


def generate_csv_report(battle_results, all_models, output_file):
    """
    Generate CSV report
    
    Args:
        battle_results: Battle statistics data
        all_models: List of all models
        output_file: Output CSV file path
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['Model'] + [f'vs_{model}' for model in all_models]
        writer.writerow(header)
        
        # Write battle data for each model
        for model1 in all_models:
            row = [model1]
            for model2 in all_models:
                if model1 == model2:
                    # Self vs self, display as "-"
                    row.append('-')
                else:
                    wins = battle_results[model1][model2]['wins']
                    losses = battle_results[model1][model2]['losses']
                    total_games = wins + losses
                    if total_games > 0:
                        win_rate = wins / total_games * 100
                        row.append(f"{wins}W{losses}L({win_rate:.1f}%)")
                    else:
                        row.append('0W0L(0.0%)')
            writer.writerow(row)


def generate_summary_csv(battle_results, all_models, output_file):
    """
    Generate simplified win/loss statistics CSV
    
    Args:
        battle_results: Battle statistics data
        all_models: List of all models
        output_file: Output CSV file path
    """
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['Model'] + all_models
        writer.writerow(header)
        
        # Write win count data for each model
        for model1 in all_models:
            row = [model1]
            for model2 in all_models:
                if model1 == model2:
                    row.append('-')
                else:
                    wins = battle_results[model1][model2]['wins']
                    row.append(wins)
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Count battle wins/losses and generate CSV files')
    parser.add_argument('--log_dir', default='.', help='Battle log files directory (default: current directory)')
    parser.add_argument('--output', default='battle_results.csv', help='Output CSV filename (default: battle_results.csv)')
    parser.add_argument('--summary', default='battle_summary.csv', help='Summary statistics CSV filename (default: battle_summary.csv)')
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.log_dir):
        print(f"Error: Directory {args.log_dir} does not exist")
        return
    
    print(f"Starting to process directory: {args.log_dir}")
    
    # Generate battle statistics
    battle_results, all_models = generate_battle_stats(args.log_dir)
    
    if not all_models:
        print("No battle log files found")
        return
    
    print(f"\nDiscovered models: {', '.join(all_models)}")
    
    # Generate detailed CSV report
    generate_csv_report(battle_results, all_models, args.output)
    print(f"\nDetailed battle report generated: {args.output}")
    
    # Generate simplified CSV report
    generate_summary_csv(battle_results, all_models, args.summary)
    print(f"Simplified win count statistics generated: {args.summary}")
    
    # Print summary information
    print("\n=== Battle Statistics Summary ===")
    for model in all_models:
        total_wins = sum(battle_results[model][opponent]['wins'] 
                        for opponent in all_models if opponent != model)
        total_losses = sum(battle_results[model][opponent]['losses'] 
                          for opponent in all_models if opponent != model)
        total_games = total_wins + total_losses
        win_rate = total_wins / total_games * 100 if total_games > 0 else 0
        print(f"{model}: {total_wins}W{total_losses}L, Win rate: {win_rate:.1f}%")


if __name__ == "__main__":
    main()
