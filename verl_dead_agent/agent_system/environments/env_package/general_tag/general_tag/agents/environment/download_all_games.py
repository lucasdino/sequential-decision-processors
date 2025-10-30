from .download_game_files import get_alfworld_games


def __main__():
    # Example usage of get_alfworld_games
    games = get_alfworld_games()
    print(f"Retrieved {len(games)} ALFWorld games.")
    for game in games[:5]:  # Print first 5 games
        print(game)