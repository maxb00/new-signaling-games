from ..signaling_game import SignalingGame


def run_game(*args) -> str:
    seed = args[0]
    # instance new game container
    game = SignalingGame(*args[1:11])
    if seed is not None:
        game.set_random_seed(seed=seed)
    # run the game
    output_file = game(*args[11:])
    # return image or gif path
    return output_file
