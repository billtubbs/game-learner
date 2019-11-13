"""Implementation of True online TD-Lambda algorithm described in
Sutton & Barto's book and experiment to test its performance on the
19-step random walk environment.  Produces a plot similar to
Figure 12.8 in the book (see Chapter 12, section 12.5).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from randomwalk import RandomWalkGame


def calculate_true_values(game):
    """Returns a list of the true values of states in a
    RandomWalk game.
    """

    xp = [0, game.size+1]
    fp = [-1.0, 1.0]

    states = [s for s in game.states if s not in game.terminal_states]
    values = np.interp(np.arange(game.size + 2), xp, fp)[1:-1]
    true_values = pd.Series(values, index=states)
    
    return true_values


def calculate_rms_error(values, true_values):
    """Root-mean-squared error of values compared to 
    true values.
    """
    return np.sqrt(((np.array(values) - true_values)**2).mean())


def test_rms_error_functions():
    # Initialize random-walk environment
    game = RandomWalkGame(size=19, terminal_rewards={'T1': -1.0, 'T2': 1.0})
    n_states = len(game.states) - len(game.terminal_states)
    n_states

    # Test true values are correct
    true_values = calculate_true_values(game)
    values = np.zeros(n_states)
    error = calculate_rms_error(values, true_values)
    assert error == 0.5477225575051662  # Calculated using code from Sutton & Barto


def true_online_td_lambda_weight_update(weights, features_next_state, features, 
                                        value_old, reward, z, 
                                        lam, gamma, learning_rate):
    """Updates the weights of a value function for the 
    previous state using the true online TD-Lambda 
    algorithm.
    
    Args:
        weights (array): Array of weights (value function 
            parameters).
        value_current_state (float): Value estimate for current state.
        value_prev_state (float): Value estimate for previous state.
        reward (float): Reward at current state.
        dv_dw (array): Partial derivatives of value function
            w.r.t. the weights at current state.
        z (array): Eligibility trace vector.
        lam (float): Lambda parameter.
        gamma (float): Discount factor.
        learning_rate (float): Learning rate parameter.
    """

    # True online TD-lambda update
    value = np.dot(weights, features)
    value_next_state = np.dot(weights, features_next_state)
    td_error = reward + gamma * value_next_state - value
    z[:] = gamma * lam * z + (1 - learning_rate * gamma * lam * z.dot(features)) * features
    weights[:] = weights + learning_rate * (td_error + value - value_old) * z \
                 - learning_rate * (value - value_old) * features
    features[:] = features_next_state
    
    return value_next_state


def run_random_walk_with_true_online_td_lambda(lam=0.5, learning_rate=0.01, gamma=1.0, 
                                               n_episodes=10, n_reps=1, initial_value=0.0, 
                                               size=19, seed=None, show=False):
    """Run n_episodes of random walk and calculate the root-
    mean-squared errror of the value function after the 
    last episode.  If n_reps > 1 then the experiment is 
    repeated n_reps times and the average rms_error returned.
    """

    # Initialize environment
    terminal_rewards = {'T1': -1.0, 'T2': 1.0}
    game = RandomWalkGame(size=size, terminal_rewards=terminal_rewards)
    role = 1
    true_values = calculate_true_values(game)
    states = np.array(game.states)

    # Dedicated random number generator
    rng = np.random.RandomState(seed)

    # Repeat n_reps times
    rms_errors = []
    for repetition in range(n_reps):
        # Initialise value function
        weights = np.array([0.0 if s in game.terminal_states else initial_value
                            for s in game.states])
        z = np.zeros_like(weights)

        for episode in range(0, n_episodes):
            game.reset()
            state = game.generate_state_key(game.start_state, role)
            features = np.array([1.0 if s == state else 0.0 for s in game.states])
            features_next_state = np.zeros_like(features)
            values = np.zeros_like(features)
            value_old = 0.0
            t = 0
            while not game.game_over:

                # Behaviour policy
                move = rng.choice(game.available_moves())
                game.make_move([1, move])

                # Get rewards
                if not game.game_over:
                    reward = game.get_rewards()[role]
                else:
                    reward = game.get_terminal_rewards()[role]
                next_state = game.generate_state_key(game.state, role)
                features_next_state = (states == next_state).astype(float)
                # Note the above is slightly faster than but equivalent to:
                # features_next_state[:] = [1.0 if s == next_state else 0.0 for s in game.states]

                value_old = true_online_td_lambda_weight_update(
                    weights, features_next_state, features, 
                    value_old, reward, z, 
                    lam, gamma, learning_rate
                )
                
                # Update timestep
                t += 1

            # Average RMS error at end of each episode
            # (Note: In this experiment, the weights are the values)
            rms_error = calculate_rms_error(weights[1:-1], true_values)
            rms_errors.append(rms_error)

        # All episodes complete

    # All repetitions complete
    avg_rms_error = np.array(rms_errors).mean()

    # Return param values
    params = {
        'lam': lam,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'n_episodes': n_episodes
    }

    return params, avg_rms_error


def plot_results(results_df, groupby, x_label='learning_rate', y_label='RMS Error', 
                 ylim=None, title=None, filename=None):
    results_by_group = results_df.groupby(groupby)
    fig, ax = plt.subplots()
    for param_value, df in results_by_group:
        y = df[y_label]
        x = df[x_label]
        ax.plot(x, y, label=f'{groupby} = {param_value:.2f}')
    if title is not None:
        plt.title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_ylim(*ylim)
    plt.legend()
    plt.grid()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


if __name__ == '__main__':

    test_rms_error_functions()

    results_dir = 'results'
    plot_dir = 'plots'
    exp_name = 'random-walk-true-online-td-lambda'

    # Prepare file folders
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Run many repetitions with varying parameter values
    n_reps = 20  # Textbook uses 100
    n_states = 19
    seed = 1

    # Full range of lambda values
    lam_values = [0.0, 0.4, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0]

    # Full range of alpha values
    alpha_values = np.logspace(-2, 0, 50)
    # For a quicker test use less points:
    #alpha_values = np.logspace(-2, 0, 10)

    # Use this dictionary to accumulate the results
    avg_rms_errors = {}

    print(f"Running experiments...")
    for lam in lam_values:
        print(f"lam: {lam}")
        for alpha in alpha_values:
            params, avg_rms_error = \
                run_random_walk_with_true_online_td_lambda(
                    lam=lam, 
                    learning_rate=alpha,
                    n_reps=n_reps, 
                    size=n_states,
                    seed=seed
                )
            avg_rms_errors[tuple(params.items())] = avg_rms_error
            if avg_rms_error > 1.5:
                break  # Saves time

    print(f"{len(avg_rms_errors)} results calculated")
    param_values = [dict(x) for x in avg_rms_errors.keys()]
    rms_error_values = list(avg_rms_errors.values())
    results_df = pd.concat([pd.DataFrame(param_values), 
                            pd.Series(rms_error_values, name='RMS error')], 
                        axis=1)
    
    # Save results to file
    results_df.to_csv(os.path.join(results_dir, f'{exp_name}-{n_states}-{n_reps}.csv'))
    
    # Plot results
    filename = f"random-walk-true-online-td-lambda-{n_states}-{n_reps}.pdf"
    filepath = os.path.join(plot_dir, filename)
    title = f'True online TD-lambda value error on random walk ({n_states} states)'
    ylim=(0.25, 0.55)  # Same as textbook
    plot_results(results_df, 'lam', y_label='RMS error', 
                 ylim=ylim, title=title, filename=filepath)
