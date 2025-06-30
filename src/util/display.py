import matplotlib.pyplot as plt
import imageio.v2 as imageio
import seaborn as sns
import os
import matplotlib
matplotlib.use("TKAgg")


def gen_gif(signal_history: list, action_history: list, state_action_history: list, ep_fn, opt_payoff: float, 
            info_measure, opt_info: float, num_iter: int, record_interval: int, duration: int, output_file: str):
    """Generates a heatmap gif of the whole simulation and saves it into 
    test.gif

    Args:
      num_images (int): the number of images in the gif
      record_interval (int): number of simulations between each image
      duration (int): the duration an image is shown in the gif
    """
    num_images = num_iter // record_interval

    if not os.path.exists("./images"):
        os.mkdir("images")

    ix = []
    epy = []
    optp_y = []
    infoy = []
    opti_y = []
    state_info_y = []
    best_signals_by_im = []

    for i in range(num_images):
        fig, axs = plt.subplots(7, 1, figsize=(10, 12), gridspec_kw={
                                'height_ratios': [2, 2, 2, 1, 1, 2, 2]})
        plt.tight_layout(pad=3)

        step_info_measure, step_best_signals = info_measure(signal_history[i])

        ix.append((i+1)*record_interval)
        epy.append(ep_fn(signal_history[i], action_history[i]))
        optp_y.append(opt_payoff)
        infoy.append(sum(step_info_measure))
        state_info_y.append([step_info_measure])
        best_signals_by_im.append([step_best_signals])
        opti_y.append(opt_info)

        sns.heatmap(signal_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True,
                    fmt=".1f", ax=axs[0])
        axs[0].set_xlabel("states")
        axs[0].set_ylabel("messages")
        axs[0].set_title("Sender\'s weights")

        sns.heatmap(action_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True,
                    fmt=".1f", ax=axs[1])
        axs[1].set_xlabel("actions")
        axs[1].set_ylabel("messages")
        axs[1].set_title("Receiver\'s Signal-Action weights")

        sns.heatmap(state_action_history[i], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True,fmt=".1f", ax=axs[2])
        axs[2].set_xlabel("actions")
        axs[2].set_ylabel("states")
        axs[2].set_title("Receiver\'s State-Action weights")

        sns.heatmap(state_info_y[i], linewidths=0.5, linecolor="white",
                    square=True, cbar=False, annot=True, fmt=".3f", ax=axs[3])
        axs[3].set_title("State info measure")
        axs[3].set_xlabel("States")

        sns.heatmap(best_signals_by_im[i], linewidths=0.5, linecolor="white",
                    square=True, cbar=False, annot=True, ax=axs[4])
        axs[4].set_title("Best signal by IM")
        axs[4].set_xlabel("States")

        axs[5].plot(ix, epy, label="expected")
        axs[5].plot(ix, optp_y, label="optimal")
        axs[5].legend(loc="upper left")
        axs[5].set_xlabel("rollout")
        axs[5].set_ylabel("expected payoff")
        axs[5].set_title("Expected payoff by rollout")

        axs[6].plot(ix, infoy, label="current")
        axs[6].plot(ix, opti_y, label="optimal")
        axs[6].legend(loc="upper left")
        axs[6].set_xlabel("rollout")
        axs[6].set_ylabel("info measure")
        axs[6].set_title("Info measure by rollout")

        fig.suptitle(f"Rollout {(i+1)*record_interval}")
        plt.savefig(f"./images/game_{(i+1)*record_interval}.png")
        plt.close(fig)

    images = []
    for filename in [f"./images/game_{(j+1)*record_interval}.png" for j in range(num_images)]:
        images.append(imageio.imread(filename))

    if not os.path.exists("./simulations"):
        os.mkdir("simulations")

    subfolder = f"{len(signal_history[0][0])}_{len(action_history[0])}_{len(action_history[0][0])}"
    if not os.path.exists(f"./simulations/{subfolder}"):
        os.makedirs(f"simulations/{subfolder}/")

    imageio.mimsave(output_file, images, duration=duration)

# f"./simulations/{self.num_states}_{self.num_signals}_{self.num_actions}/{self.reward_param}{'_null' if self.null_signal else ''}_{num_iter}.gif"


def gen_single_heatmap(signal_history: list, action_history: list, state_action_history: list, ep_fn, opt_payoff: float, 
                       info_measure, opt_info: float, num_iter: int, record_interval: int, duration: int, output_file: str):
    if not os.path.exists("./images"):
        os.mkdir("images")

    ix = []
    epy = []
    optp_y = []
    infoy = []
    opti_y = []
    state_info_y = []
    best_signals_by_im = []

    fig, axs = plt.subplots(7, 1, figsize=(10, 12), gridspec_kw={
                            'height_ratios': [2, 2, 2, 1, 1, 2, 2]})
    plt.tight_layout(pad=3)

    # get info measure stuff
    for i in range(num_iter // record_interval):
        step_info_measure, step_best_signals = info_measure(signal_history[i])
        ix.append((i+1)*record_interval)
        epy.append(ep_fn(signal_history[i], action_history[i]))
        optp_y.append(opt_payoff)
        infoy.append(sum(step_info_measure))
        state_info_y.append([step_info_measure])
        best_signals_by_im.append([step_best_signals])
        opti_y.append(opt_info)

    # draw the graph
    sns.heatmap(signal_history[-1], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True,
                fmt=".1f", ax=axs[0])
    axs[0].set_xlabel("states")
    axs[0].set_ylabel("messages")
    axs[0].set_title("Sender\'s weights")

    sns.heatmap(action_history[-1], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True,
                fmt=".1f", ax=axs[1])
    axs[1].set_xlabel("actions")
    axs[1].set_ylabel("messages")
    axs[1].set_title("Receiver\'s Signal-Action weights")

    sns.heatmap(state_action_history[-1], linewidths=0.5, linecolor="white", square=True, cbar=False, annot=True, fmt=".1f", ax=axs[2])
    axs[2].set_xlabel("actions")
    axs[2].set_ylabel("states")
    axs[2].set_title("Receiver\'s State-Action weights")

    sns.heatmap(state_info_y[-1], linewidths=0.5, linecolor="white",
                square=True, cbar=False, annot=True, fmt=".3f", ax=axs[3])
    axs[3].set_title("State info measure")
    axs[3].set_xlabel("States")

    sns.heatmap(best_signals_by_im[-1], linewidths=0.5,
                linecolor="white", square=True, cbar=False, annot=True, ax=axs[4])
    axs[4].set_title("Best signal by IM")
    axs[4].set_xlabel("States")

    axs[5].plot(ix, epy, label="expected")
    axs[5].plot(ix, optp_y, label="optimal")
    axs[5].legend(loc="upper left")
    axs[5].set_xlabel("rollout")
    axs[5].set_ylabel("expected payoff")
    axs[5].set_title("Expected payoff by rollout")

    axs[6].plot(ix, infoy, label="current")
    axs[6].plot(ix, opti_y, label="optimal")
    axs[6].legend(loc="upper left")
    axs[6].set_xlabel("rollout")
    axs[6].set_ylabel("info measure")
    axs[6].set_title("Info measure by rollout")

    # fig.suptitle(f"Final Strategy")
    plt.savefig(output_file)
    plt.close(fig)

    return
