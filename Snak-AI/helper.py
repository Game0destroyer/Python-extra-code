import matplotlib.pyplot as plt



def plot(scores, mean_scores):
    plt.ion()  # Enable interactive mode
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.legend()
    # Move the window only the first time
    if not hasattr(plot, '_moved'):
        try:
            mng = plt.get_current_fig_manager()
            mng.window.wm_geometry("+100+100")
            plot._moved = True
        except Exception:
            pass
    plt.show()
    plt.pause(0.05)
