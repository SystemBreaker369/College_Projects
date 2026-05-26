import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_dqn_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define the positions of the boxes
    positions = {
        'Initialize Environment': (0.5, 0.9),
        'Initialize Q-Network': (0.5, 0.75),
        'Set Hyperparameters': (0.5, 0.6),
        'Start Training': (0.5, 0.45),
        'For each episode': (0.5, 0.3),
        'Reset Environment': (0.5, 0.2),
        'For each timestep': (0.5, 0.1),
        'Select Action': (0.5, 0.0),
        'Execute Action': (0.5, -0.1),
        'Store Experience': (0.5, -0.2),
        'Sample Batch': (0.5, -0.3),
        'Compute Target Q': (0.5, -0.4),
        'Update Q-Network': (0.5, -0.5),
        'Evaluate Model': (0.5, -0.6),
        'Collect Rewards': (0.5, -0.7),
    }

    # Draw boxes
    for key, (x, y) in positions.items():
        ax.add_patch(patches.Rectangle((x - 0.15, y - 0.05), 0.3, 0.1, fill=True, edgecolor='black', facecolor='lightblue'))
        ax.text(x, y, key, ha='center', va='center', fontsize=10)

    # Draw arrows
    for i in range(len(positions) - 1):
        ax.annotate('', 
                    xy=(0.5, positions[list(positions.keys())[i + 1]][1] + 0.05), 
                    xytext=(0.5, positions[list(positions.keys())[i]][1] - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.8, 1)
    ax.axis('off')
    plt.title('DQN Flowchart', fontsize=14)
    plt.savefig('dqn_flowchart.png', bbox_inches='tight')
    plt.show()

def draw_ppo_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define the positions of the boxes
    positions = {
        'Initialize Environment': (0.5, 0.9),
        'Initialize Policy and Value Network': (0.5, 0.75),
        'Set Hyperparameters': (0.5, 0.6),
        'Start Training': (0.5, 0.45),
        'For each episode': (0.5, 0.3),
        'Reset Environment': (0.5, 0.2),
        'Collect Trajectories': (0.5, 0.1),
        'Compute Advantages': (0.5, 0.0),
        'Update Policy and Value Networks': (0.5, -0.1),
        'Evaluate Model': (0.5, -0.2),
        'Collect Rewards': (0.5, -0.3),
    }

    # Draw boxes
    for key, (x, y) in positions.items():
        ax.add_patch(patches.Rectangle((x - 0.15, y - 0.05), 0.3, 0.1, fill=True, edgecolor='black', facecolor='lightgreen'))
        ax.text(x, y, key, ha='center', va='center', fontsize=10)

    # Draw arrows
    for i in range(len(positions) - 1):
        ax.annotate('', 
                    xy=(0.5, positions[list(positions.keys())[i + 1]][1] + 0.05), 
                    xytext=(0.5, positions[list(positions.keys())[i]][1] - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=1.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 1)
    ax.axis('off')
    plt.title('PPO Flowchart', fontsize=14)
    plt.savefig('ppo_flowchart.png', bbox_inches='tight')
    plt.show()