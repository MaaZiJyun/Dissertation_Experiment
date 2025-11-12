def compute_reward(total_reward, delay_penalty, energy_penalty, w1=0.5, w2=0.5):
    return total_reward + (w1 * delay_penalty + w2 * energy_penalty)
