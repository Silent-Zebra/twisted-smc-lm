import numpy as np
import torch

# I think this is a minimal implementation that reproduces the issue. You can see that with a small enough prob, and a small enough sample size, if you just so happen to draw the one thing that has high posterior prob but low prior prob, then you can get an IWAE estimate that overshoots the true value significantly. Of course if this happens consistently that's weird, but it's definitely possible for it to happen. And if there are multiple such sequences in the posterior that have this criteria, it's more likely we may see this behaviour

torch.manual_seed(0)
n_vocab = 5

# sd = 50
# q = torch.tensor(np.random.normal(0, sd, size=n_vocab))
# q_probs = torch.nn.functional.softmax(q)

# sigma = torch.tensor(np.random.normal(0, sd, size=n_vocab))
# sigma_probs = torch.nn.functional.softmax(sigma)
#
# high = 100
# sigma_modifier = torch.randint(0, high, sigma_probs.shape)
# tilde_sigma = sigma_probs + sigma_modifier

q = torch.tensor([0.1,2,3,4,5], dtype=torch.float32)
q_probs = torch.nn.functional.softmax(q)

tilde_sigma = torch.tensor([1000, 10, 5, 2, 1], dtype=torch.float32)

print(q_probs)
print(tilde_sigma)

true_Z = tilde_sigma.sum()
true_log_Z = tilde_sigma.sum().log()


n_samples = 60

q_samples = torch.multinomial(q_probs, num_samples=n_samples, replacement=True)

q_probs_of_samples = q_probs[q_samples]
print(q_samples)
# print(q_probs_of_samples)

tilde_sigma_of_samples = tilde_sigma[q_samples]

f_qs = tilde_sigma_of_samples.log() - q_probs_of_samples.log()

print(f_qs)

f_q_estimate = f_qs.mean()

iwae_lb = f_qs.exp().mean().log()

print(f_q_estimate)
print(iwae_lb)
print(true_log_Z)
