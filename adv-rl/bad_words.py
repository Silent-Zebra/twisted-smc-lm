import jax.numpy as jnp

import jax

from custom_transformer_prob_utils import *


bad_word_list = ["\u0120fuck", "\u0120fucking", "\u0120fucked", "\u0120Fuck", "\u0120FUCK", "fuck", "Fuck", "\u0120fuckin",
                  "\u0120bitch", "\u0120cunt", "\u0120shit", "shit", "\u0120bullshit", "\u0120shitty", "\u0120Shit",
                  "\u0120bastard", "\u0120Bastard", "\u0120pussy", "\u0120asshole", ]

bad_word_indices = jnp.array([5089, 9372, 20654, 25617, 30998, 31699, 34094, 46733,
                     21551, 40267, 7510, 16211, 20041, 32574, 41356,
                     31030, 47209, 18185, 29836 ], dtype=jnp.int32)

def calc_analytic_bad_word_probs(n_vocab, prompt, params_p, huggingface_model, output_len, batch_size=512):
    # ASSUMES OUTPUT LEN 1 (NOT 2) RIGHT NOW
    # Calculates the probability of bad words, for each bad word in bad_word_indices
    # Provides the probability values for sequences that only contain the bad word in the first position (the first token after the prompt)
    # and also does it for all sequences that contain the bad word in only the second position, BUT NOT IN THE FIRST POSITION
    # You can get the probability value for sequences with a bad word in either position by summing two above those up.
    # The first position calculation is easy: simply take the logits from one single seq passed in.
    # The second calculation is harder. 50k+ sequences takes too much memory. So instead I break it into batches.
    # Again, we index into the logits of the bad words we care about.

    print("Calculating analytic probs of bad words (up to 2 output len)")
    if output_len > 2:
        raise NotImplementedError

    prompt_len = prompt.shape[-1]
    # print(prompt.shape)

    if len(prompt.shape) == 1:
        prompt = prompt.reshape(1, -1)

    # print(prompt.shape)

    log_p_all_tokens = get_log_p_all_tokens(prompt, params_p, huggingface_model)
    # TODO CHECK SHAPE log_p_all_tokens has shape (batch, seq_len, n_vocab)
    # print(log_p_all_tokens.shape)

    log_p_of_interest = log_p_all_tokens[:, -1, :].squeeze() # Just the very last time step (before the first generated token)
    # print(log_p_of_interest.shape)

    log_p_select_tokens = log_p_of_interest[bad_word_indices]

    total_bad_word_log_p_t_0 = jax.nn.logsumexp(log_p_select_tokens)

    total_p_bad_t_1_but_not_t_0 = None
    total_prob_bad_by_word = None

    if output_len == 2:
        highest_log_bad_word_prob_at_t_1 = -jnp.inf

        n_bad_words = len(bad_word_indices)

        batch_prompt = jnp.full((n_vocab - n_bad_words, prompt_len), prompt)

        # Do this so that you don't double count - only count the sequences that don't have a bad token in the first position
        tokens_excluding_bad = jnp.setdiff1d(jnp.arange(n_vocab), bad_word_indices)
        # print(tokens_excluding_bad.shape)

        full_seq = jnp.concatenate((batch_prompt, tokens_excluding_bad[:, None]), axis=1)
        # print(full_seq)
        # print(full_seq.shape)

        log_p_bad_tokens_t_1_but_not_t_0 = jnp.zeros((n_bad_words,))

        # Break up evaluation into batches to avoid running out of memory
        for i in range(n_vocab // batch_size + 1):
            batch_to_inspect = full_seq[i * batch_size:(i+1) * batch_size]
            # print(batch_to_inspect.shape)
            # output_unnormalized_batch = trainstate_p.apply_fn(
            #     input_ids=batch_to_inspect, params=trainstate_p.params,
            #     train=False)
            # log_p_all_tokens = jax.nn.log_softmax(output_unnormalized_batch,
            #                                       axis=-1)
            log_p_all_tokens = get_log_p_all_tokens(batch_to_inspect, params_p, huggingface_model)
            log_p_t_1_all = log_p_all_tokens[:, -1, :].squeeze()
            # print(log_p_of_interest)
            # print(log_p_of_interest.shape)
            log_p_t_1_select_tokens = log_p_t_1_all[:, bad_word_indices]
            # print(log_p_select_tokens)
            # print(log_p_select_tokens.shape)
            # print(log_p_of_interest[0, 5089])

            log_p_t_0 = evaluate_log_p_selected_tokens(batch_to_inspect, prompt_len, params_p, huggingface_model)
            # print(log_p_t_1_select_tokens)
            # print(log_p_t_1_select_tokens.shape)
            # print(log_p_t_0)
            # print(log_p_t_0.shape)

            # rng_key, dropout_rng = jax.random.split(rng_key)
            # log_p_t_0 = jax.nn.log_softmax(output_unnormalized_batch[:,-2,:])[jnp.arange(batch_to_inspect.shape[0]), batch_to_inspect[:,-1]]
            # print(log_p_t_0)
            # print(log_p_t_0.shape)

            log_p_t_0_to_1 = log_p_t_0 + log_p_t_1_select_tokens
            # print(log_p_t_0_to_1)
            # print(log_p_t_0_to_1.shape)

            # print(jnp.exp(log_p_t_0_to_1).sum(axis=0))
            # print(jnp.exp(jax.nn.logsumexp(log_p_t_0_to_1, axis=0)))

            log_p_bad_tokens_t_1_but_not_t_0 += jax.nn.logsumexp(log_p_t_0_to_1, axis=0)

            # print("prob of all sequences not containing a bad word in the first time step but containing a bad word in the second time step (by bad word)")
            # print(p_bad_tokens_t_1_but_not_t_0)


    print("Prob of bad words at t_0 by bad word")
    total_prob_bad_t_0_by_word = jnp.exp(log_p_select_tokens)
    print(total_prob_bad_t_0_by_word)

    print("Total prob of bad words at t_0")
    total_prob_bad_t_0 = jnp.exp(total_bad_word_log_p_t_0)
    print(total_prob_bad_t_0)
    total_log_prob_bad = total_bad_word_log_p_t_0

    if output_len == 2:

        print("Prob of bad words at t_1 (no bad word at t_0) by bad word")
        print(jnp.exp(log_p_bad_tokens_t_1_but_not_t_0))

        print("Total prob of bad words at t_1 (but not t_0)")
        total_p_bad_t_1_but_not_t_0 = jnp.exp(log_p_bad_tokens_t_1_but_not_t_0).sum()
        print(total_p_bad_t_1_but_not_t_0)

        print("Total prob of sequence containing a bad word by bad word")
        total_prob_bad_by_word = total_prob_bad_t_0_by_word + jnp.exp(log_p_bad_tokens_t_1_but_not_t_0) # sum of these probs (not log probs) is correct; we are adding up prob of all sequences that have a bad word in the t_0 position, with the prob of all sequences that have no bad word in t_0, but a bad word in the t_1 position. Together this gives us the total marginal probability of the bad word
        print(total_prob_bad_by_word)

        print("Total prob of sequence containing a bad word")
        total_prob_bad = jnp.exp(total_bad_word_log_p_t_0) + total_p_bad_t_1_but_not_t_0
        print(total_prob_bad)

        total_log_prob_bad = jax.nn.logsumexp(jnp.concatenate((total_bad_word_log_p_t_0, total_p_bad_t_1_but_not_t_0)))

        print(total_prob_bad)
        print(total_log_prob_bad)
        print(jnp.exp(total_log_prob_bad))
        1/0


    return total_prob_bad_t_0_by_word, total_prob_bad_t_0, total_p_bad_t_1_but_not_t_0, total_prob_bad_by_word, total_log_prob_bad

