import jax.numpy as jnp
import jax

from custom_transformer_prob_utils import smc_procedure, evaluate_log_p_theta_1_to_t, calculate_kl_term, calculate_entropy_gradient_term
from custom_transformer import transformer, stochastic_transformer_sample


def rl_loss(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist,
                rew_model, output_len, n_samples, prompt_len, cfg_baseline, params_baseline,
                cfg_p_0, params_p_0, beta_kl, beta_ent, analytic_sigma_sample, n_vocab):

    sk, sk2, sk3 = jax.random.split(sk, 3)

    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                    cfg_p, params_p,
                                                    cfg_twist,
                                                    params_twist,
                                                    final_twist,
                                                    output_len,
                                                    n_samples,
                                                      analytic_sigma_sample=analytic_sigma_sample, n_vocab=n_vocab)

    # r_seqs = evaluate_log_phi_final(prompt_w_sigma_sample_s_1_to_t,
    #                                   rew_model)
    r_seqs = rew_model(prompt_w_sigma_sample_s_1_to_t, prompt_len)

    # print(prompt_w_sigma_sample_s_1_to_t)
    # print(r_seqs)

    log_p_theta_full_seq = evaluate_log_p_theta_1_to_t(
        prompt_w_sigma_sample_s_1_to_t, cfg_p, params_p, prompt_len,
        output_len)

    # print(log_p_theta_full_seq)

    baseline = transformer(cfg_baseline, params_baseline, prompt)[-1].squeeze()
    baseline_no_grad = jax.lax.stop_gradient(baseline)
    # print("Baseline value (Custom)")
    # print(jax.lax.stop_gradient(baseline))

    # Use baseline_no_grad here because we don't want the gradient for the baseline to flow through the model reward loss
    first_term = ((r_seqs - baseline_no_grad) * log_p_theta_full_seq).mean()  # Use empirical mean as estimate of the expectation
    second_term = log_p_theta_full_seq.mean() * (r_seqs - baseline_no_grad).mean()

    objective = first_term - second_term

    model_seqs = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_samples)
    p_0_seqs = stochastic_transformer_sample(sk3, cfg_p_0, params_p_0, prompt, output_len, n_samples)
    kl_term = calculate_kl_term(p_0_seqs, cfg_p, params_p, prompt_len, output_len)
    ent_term = calculate_entropy_gradient_term(model_seqs, cfg_p, params_p, prompt_len, output_len)
    loss = -objective + beta_kl * kl_term - beta_ent * ent_term # - on entropy because the loss is the negative of objective. Regularization objective is to increase entropy, so negative entropy goes into the loss

    # Baseline term; use empirical mean of r_seqs drawn from sigma, to approximate E_sigma[r(s)]
    # Then MSE loss: (baseline - r_seqs.mean()) ^ 2
    # This term is only used for training the baseline
    baseline_loss = (baseline - r_seqs.mean()) ** 2
    return loss + baseline_loss


def rl_loss_custom_baselinep(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist,
                rew_model, output_len, n_samples, prompt_len, cfg_baseline, params_baseline,
                cfg_p_0, params_p_0, beta_kl, beta_ent, analytic_sigma_sample, n_vocab):
    sk, sk2, sk3 = jax.random.split(sk, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                    cfg_p, params_p,
                                                    cfg_twist,
                                                    params_twist,
                                                    final_twist,
                                                    output_len,
                                                    n_samples,
                                                      analytic_sigma_sample=analytic_sigma_sample, n_vocab=n_vocab)

    r_seqs = rew_model(prompt_w_sigma_sample_s_1_to_t, prompt_len)

    log_p_theta_full_seq = evaluate_log_p_theta_1_to_t(
        prompt_w_sigma_sample_s_1_to_t, cfg_p, params_p, prompt_len,
        output_len)

    baseline = transformer(cfg_baseline, params_baseline, prompt)[-1].squeeze()
    baseline_no_grad = jax.lax.stop_gradient(baseline)
    print("Baseline value (Custom)")
    print(jax.lax.stop_gradient(baseline))

    # Use baseline_no_grad here because we don't want the gradient for the baseline to flow through the model reward loss
    first_term = ((r_seqs - baseline_no_grad) * log_p_theta_full_seq).mean()  # Use empirical mean as estimate of the expectation

    objective = first_term

    # Baseline term; use empirical mean of r_seqs drawn from p, to approximate E_p[r(s)]
    # Then MSE loss: (baseline - r_seqs.mean()) ^ 2
    # This term is only used for training the baseline
    model_seqs = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_samples)
    r_seqs_model = rew_model(model_seqs, prompt_len)
    baseline_loss = (baseline - r_seqs_model.mean()) ** 2

    p_0_seqs = stochastic_transformer_sample(sk3, cfg_p_0, params_p_0, prompt, output_len, n_samples)
    kl_term = calculate_kl_term(p_0_seqs, cfg_p, params_p, cfg_p_0, params_p_0, prompt_len, output_len)
    ent_term = calculate_entropy_gradient_term(model_seqs, cfg_p, params_p,
                                               prompt_len, output_len)
    loss = -objective + beta_kl * kl_term - beta_ent * ent_term # - on entropy because the loss is the negative of objective. Regularization objective is to increase entropy, so negative entropy goes into the loss

    return loss + baseline_loss



# TODO JUL 26 do a mix of maybe half adversarial and half regular sample. Well no, you don't need that then. You can just alternate (or do simultaneous!) steps of regular RL
# with the custom baselinep adv training scheme where both use the regular RL baseline value.
def rl_loss_custom_mixed_sampling(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist,
                rew_model, output_len, n_samples, prompt_len, cfg_baseline, params_baseline,
                cfg_p_0, params_p_0, beta_kl, beta_ent, analytic_sigma_sample, n_vocab):
    sk, sk2, sk3 = jax.random.split(sk, 3)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                    cfg_p, params_p,
                                                    cfg_twist,
                                                    params_twist,
                                                    final_twist,
                                                    output_len,
                                                    n_samples,
                                                      analytic_sigma_sample=analytic_sigma_sample, n_vocab=n_vocab)

    r_seqs_adv = rew_model(prompt_w_sigma_sample_s_1_to_t, prompt_len)

    log_p_theta_adv_full_seq = evaluate_log_p_theta_1_to_t(
        prompt_w_sigma_sample_s_1_to_t, cfg_p, params_p, prompt_len,
        output_len)

    baseline = transformer(cfg_baseline, params_baseline, prompt)[-1].squeeze()
    baseline_no_grad = jax.lax.stop_gradient(baseline)
    print("Baseline value (Custom)")
    print(jax.lax.stop_gradient(baseline))

    # Use baseline_no_grad here because we don't want the gradient for the baseline to flow through the model reward loss
    adv_rl_term = ((r_seqs_adv - baseline_no_grad) * log_p_theta_adv_full_seq).mean()

    # Baseline term; use empirical mean of r_seqs drawn from p, to approximate E_p[r(s)]
    # Then MSE loss: (baseline - r_seqs.mean()) ^ 2
    # This term is only used for training the baseline
    model_seqs = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_samples)
    r_seqs_model = rew_model(model_seqs, prompt_len)
    baseline_loss = (baseline - r_seqs_model.mean()) ** 2

    p_0_seqs = stochastic_transformer_sample(sk3, cfg_p_0, params_p_0, prompt, output_len, n_samples)
    kl_term = calculate_kl_term(p_0_seqs, cfg_p, params_p, prompt_len, output_len)
    ent_term = calculate_entropy_gradient_term(model_seqs, cfg_p, params_p,
                                               prompt_len, output_len)

    log_p_theta_standard_full_seq = evaluate_log_p_theta_1_to_t(
        model_seqs, cfg_p, params_p, prompt_len, output_len)

    # We can use the same baseline here as above if it's per prompt, and not per token
    standard_rl_term = ((r_seqs_model - baseline_no_grad) * log_p_theta_standard_full_seq).mean()

    objective = adv_rl_term + standard_rl_term

    loss = -objective + beta_kl * kl_term - beta_ent * ent_term # - on entropy because the loss is the negative of objective. Regularization objective is to increase entropy, so negative entropy goes into the loss

    return loss + baseline_loss

def rl_loss_custom_extremes(sk, prompt, cfg_p, params_p, cfg_twist, params_twist, final_twist,
                rew_model, output_len, n_samples, prompt_len, cfg_baseline, params_baseline,
                cfg_p_0, params_p_0, beta_kl, beta_ent, cfg_twist_pos,
                            params_twist_pos, final_twist_pos, analytic_sigma_sample, n_vocab):
    sk, sk_pos, sk2, sk3 = jax.random.split(sk, 4)
    _, prompt_w_sigma_sample_s_1_to_t = smc_procedure(sk, prompt,
                                                    cfg_p, params_p,
                                                    cfg_twist,
                                                    params_twist,
                                                    final_twist,
                                                    output_len,
                                                    n_samples // 2,
                                                      analytic_sigma_sample=analytic_sigma_sample, n_vocab=n_vocab)
    _, prompt_w_sigma_pos_sample_s_1_to_t = smc_procedure(sk_pos, prompt,
                                                    cfg_p, params_p,
                                                    cfg_twist_pos,
                                                    params_twist_pos,
                                                    final_twist_pos,
                                                    output_len,
                                                    n_samples // 2,
                                                      analytic_sigma_sample=analytic_sigma_sample, n_vocab=n_vocab)

    # for sample in prompt_w_sigma_pos_sample_s_1_to_t[:10, prompt_len:]:
    #     print(indices_to_tokens(ordered_token_list, sample))
    # print(prompt_w_sigma_pos_sample_s_1_to_t.shape)

    prompt_w_combined_samples_s_1_to_t = jnp.concatenate((prompt_w_sigma_sample_s_1_to_t, prompt_w_sigma_pos_sample_s_1_to_t)) # WHICH AXIS?


    r_seqs_extremes = rew_model(prompt_w_combined_samples_s_1_to_t, prompt_len)

    log_p_theta_extremes_full_seq = evaluate_log_p_theta_1_to_t(
        prompt_w_combined_samples_s_1_to_t, cfg_p, params_p, prompt_len,
        output_len)

    baseline = transformer(cfg_baseline, params_baseline, prompt)[-1].squeeze()
    baseline_no_grad = jax.lax.stop_gradient(baseline)
    print("Baseline value (Custom)")
    print(jax.lax.stop_gradient(baseline))

    # Use baseline_no_grad here because we don't want the gradient for the baseline to flow through the model reward loss
    rl_term = ((r_seqs_extremes - baseline_no_grad) * log_p_theta_extremes_full_seq).mean()

    baseline_loss = (baseline - r_seqs_extremes.mean()) ** 2

    # TODO: consider baseline from p?
    # # Baseline term; use empirical mean of r_seqs drawn from p, to approximate E_p[r(s)]
    # # Then MSE loss: (baseline - r_seqs.mean()) ^ 2
    # # This term is only used for training the baseline
    # model_seqs = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_samples)
    # r_seqs_model = rew_model(model_seqs, prompt_len)
    # baseline_loss = (baseline - r_seqs_model.mean()) ** 2

    p_0_seqs = stochastic_transformer_sample(sk3, cfg_p_0, params_p_0, prompt, output_len, n_samples)
    kl_term = calculate_kl_term(p_0_seqs, cfg_p, params_p, prompt_len, output_len)
    model_seqs = stochastic_transformer_sample(sk2, cfg_p, params_p, prompt, output_len, n_samples)
    ent_term = calculate_entropy_gradient_term(model_seqs, cfg_p, params_p,
                                               prompt_len, output_len)

    loss = -rl_term + beta_kl * kl_term - beta_ent * ent_term # - on entropy because the loss is the negative of objective. Regularization objective is to increase entropy, so negative entropy goes into the loss

    return loss + baseline_loss
