import jax.numpy as jnp

def concatenate_samples(existing_samples, new_samples):
    """Concatenate new samples with existing ones."""
    combined_samples = []
    
    for i in range(len(existing_samples)):
        print("----")
        print(existing_samples[i].shape)
        print(new_samples[i].shape)
        combined = jnp.concatenate((existing_samples[i], new_samples[i]))
        combined_samples.append(combined)
        print(combined.shape)
        
    return combined_samples 