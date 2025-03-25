from transformers import AutoTokenizer, FlaxAutoModelForSequenceClassification, AutoModelForSequenceClassification
import torch

def get_tokenizer_and_rewardModel(rm_type):
    """Get the appropriate tokenizer and reward model based on reward type."""
    if rm_type in ["toxicity_threshold", "exp_beta_toxicity_class_logprob"]:
        model_name = "nicholasKluge/ToxicityModel"
    elif rm_type == "sentiment_threshold":
        model_name = "m-aamir95/finetuning-sentiment-classification-model-with-amazon-appliances-data"
    elif rm_type in ["exp_beta_sentiment_class_logprob", "sent_cond_twist"]:
        model_name = "LiYuan/amazon-review-sentiment-analysis"
    elif rm_type in ["toy_rlhf"]:
        model_name = "OpenAssistant/reward-model-deberta-v3-base"
    else:
        return None, None # e.g. for stuff like infilling where you don't need a separate reward model

    tokenizer_RM = AutoTokenizer.from_pretrained(model_name)
    if rm_type in ["toy_rlhf"]:
        rewardModel = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rewardModel = rewardModel.to(device)
    else:
        # Throws a warning message but as far as I can see in my testing, there's no difference 
        # in the outputs under this flax version vs the pytorch original version
        rewardModel = FlaxAutoModelForSequenceClassification.from_pretrained(model_name, from_pt=True) 

    return tokenizer_RM, rewardModel 