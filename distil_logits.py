import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import yaml
import wandb
import argparse
from transformers import TrainerCallback
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--student", type=str, default=None)
    parser.add_argument("--teacher", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--distill_type", type=str, default="shiq_kd")
    parser.add_argument("--num_train_epochs", type=int, default=None)
    

    return parser.parse_args()


# Configuration
config = {
    "project_name": "shiq_kd",
    "dataset": {
        "name": "openai/gsm8k",#"mlabonne/FineTome-100k",
        "config": "main", 
        "split": "train",
        # "num_samples": , # You can pass a number here to limit the number of samples to use.
        "seed": 42
    },
    "models": {
        "teacher": "Qwen/Qwen2.5-Math-1.5B",#"Qwen/Qwen2.5-0.5B", #"Qwen/Qwen2.5-0.5B",#"Qwen/Qwen2.5-0.5B",# "Qwen/Qwen2.5-Math-1.5B",
        "student": "Qwen/Qwen2.5-0.5B",#"Qwen/Qwen2.5-0.5B"
    },
    "tokenizer": {
        "max_length": 4096,
        "chat_template": "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    },
    "training": {
        "output_dir": "./results",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "save_steps": 1000,
        "logging_steps": 1,
        "learning_rate": 1e-6,#2e-5,
        "weight_decay": 0.05,
        "warmup_ratio": 0.1,
        "lr_scheduler_type": "cosine",
        "resume_from_checkpoint": None,  # Set to a path or True to resume from the latest checkpoint
        "fp16": False,
        "bf16": True,
        "report_to": 'wandb',
        # ------------ Evaluations
        "eval_strategy": "steps",
        "do_eval": True,
        "prediction_loss_only": False,
        "eval_steps":100,
    },
    "distillation": {
        "temperature": 2.0,
        "alpha": 0.5
    },
    "model_config": {
        "use_flash_attention": True
    }
}

# Set up environment
os.environ['WANDB_PROJECT'] = config["project_name"]

# Load and preprocess dataset
dataset = load_dataset(config["dataset"]["name"],config["dataset"]["config"], split=config["dataset"]["split"])
dataset = dataset.shuffle(seed=config["dataset"]["seed"])
if "num_samples" in config["dataset"]:
    dataset = dataset.select(range(config["dataset"]["num_samples"]))

# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(config["models"]["teacher"])
student_tokenizer = AutoTokenizer.from_pretrained(config["models"]["student"])

# Apply chat template to student tokenizer
student_tokenizer.chat_template = config["tokenizer"]["chat_template"]

def sharegpt_format(example):
    q = example["question"].strip()
    a = example["answer"].strip()
    return {
        "text": f"Question: {q}\nAnswer: {a}"
    }

# Preprocess and tokenize the dataset
print("Preprocessing and tokenizing dataset...")
original_columns = dataset.column_names
dataset = dataset.map(sharegpt_format, remove_columns=original_columns)

def tokenize_function(examples):
    return student_tokenizer(examples["text"], truncation=True, max_length=config["tokenizer"]["max_length"], padding="max_length")

# Optionally freeze layers of the student model based on spectrum configuration
if "spectrum" in config and "layers_to_unfreeze" in config["spectrum"]:
    def freeze_student_spectrum(model, unfrozen_layers_file):
        with open(unfrozen_layers_file, 'r') as file:
            unfrozen_layers = yaml.safe_load(file)['unfrozen_parameters']
        
        for name, param in model.named_parameters():
            if not any(layer in name for layer in unfrozen_layers):
                param.requires_grad = False
            else:
                param.requires_grad = True

    # Apply freezing to student model
    freeze_student_spectrum(student_model, config["spectrum"]["layers_to_unfreeze"])
else:
    print("Spectrum configuration not found. All layers of the student model will be trainable.")

def pad_logits(student_logits, teacher_logits):
    student_size, teacher_size = student_logits.size(-1), teacher_logits.size(-1)
    if student_size != teacher_size:
        pad_size = abs(student_size - teacher_size)
        pad_tensor = torch.zeros((*teacher_logits.shape[:-1], pad_size), dtype=teacher_logits.dtype, device=teacher_logits.device)
        return (torch.cat([student_logits, pad_tensor], dim=-1), teacher_logits) if student_size < teacher_size else (student_logits, torch.cat([teacher_logits, pad_tensor], dim=-1))
    return student_logits, teacher_logits

class LogitsTrainer(SFTTrainer):
    def __init__(self, distillation_loss='standard',beta=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_loss = distillation_loss
        self._beta = beta
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        self.teacher_model = self.teacher_model.to(device)
        
        student_model = model.module if hasattr(model, 'module') else model

        if not student_model.training:
            # ----------- If not training, directly track CE loss
            student_outputs = student_model(**inputs)
            loss = student_outputs.loss
            self.log({
                "eval/loss":loss.item(),
            })            
            return (loss, student_outputs) if return_outputs else loss
        else:
            teacher_model = self.teacher_model.module if hasattr(self.teacher_model, 'module') else self.teacher_model
            student_outputs = student_model(**inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)

            if self.distillation_loss=='standard':
                custom_loss = self.standard_KD_loss(model, student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
            elif self.distillation_loss=='shiq_kd':
                custom_loss = self.shiq_KD_loss(model, student_outputs.logits, teacher_outputs.logits, inputs, student_outputs.loss)
            return (custom_loss, student_outputs) if return_outputs else custom_loss

    def shiq_KD_loss(self, model, student_logits, teacher_logits, inputs, original_loss):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        student_logits_scaled = student_logits / config["distillation"]["temperature"]  # [B, L, V]
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]  # [B, L, V]
        labels = inputs['labels'] # [B, L]
        labels_unsqueezed = labels.unsqueeze(-1)

        # ---------- Value gap in ShiQ_KD
        s_select_logits = torch.gather(student_logits_scaled, dim=-1, index=labels_unsqueezed).squeeze(-1)
        s_select_logp = torch.gather(F.log_softmax(student_logits_scaled, dim=-1), dim=-1, index=labels_unsqueezed).squeeze(-1)
        s_value_function = s_select_logits - s_select_logp  #[B, L]

        t_select_logits = torch.gather(teacher_logits_scaled, dim=-1, index=labels_unsqueezed).squeeze(-1)
        t_select_logp = torch.gather(F.log_softmax(teacher_logits_scaled, dim=-1), dim=-1, index=labels_unsqueezed).squeeze(-1)
        t_value_function = t_select_logits - t_select_logp

        diff_value_function  = s_value_function - t_value_function  

        # --------- Accumulated logp gap in ShiQ_KD
        log_gap = s_select_logp - t_select_logp
        mask_log_gap = inputs['attention_mask'] * log_gap
        ratio_inverse_cum_lprob = torch.cumsum(mask_log_gap.flip(dims=[-1]),dim=-1).flip(dims=[-1])* inputs['attention_mask']  # [B,L]

        # --------- Final loss of ShiQ_KD
        token_level_loss = (- self._beta*(ratio_inverse_cum_lprob + diff_value_function))**2 # [B,L]
        token_level_loss = (token_level_loss * inputs['attention_mask']).sum(dim=-1)
        normalization = torch.clamp(inputs['attention_mask'].sum(), min=1)
        avg_loss = (token_level_loss / normalization).mean()

        
        self.log({
            "train/loss_cumsum":(ratio_inverse_cum_lprob*inputs['attention_mask']).sum(dim=-1).item(),
            "train/loss_diffvalue": (diff_value_function*inputs['attention_mask']).sum(dim=-1).item()
        })

        return avg_loss
    
    def standard_KD_loss(self, model, student_logits, teacher_logits, inputs, original_loss):
        device = next(model.parameters()).device
        student_logits, teacher_logits = pad_logits(student_logits.to(device), teacher_logits.to(device))
        
        student_logits_scaled = student_logits / config["distillation"]["temperature"] 
        teacher_logits_scaled = teacher_logits / config["distillation"]["temperature"]

        loss_kd = F.kl_div(
            F.log_softmax(student_logits_scaled, dim=-1),
            F.softmax(teacher_logits_scaled, dim=-1),
            reduction='batchmean'
        ) * (config["distillation"]["temperature"] ** 2) / config["tokenizer"]["max_length"]

        return config["distillation"]["alpha"] * loss_kd + (1 - config["distillation"]["alpha"]) * original_loss
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # 先跑原生 evaluate（得到 eval_loss 等）
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # 多卡只让 rank0 做外部评测 + 记录
        if hasattr(self.state, "is_world_process_zero") and (not self.state.is_world_process_zero):
            return metrics

        # 这里跑 AMC acc（注意：用 self.model，不要用 trainer 变量）
        amc_acc = evaluate_on_amc(self.model, student_tokenizer, max_samples=None, max_new_tokens=256)
        metrics[f"{metric_key_prefix}_amc_acc"] = float(amc_acc)

        gsm_acc = evaluate_on_gsm8k(self.model, student_tokenizer, split="test", max_samples=50, max_new_tokens=256)
        metrics[f"{metric_key_prefix}_gsm8k_acc"] = float(gsm_acc)
        # 这句会进入 logging pipeline（wandb 会收）
        self.log(metrics)
        return metrics



# def compute_metrics(_):
#     acc = evaluate_on_amc(trainer.model, student_tokenizer)
#     return {"amc_acc": acc}

# def evaluate_on_amc(model, tokenizer=None, max_samples=None):
#     # unwrap DDP / FSDP
#     if hasattr(model, "module"):
#         model = model.module

#     device = next(model.parameters()).device
#     model.eval()

#     dataset = load_dataset("math-ai/amc23")["test"]
#     if max_samples:
#         dataset = dataset.select(range(max_samples))

#     correct = 0
#     for item in dataset:
#         q = item["question"]
#         a = item["answer"]

#         inputs = tokenizer(q, return_tensors="pt").to(device)
#         with torch.no_grad():
#             outputs = model.generate(**inputs, max_new_tokens=64)

#         pred = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#         if pred == a:
#             correct += 1

#     return correct / len(dataset)


# class AMCEvalCallback(TrainerCallback):
#     def __init__(self, tokenizer, max_samples=None):
#         self.tokenizer = tokenizer
#         self.max_samples = max_samples

#     def on_evaluate(self, args, state, control, model=None, **kwargs):
#         acc = evaluate_on_amc(model, self.tokenizer, max_samples=self.max_samples)
#         print("aaaaaaaaaaaaaaaaaaaaaaaaa")
#         wandb.log({"amc_acc": float(acc)}, step=state.global_step)


# ======================= Wandb setting

args = parse_args()
if args.student is not None:
    config["models"]["student"] = args.student
if args.teacher is not None:
    config["models"]["teacher"] = args.teacher
if args.learning_rate is not None:
    config["training"]["learning_rate"] = args.learning_rate
if args.num_train_epochs is not None:
    config["training"]["num_train_epochs"] = args.num_train_epochs

student_name = config["models"]["student"].split('-')[-1]
teacher_name = config["models"]["teacher"].split('-')[-1]

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=8, remove_columns=["text"])
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)


# Load models with configurable flash attention
model_kwargs = {"torch_dtype": torch.bfloat16}
if config["model_config"]["use_flash_attention"]:
    model_kwargs["attn_implementation"] = "flash_attention_2"

teacher_model = AutoModelForCausalLM.from_pretrained(config["models"]["teacher"], **model_kwargs)
student_model = AutoModelForCausalLM.from_pretrained(config["models"]["student"], **model_kwargs)

name = "%s_%s_to_%s"%(args.distill_type, teacher_name, student_name )
wandb.login()
wandb.init(
    project='shiq_kd',
    name=name,
    config=config
)


# Training arguments
training_arguments = TrainingArguments(**config["training"])
tokenizer = student_tokenizer
# Create the custom SFT Trainer
trainer = LogitsTrainer(
    model=student_model,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    args=training_arguments,
    distillation_loss = args.distill_type
    #compute_metrics=compute_metrics,
    #max_seq_length=config["tokenizer"]["max_length"],
    #dataset_text_field="text",
)
# trainer.add_callback(AMCEvalCallback(student_tokenizer, max_samples=5))

# Add the teacher model to the trainer
trainer.teacher_model = teacher_model
trainer.evaluate()


# Train the model
trainer.train(resume_from_checkpoint=config["training"]["resume_from_checkpoint"])

# Save the final model
trainer.save_model(config["training"]["output_dir"])


