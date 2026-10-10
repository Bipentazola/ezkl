from pathlib import Path

from transformers import AutoModelForSequenceClassification

MODEL_DIR = Path(__file__).resolve().parent / "BertForSequenceClassification"


def main():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

    total_params = 0
    trainable_params = 0

    print(f"{'Module':<60}{'Shape':<20}{'Params':>12}")
    print("-" * 92)
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        if param.requires_grad:
            trainable_params += numel
        print(f"{name:<60}{str(tuple(param.shape)):<20}{numel:>12,}")

    print("-" * 92)
    print(f"Total parameters    : {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {total_params - trainable_params:,}")
    print(f"Approx. size (fp32) : {total_params * 4 / (1024 ** 2):.2f} MB")


if __name__ == "__main__":
    main()
