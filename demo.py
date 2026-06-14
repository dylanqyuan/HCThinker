import re
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


MODEL_PATH = "weights_HumanCropThinker"
IMAGE_PATH = "example.jpg"

TEXT_PROMPT = (
    "A conversation between a user and an expert in human aesthetics and photography. "
    "The expert first thinks about the reasoning process in mind following the philosophy of "
    "\"enhancing the good while reducing the bad\", and then provides the user with the answer. "
    "The reasoning process and the cropping box are enclosed within <think> </think> and <answer> </answer> tags. "
    "Following \"<think>thinking process</think>\\n<answer>(x1,y1),(x2,y2)</answer>\" format. "
    "x1,y1,x2,y2 are range from 0 to 1000."
)

GEN_KWARGS = {
    "max_new_tokens": 128,
    "do_sample": False,
    "temperature": 0.0,
}

ANSWER_RE = re.compile(
    r"<answer>\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*,\s*\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)\s*</answer>",
    re.IGNORECASE,
)


def parse_answer_box(text):
    match = ANSWER_RE.search(text)
    if match is None:
        return None

    x1, y1, x2, y2 = map(int, match.groups())
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def run_inference(model, processor, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": TEXT_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **GEN_KWARGS)

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    parsed_answer = parse_answer_box(output_text)

    return output_text, parsed_answer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto" if device == "cuda" else None,
    )

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    raw_response, parsed_answer = run_inference(
        model=model,
        processor=processor,
        image_path=IMAGE_PATH,
    )

    print("Image path:")
    print(IMAGE_PATH)

    print("\nRaw response:")
    print(raw_response)

    print("\nParsed answer:")
    print(parsed_answer)


if __name__ == "__main__":
    main()