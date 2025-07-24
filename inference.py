import argparse
import torch
import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
import json
from tqdm import tqdm
import shortuuid
import whisper

from omnicharacter.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from omnicharacter.conversation import conv_templates, SeparatorStyle
from omnicharacter.mm_utils import process_images
from omnicharacter.model.builder import load_pretrained_qwen_model
from omnicharacter.utils import disable_torch_init
from torch.utils.data import Dataset, DataLoader
import time
import math
from PIL import Image
import os
import os.path as osp
from omnicharacter.flow_inference import AudioDecoder
import torchaudio
import copy
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def ctc_postprocess(tokens, blank):
    _toks = tokens.squeeze(0).tolist()
    print(_toks,len(_toks))
    # deduplicated_toks = [v for i, v in enumerate(_toks) if i == 0 or v != _toks[i - 1]]
    deduplicated_toks = [v for i, v in enumerate(_toks)]
    hyp = [v for v in deduplicated_toks if v != blank]
    hyp = " ".join(list(map(str, hyp)))
    return hyp

def get_model():

    disable_torch_init()

    model_path = os.path.expanduser(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_qwen_model(model_path, args.model_base, is_lora=args.is_lora)

    return tokenizer, model, image_processor, context_len

def get_dialog(sample):

    source = sample['conversations'][:-1]

    if source[0]["from"] != "human":
        source = source[1:]

    speech_file = sample["question_audio"][-1]

    message = []

    roles = {"human": "user", "gpt": "assistant"}

    for idx, conv in enumerate(source):

        if idx == (len(source) - 1):
            conv["value"] = "<speech>\n Please answer the questions in the user's input speech"

        role = conv["from"]
        content = conv["value"]

        role = roles.get(role, role)
        
        conv = {"role" : role, "content" : content}

        message.append(conv)

    return message, speech_file

def eval_model(model, tokenizer, image_processor, sample):

    tokenizer.add_tokens(["<image>"], special_tokens=True)
    tokenizer.add_tokens(["<speech>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    speech_token_index = tokenizer.convert_tokens_to_ids("<speech>")

    input_id = []
    
    temp = copy.deepcopy(sample)
    system_message = sample['conversations'][0]['value']

    input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
    # input_id += tokenizer.apply_chat_template([{"role" : "user", "content" : "<speech>\n Please answer the questions in the user's input speech"}])

    message, speech_file = get_dialog(sample)

    encode_id = tokenizer.apply_chat_template(message)
    # print(message)
    input_id += encode_id

    for idx, encode_id in enumerate(input_id):
        if encode_id == image_token_index:
            input_id[idx] = IMAGE_TOKEN_INDEX
        if encode_id == speech_token_index:
            input_id[idx] = SPEECH_TOKEN_INDEX

    input_ids = torch.tensor([input_id], dtype=torch.long)
    input_ids = input_ids.to(device='cuda', non_blocking=True)

    speech_tensor, speech_length = None, None

    if osp.exists(speech_file):
        speech = whisper.load_audio(speech_file)
        if args.input_type == "raw":
            speech = torch.from_numpy(speech)
        elif args.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech_tensor = whisper.log_mel_spectrogram(speech, n_mels=args.mel_size).permute(1, 0)

        speech_length = torch.LongTensor([speech_tensor.shape[0]])
        speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True).unsqueeze(0)
        speech_length = speech_length.to(device='cuda', non_blocking=True)
    else:
        speech_tensor, speech_length = None, None

    image_file="/mnt/workspace/haonan/code/omnicharacter/model/assets/example.png"
    image = Image.open(os.path.join(
        '', image_file)).convert('RGB')
    image_tensor = process_images(
        [image], image_processor, model.config)[0]

    with torch.inference_mode():
        time1 = time.time()
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).half().cuda(),
            image_sizes=[image.size],
            speech=speech_tensor,
            speech_lengths=speech_length,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            faster_infer=False,
            streaming_unit_gen=False,
        )
        time2 = time.time()
        output_ids, output_units = outputs

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if args.s2s:
            if args.speech_generator_type=="ar":
                output_units = output_units
            elif args.speech_generator_type=="ctc":
                output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)

        print(f"H-{time2-time1}-{idx}\t{outputs}")
        if args.s2s:
            print(f"U-{idx}\t{output_units}")

        temp['text_response'] = outputs
        temp['audio_units'] = output_units

        return temp

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/haonan/code/omnicharacter/checkpoints/omnicharacter_stage1_qwen_2.5/checkpoint-604")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_qwen2")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=True)
    parser.add_argument("--speech_generator_type", action="store_true", default="ar")
    parser.add_argument("--is_lora", action="store_true", default=False)

    args = parser.parse_args()

    tokenizer, model, image_processor, context_len = get_model()

    datas = json.load(open("/mnt/workspace/haonan/code/omnicharacter/data_engine/data_source/omnicharacter_test_400.json", "r"))
    
    results = []

    for data in tqdm(datas):
        results.append(eval_model(model, tokenizer, image_processor, data))

    # with open("/mnt/workspace/haonan/code/omnicharacter/eval/omnicharacter_test_400/results/results.json", "w") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)
