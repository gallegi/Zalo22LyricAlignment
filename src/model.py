import copy
import torch

from difflib import SequenceMatcher

def predict(processor, model, wav, sampling_rate=16000, expand_time=100, device='cpu'):
    input_values = processor(wav, sampling_rate=sampling_rate, return_tensors="pt").input_values
    with torch.no_grad():
        logits = model(input_values.to(device)).logits
    pred_ids = torch.argmax(logits, dim=-1)
    pred_transcript = processor.batch_decode(pred_ids)[0]
    # print(f"transcript: {pred_transcript}")

    time_offset = model.config.inputs_to_logits_ratio / sampling_rate

    outputs = processor.tokenizer.decode(pred_ids[0], output_word_offsets=True)
    lookup = [
        {
            "word": d["word"],
            "start_time": int(round(d["start_offset"] * time_offset * 1000, 2)) - expand_time,
            "end_time": int(round(d["end_offset"] * time_offset * 1000, 2)) + expand_time,
    }
        for d in outputs.word_offsets
    ]
    
    return lookup


def is_capital(word):
    if word[0].isupper():
        return True
    return False

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def search_for_closest(word, lookup, ind, k_nearest=3):
    if len(lookup) == 0:
        return None, 0, 0
    matched_word = None
    best_sim_score = -1
    start = 0
    end = lookup[-1]['end_time']
    for j in range(ind-k_nearest, ind+k_nearest+1):
        if j < 0 or j >= len(lookup):
            continue
        pred_word = lookup[j]['word']
        sim_score = similar(word.lower(), pred_word)
        # print(pred_word, sim_score)
        if sim_score > best_sim_score:
            matched_word = pred_word
            best_sim_score = sim_score
            start = lookup[j]['start_time']
            end = lookup[j]['end_time']
    return matched_word, start, end

def align(lyric, lookup):
    struct = []

    sentence = {'l':[], 's':0, 'e':0}

    update_start = True

    for ind, word in enumerate(lyric):
        matched_word, start, end = search_for_closest(word, lookup, ind)
        # print(word, matched_word, start, end)

        if update_start:
            sentence['s'] = start
            update_start = False

        sentence['l'].append({'d':word, 's':start, 'e':end})
        sentence['e'] = end

        if ind == len(lyric)-1:
            struct.append(copy.deepcopy(sentence))
            sentence = {'l':[], 's':0, 'e':0}
            update_start = True
        elif is_capital(lyric[ind+1]): # end of current sentence
            # sentence['l'].append({'d':word, 's':start, 'e':end})
            # sentence['e'] += end
            struct.append(copy.deepcopy(sentence))
            sentence = {'l':[], 's':0, 'e':0}
            update_start = True
    return struct