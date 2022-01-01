import json
import argparse
import os
import random
from copy import deepcopy
import logging
import pickle
from tqdm import tqdm, trange
import timeit

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import transformers
from transformers import AutoModelForMaskedLM, AutoConfig, AutoTokenizer
from transformers.trainer_utils import is_main_process

logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self, 
                 input_ids, 
                 token_type_ids, 
                 attention_mask, 
                 labels, 
                 label_indexs):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.label_indexs = label_indexs

def convert_data_to_features(args, tokenizer, context, answer, answer_start):

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token

    context_tokens = tokenizer.tokenize(context)
    answer_tokens = tokenizer.tokenize(answer)

    max_context_length = args.max_seq_length - len(answer_tokens) - args.max_query_length - 4

    if len(context_tokens) > max_context_length:
        if answer_start == -1:
            context_tokens = context_tokens[:max_context_length]
        else:
            context_half_len = int(max_context_length / 2)
            char_num = 0

            for i, context_token in enumerate(context_tokens):
                if '##' in context_token:
                    char_num += len(context_token.replace('##',''))
                else:
                    char_num += len(context_token) + 1

                if context_token == answer_tokens[0] and char_num >= answer_start:
                    answer_token_start = i
                    break

            left_bound = answer_token_start - context_half_len
            right_bound = answer_token_start + context_half_len

            if left_bound < 0:
                context_tokens = context_tokens[:max_context_length]
            elif right_bound > len(context_tokens):
                context_tokens = context_tokens[len(context_tokens) - max_context_length:]
            else:
                context_tokens = context_tokens[answer_token_start - context_half_len:answer_token_start + context_half_len]

    input_tokens = [cls_token] + context_tokens + [sep_token]
    token_type_ids = [0] * len(input_tokens)

    input_tokens += answer_tokens + [sep_token]
    while len(token_type_ids) < len(input_tokens):
        token_type_ids.append(1)

    label_indexs = len(input_tokens) 
    input_tokens += [mask_token]
    token_type_ids.append(0)

    attention_mask = [1] * len(input_tokens)

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    # label_indexs = len(input_ids) - 1

    # Zero-pad up to the sequence length.
    while len(input_ids) < args.max_seq_length:
        input_ids.append(0)
        token_type_ids.append(0)
        attention_mask.append(0)    

    assert len(input_ids) == args.max_seq_length
    assert len(token_type_ids) == args.max_seq_length
    assert len(attention_mask) == args.max_seq_length

    # logger.info("*** data features***")
    # logger.info("tokens: %s" % " ".join(input_tokens))
    # logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    # logger.info(
    #     "segment_ids: %s" % " ".join([str(x) for x in token_type_ids]))                    
    # logger.info(
    #     "input_mask: %s" % " ".join([str(x) for x in attention_mask]))
    # logger.info("label_indexs: %d" ,label_indexs)

    return InputFeatures(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            labels = '',
            label_indexs = label_indexs
            )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def predict(args, model, tokenizer, features, beam_size=1):
    
    sep_token = tokenizer.sep_token
    mask_token = tokenizer.mask_token

    input_ids = torch.tensor([features.input_ids], dtype=torch.long)
    attention_mask = torch.tensor([features.attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([features.token_type_ids], dtype=torch.long)
    
    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)
    token_type_ids = token_type_ids.to(args.device)

    result = []
    model.eval()
    all_candidates = [{'prediction_ids' : [] , 'score' : 0, 'iter' : 0}]
    EOF_flag = tokenizer.convert_tokens_to_ids(sep_token)
    error = False

    with torch.no_grad():
        while(len(result) < beam_size):
            iter_candidates = []
            for i in range(len(all_candidates)):

                seq_input_ids = deepcopy(input_ids)
                seq_token_type_ids = deepcopy(token_type_ids)
                seq_attention_mask = deepcopy(attention_mask)
                label_indexs = features.label_indexs

                if all_candidates[i]['iter'] != 0:
                    for id in all_candidates[i]['prediction_ids']:
                        seq_input_ids[0][label_indexs] = id
                        seq_token_type_ids[0][label_indexs] = 0
                        seq_attention_mask[0][label_indexs] = 1
                        label_indexs += 1

                    seq_input_ids[0][label_indexs] = tokenizer.convert_tokens_to_ids(mask_token)
                    seq_token_type_ids[0][label_indexs] = 0
                    seq_attention_mask[0][label_indexs] = 1   

                # logger.info("*** data features***")
                # logger.info("input_ids: %s" % " ".join([str(x) for x in seq_input_ids]))
                # logger.info(
                #     "segment_ids: %s" % " ".join([str(x) for x in seq_token_type_ids]))                    
                # logger.info(
                #     "input_mask: %s" % " ".join([str(x) for x in seq_attention_mask]))
                # logger.info("label_indexs: %d" ,label_indexs)

                inputs = {
                    "input_ids": seq_input_ids,
                    "attention_mask": seq_attention_mask,
                    "token_type_ids": seq_token_type_ids              
                }

                if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                    del inputs["token_type_ids"]

                # XLNet and XLM use more arguments for their predictions
                if args.model_type in ["xlnet", "xlm"]:
                    inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                    # for lang_id-sensitive xlm models
                    if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                        inputs.update(
                            {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                        )

                outputs = model(**inputs)

                # logit_prob = F.log_softmax(outputs['logits'][0][label_indexs], dim=0).data.tolist()
                # prob_result = {id: prob for id, prob in enumerate(logit_prob)}
                # prob_result = sorted(prob_result.items(), key=lambda x: x[1], reverse=True)

                prob_result = []
                logit_prob = F.log_softmax(outputs['logits'][0][label_indexs], dim=0)

                while(len(prob_result) < beam_size + 1):
                    predicted_id = torch.argmax(logit_prob).item()
                    score = logit_prob[predicted_id].item()
                    logit_prob[predicted_id] = -1000000000
                    prob_result.append((predicted_id, score))
                prob_result = sorted(prob_result, key=lambda x: x[1], reverse=True)

                tmp_candidates = []
                for id, logits in prob_result:
                    if id in all_candidates[i]['prediction_ids'][-1:]:
                        # print('repeat')
                        continue
                    
                    prediction_ids = all_candidates[i]['prediction_ids'] + [id]
                    
                    score = all_candidates[i]['score'] + logits
                    iter = all_candidates[i]['iter'] + 1

                    tmp_candidates.append({'prediction_ids' : prediction_ids, 'score' : score, 'iter' : iter})
                    
                    if len(tmp_candidates) == beam_size:
                        break

                iter_candidates += tmp_candidates

            iter_result = []
            for candidate in sorted(iter_candidates, key=lambda x: x['score'], reverse=True):
                if EOF_flag in candidate['prediction_ids']:
                    result.append(candidate)
                else:
                    iter_result.append(candidate)

                if len(iter_result) == beam_size:
                    break

            if len(result) == beam_size:
                break
                
            all_candidates = iter_result

            if iter == args.max_query_length:
                if len(result) == 0:
                    error = True
                result += all_candidates[:beam_size-len(result)]
                break

    predictions = []
    for candidate in result:
        prediction_tokens = tokenizer.convert_ids_to_tokens(candidate['prediction_ids'])
        prediction_text = tokenizer.convert_tokens_to_string(prediction_tokens).replace(sep_token,'')
        score = candidate['score'] / candidate['iter']
        predictions.append({'prediction_text' : prediction_text, 'score' : score, 'error' : error})

    return sorted(predictions, key=lambda x: x['score'], reverse=True)


def evaluate(args, model, tokenizer, beam_size=1):

    start_time = timeit.default_timer()
    
    """ Load datas """
    with open(args.predict_file, 'rb') as f:
        eval_dataset = json.load(f)
    
    num = 0
    error = []
    gen_question_text = ''
    for index, data in enumerate(tqdm(eval_dataset)):
        try:
            context = data['context']
            answer_text = ''
            result = []
            if 'squad' in args.predict_file:
                for answer in data['answers']:
                    if answer_text == answer['text']:
                        continue
                    answer_text = answer['text']
                    answer_start = answer['answer_start']
                    features = convert_data_to_features(args, tokenizer, context, answer_text, answer_start)
                    result += predict(args, model, tokenizer, features, beam_size)                
            else:
                answer_text = data['answer']
                features = convert_data_to_features(args, tokenizer, context, answer_text, -1)
                result += predict(args, model, tokenizer, features, beam_size)
            
            if len(result) > beam_size:
                result = sorted(result, key=lambda x: x['score'], reverse=True)

            gen_questions = []
            for ele in result:
                gen_questions.append(ele['prediction_text'])

            data['gen_questions'] = gen_questions
            # print(data['gen_questions'])
            # input()
            if len(gen_questions) > 0:
                gen_question_text += gen_questions[0] + '\n'
                num += 1
                if result[0]['error'] == True:
                    error.append(index)
            else:
                gen_question_text += '\n'

        except Exception as e:
            data['gen_questions'] = []
            gen_question_text += '\n'
            # raise e
            continue
        
    evalTime = timeit.default_timer() - start_time

    logger.info("Evaluation done %d in %f secs (%f sec per example)", num, evalTime, evalTime/num)
    logger.info("error_list: %s" % " ".join([str(x) for x in error]))

    if 'dev' in args.predict_file:
        data_type = 'dev'
    elif 'test' in args.predict_file:
        data_type = 'test'
    else:
        data_type = 'eval'

    output_file = args.model_name_or_path+'{0}_beam_size_{1}'.format(str(data_type), str(beam_size))

    json.dump(eval_dataset, open(output_file + '.json','w'))
    with open(output_file + '.txt', 'w') as file:
        file.write(gen_question_text.strip())

    print(args.model_name_or_path)
    print(evalTime)
    with open(args.model_name_or_path+'evalTime' + '.txt', 'w') as file:
        file.write(str(evalTime))

def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type bert",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    # Other parameters
    # parser.add_argument(
    #     "--output_dir",
    #     default=None,
    #     type=str,
    #     help="The output directory where the model checkpoints and predictions will be written.",
    # )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--beam_size", type=int, default=1, help="beam search size"
    )

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    
    if args.predict_file != None:
        evaluate(args, model=model, tokenizer=tokenizer, beam_size=args.beam_size)
    else:
        while(1):
            context = input("context: ")
            answer_text = input("answer: ")
            answer_start = context.find(answer_text)

            features = convert_data_to_features(args, tokenizer, context, answer_text, answer_start)
            result = predict(args, model, tokenizer, features, args.beam_size)
            print(result)

if __name__ == "__main__":
    main()