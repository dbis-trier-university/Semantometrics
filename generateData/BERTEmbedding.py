import torch
import json
from pytorch_pretrained_bert import BertTokenizer, BertModel
import regex as re
from general.baseFileExtractor import get_file_base, get_seminal_u, get_survey_u, get_uninfluential_u

# thanks to https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/


# Load pre-trained model

if torch.cuda.is_available():
    model = BertModel.from_pretrained('bert-base-uncased').cuda()
    model.eval().cuda()
else:
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()


def write_to_file(file, raw, ref_p, cit_p, type_p):
    f = open(file, 'w', encoding='utf8')

    pub_ct = 1
    pubs = []
    for raw_p in raw:
        # write bert feature embedding for seminal, survey and uninfluential data
        bert_p = raw[raw_p]
        entry = {'key': raw_p, 'bert': bert_p}

        bert_ref = []
        bert_cit = []

        # append bert embedding for seminal, survey and uninfluential references & citations
        for a_i in ref_p[raw_p]:
            bert_ref.append(ref_p[raw_p][a_i])
        for a_o in cit_p[raw_p]:
            bert_cit.append(cit_p[raw_p][a_o])

        entry['ref'] = bert_ref
        entry['cit'] = bert_cit

        pubs.append(entry)
        pub_ct += 1

    f.write(json.dumps({type_p: pubs}) + '\n')
    f.close()


def calculate_bert_embedding(indexed_tokens, segments_ids, length):
    # Convert inputs to PyTorch tensors
    if torch.cuda.is_available():
        tokens_tensor = torch.tensor([indexed_tokens], dtype=torch.long).cuda()

        segments_tensors = torch.tensor([segments_ids], dtype=torch.long).cuda()
    else:
        tokens_tensor = torch.tensor([indexed_tokens], dtype=torch.long)

        segments_tensors = torch.tensor([segments_ids], dtype=torch.long)

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # Convert the hidden state embeddings into single token vectors
    # Holds the list of 12 layer embeddings for each token
    # Will have the shape: [# tokens, # layers, # features]
    token_embeddings = []

    # For each token in the sentence...
    for token_i in range(length):

        # Holds 12 layers of hidden states for each token
        hidden_layers = []

        # For each of the 12 layers...
        for layer_i in range(len(encoded_layers)):
            # Lookup the vector for `token_i` in `layer_i`
            vec = encoded_layers[layer_i][0][token_i]

            hidden_layers.append(vec)

        token_embeddings.append(hidden_layers)

    # concatenated last 4 layers of embeddings for every word of sentence(s)
    return [torch.cat((layer[-1], layer[-2], layer[-3], layer[-4]), 0).cpu() for layer in token_embeddings]


def do_bert(string, tokenizer):
    sentences = re.split('\s\s+', string.strip())

    number_of_words = 0

    curr_sentence = 0
    while curr_sentence < len(sentences):
        if len(tokenizer.tokenize(sentences[curr_sentence])) > 300:
            half = len(tokenizer.tokenize(sentences[curr_sentence])) / 2

            new_sentences = []

            # copy all already observed sentences in new list
            for new_sentence in range(0, curr_sentence):
                new_sentences.append(sentences[new_sentence])

            splittable_sentence = sentences[curr_sentence].split(' ')
            first_half = ''
            second_half = ''

            for token in range(0, len(splittable_sentence)):
                if token < half:
                    first_half += ' ' + splittable_sentence[token]
                else:
                    second_half += ' ' + splittable_sentence[token]

            new_sentences.append(first_half)
            new_sentences.append(second_half)

            for new_sentence in range(curr_sentence + 1, len(sentences)):
                new_sentences.append(sentences[new_sentence])

            sentences = new_sentences

            # do not increment curr_sentence to observe it another time
        else:
            curr_sentence += 1

    for sentence in sentences:
        number_of_words += len(tokenizer.tokenize(sentence))

    # embedding vector which is going to hold the embeddings for every word of the document
    embedding = [None] * number_of_words
    curr_word = 0
    curr_sentence_id = 0

    # title of document -> first sentence
    if len(sentences) > 1:
        curr_string = sentences[curr_sentence_id] + ' ' + sentences[curr_sentence_id + 1]
    else:
        curr_string = sentences[curr_sentence_id]

    tokenized_text = tokenizer.tokenize(curr_string)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(tokenized_text)

    # curr sentence is title, so the first part of the observed two sentence construct
    curr_sentence_length = len(tokenizer.tokenize(sentences[curr_sentence_id]))

    for word_index in range(0, curr_sentence_length):
        segments_ids = [0]

    # embed sentence(s)
    concatinated_last_4_layers = calculate_bert_embedding(indexed_tokens, segments_ids, len(tokenized_text))

    # write sentence(s) embedding in embedding vector
    for word in range(0, len(tokenized_text)):
        embedding[curr_word] = concatinated_last_4_layers[word].numpy()
        curr_word += 1

    begin_current_sentence = curr_sentence_length
    curr_word = begin_current_sentence

    curr_sentence_id += 2

    # abstract of document -> all other sentences
    while curr_sentence_id < len(sentences):
        curr_string = sentences[curr_sentence_id - 1] + ' ' + sentences[curr_sentence_id]
        tokenized_text = tokenizer.tokenize(curr_string)

        # curr sentence is part of the abstract, so the second part of the observed two sentence construct
        previous_sentence_length = len(tokenizer.tokenize(sentences[curr_sentence_id - 1]))

        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        for index in range(0, previous_sentence_length):
            segments_ids[index] = 0

        # embed sentence(s)
        concatinated_last_4_layers = calculate_bert_embedding(indexed_tokens, segments_ids, len(tokenized_text))

        # write sentence(s) embedding in embedding vector
        for word in range(0, previous_sentence_length):
            # blending of embeddings
            new_embedding = concatinated_last_4_layers[word].numpy()

            for pos in range(0, len(embedding[curr_word])):
                embedding[curr_word][pos] = (embedding[curr_word][pos] + new_embedding[pos]) / 2

            curr_word += 1

        for word in range(previous_sentence_length, len(tokenized_text)):
            embedding[curr_word] = concatinated_last_4_layers[word].numpy()
            curr_word += 1

        begin_current_sentence += previous_sentence_length
        curr_word = begin_current_sentence
        curr_sentence_id += 1

    # calculate vector representation of embedding of whole document
    sentence_embedding = [None] * 3072

    for x in range(0, 3072):
        val = 0
        for word in range(0, number_of_words):
            val += embedding[word][x]

        sentence_embedding[x] = val / number_of_words

    return sentence_embedding


def make_bert():
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(get_seminal_u(), encoding='latin-1') as s:
        seminal_hlp = json.load(s)
    with open(get_survey_u(), encoding='latin-1') as s:
        survey_hlp = json.load(s)
    with open(get_uninfluential_u(), encoding='latin-1') as s:
        uninfluential_hlp = json.load(s)

    print('sem')

    seminal_p = {}
    seminal_x = {}
    seminal_y = {}

    ct = 0
    for p in seminal_hlp['seminal']:
        seminal_p[ct] = do_bert(p['abs'], tokenizer)

        seminal_x[ct] = {}
        seminal_y[ct] = {}

        ct_ref = 0
        for ref in p['ref']:
            seminal_x[ct][ct_ref] = do_bert(ref['abs'], tokenizer)
            ct_ref += 1

        ct_cit = 0
        for cit in p['cit']:
            seminal_y[ct][ct_cit] = do_bert(cit['abs'], tokenizer)
            ct_cit += 1

        ct += 1

    write_to_file(get_file_base() + 'bert_data/sem_bert_unstemmed.json', seminal_p, seminal_x, seminal_y, 'seminal')

    survey_p = {}
    survey_x = {}
    survey_y = {}

    print('sur')

    ct = 0
    for p in survey_hlp['survey']:
        survey_p[ct] = do_bert(p['abs'], tokenizer)

        survey_x[ct] = {}
        survey_y[ct] = {}

        ct_ref = 0
        for ref in p['ref']:
            survey_x[ct][ct_ref] = do_bert(ref['abs'], tokenizer)
            ct_ref += 1

        ct_cit = 0
        for cit in p['cit']:
            survey_y[ct][ct_cit] = do_bert(cit['abs'], tokenizer)
            ct_cit += 1

        ct += 1

    write_to_file(get_file_base() + 'bert_data/sur_bert_unstemmed.json', survey_p, survey_x, survey_y, 'survey')

    print('uni')

    uninfluential_p = {}
    uninfluential_x = {}
    uninfluential_y = {}

    ct = 0
    for p in uninfluential_hlp['uninfluential']:
        uninfluential_p[ct] = do_bert(p['abs'], tokenizer)

        uninfluential_x[ct] = {}
        uninfluential_y[ct] = {}

        ct_ref = 0
        for ref in p['ref']:
            uninfluential_x[ct][ct_ref] = do_bert(ref['abs'], tokenizer)
            ct_ref += 1

        ct_cit = 0
        for cit in p['cit']:
            uninfluential_y[ct][ct_cit] = do_bert(cit['abs'], tokenizer)
            ct_cit += 1

        ct += 1

    write_to_file(get_file_base() + 'bert_data/uni_bert_unstemmed.json', uninfluential_p, uninfluential_x,
                  uninfluential_y, 'uninfluential')
