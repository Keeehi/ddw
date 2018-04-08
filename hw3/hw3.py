import nltk
import wikipedia
import warnings
warnings.filterwarnings("ignore")


def extract_named_entities(chunked_sentence):
    data = {}

    for entity in chunked_sentence:
        if isinstance(entity, nltk.tree.Tree):
            text = ' '.join([word for word, tag in entity.leaves()])
            ent = entity.label()
            data[text] = ent
        else:
            continue
    return data


def extract_custom_entities(tagged_sentence):
    data = {}

    adj = []
    nn = []

    for token in tagged_sentence:
        if token[1].startswith('JJ'):
            adj.append(token[0])
            continue

        if token[1].startswith('NN'):
            nn.append(token[0])
            continue

        if adj and nn:
            data[' '.join(adj + nn)] = 'CUS'

        adj = []
        nn = []

    if adj and nn:
        data[' '.join(adj + nn)] = 'CUS'

    return data


def get_wiki_class(tagged_sentence):
    result = []
    part = []

    for token in tagged_sentence:
        if token[1].startswith('JJ') or token[1].startswith('NN') or token[1].startswith('CC'):
            part.append(token[0])
        else:
            if part:
                result.append(part)
                part = []

    if part:
        result.append(part)

    return result


text = None
with open('orwell.txt', 'r') as f:
    text = f.read()

sentences = nltk.sent_tokenize(text)

tagged_sentences = [nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences]

chunked_sentences = nltk.ne_chunk_sents(tagged_sentences, binary=True)

entities = {}
for chunked_sentence in chunked_sentences:
    entities.update(extract_named_entities(chunked_sentence))

for tagged_sentence in tagged_sentences:
    entities.update(extract_custom_entities(tagged_sentence))

for key in entities:
    original_key = key
    while True:
        try:
            page = wikipedia.page(key)

            wikipedia_sentences = nltk.sent_tokenize(page.summary)

            if not wikipedia_sentences:
                print('({}) {} > {}: {}'.format(entities[original_key], original_key, key, 'Thing'))
                break

            wikipedia_tagged_first_sentence = nltk.pos_tag(nltk.word_tokenize(wikipedia_sentences[0]))

            split_key = list(map(lambda word: word.lower(), key.split()))

            print('({}) {} > {}: {}'.format(entities[original_key], original_key, key, ' '.join(max([list(filter(lambda word: word.lower() not in split_key, part)) for part in
                                        (get_wiki_class(wikipedia_tagged_first_sentence))], key=len))))
            break
        except wikipedia.exceptions.DisambiguationError as e:
            key = e.options[0]
        except wikipedia.exceptions.WikipediaException as e:
            print('({}) {} > {}: {}'.format(entities[original_key], original_key, key, 'Thing'))
            break
