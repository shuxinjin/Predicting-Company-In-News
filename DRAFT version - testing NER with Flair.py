from flair.data import Sentence
from flair.models import SequenceTagger

# making a test sentence
sentence = Sentence('I love Finland!')

# loading the NER tagger
tagger = SequenceTagger.load('ner')

# running NER over the sentence
tagger.predict(sentence)

# testing out whether the model found out that Finland is the named entity.
print(sentence)
print('the following NER tags were wound')

# iterating over entities and printing
for entity in sentence.get_spans('ner'):
    print(entity)


# # # Creating Sentence # # #
sentence_2 = Sentence('The grass is green .')
print(sentence_2)

# # # Testing FlairEmbeddings # # #
from flair.embeddings import FlairEmbeddings

# init embedding
flair_embedding_forward = FlairEmbeddings('news-forward')

# creating a sentence
sentence = Sentence('The moon is made of cheese.')

# embedding words in the sentence with following
x = flair_embedding_forward.embed(sentence)



# # # Creating stacked embedding # # # 
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings

# creating stackedembedding object which combines 
stacked_embeddings = StackedEmbeddings([
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
])

sentence = Sentence('The moon is made of cheese.')

stacked_embeddings.embed(sentence)

# checking out the embedded tokens
for token in sentence:
    print(token)
    print(token.embedding)

# # # Training a Sequence labeling model # # # 
from flair.data import Corpus
from flair.datasets import WIKINER_ENGLISH
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List

# loading corpus
corpus: corpus = WIKINER_ENGLISH()
print(corpus)

# 2 choosing what tag type we want to predict
tag_type = 'ner'

# 3. making the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# 4. initializing embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('glove'),

    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initializing sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

# 6. initializing trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

# 7 start training
trainer.train('resources/taggers/example-ner',
                learning_rate=0.1,
                mini_batch_size=32,
                max_epochs=10)


# testing multiple sentences (article about provad)

test_1 = '''
Provad expands to Sweden
INDUSTRY  ›  SERVICE INDUSTRY  ›  TECHNOLOGY  ›  ICT

SHARE


Provad Sweden's country manager Jussi Mankki speaking before the audience here, thinks the importance of customer service cannot be stressed enough.
Provad Sweden's country manager Jussi Mankki speaking before the audience here, thinks the importance of customer service cannot be stressed enough.SCREENSHOT/PROVAD
Finnish customer service solutions provider Provad is expanding its operations to Sweden.

Provad has already secured two large customers in Sweden: customer services company K2C and financial sector group Svea Ekonomi.

Both companies will start using Provad’s customer service system and will make use of its different robotic solutions.

“When we can gather all communication channels on one platform and interface, we get an excellent overview of customer relationship management,” comments Markus Karlsson, contact centre manager at Svea Ekonomi.

The decision to expand to Sweden came naturally according to a blog post on Provad’s website, due to the company having Finnish customers operating in Sweden and the widespread recognition among Swedish businesses regarding the importance of progressive customer service systems.

“The importance of customer service cannot be stressed enough,” says Jussi Mankki, country manager of Provad Sweden. “On every level of any company, the significance of customer service on the business must be understood. A successful customer service requires a multi-channelled and flexible customer service system, capable of meeting today’s constantly changing customer expectations and demands.”
'''


# unikie
test_2 = """
Unikie drives the future of software development
TECHNOLOGY  ›  ICT  ›  APPLICATIONS & SOFTWARE

SHARE



Unikie is developing machine vision and shape recognition solutions that enable cars and other equipment to detect and dodge people and obstacles.UNIKIE
Taking Finnish engineering expertise overseas has always been a core element of the internationalisation strategy of Unikie, a software developer headquartered in Tampere.

Unikie, a software developer headquartered in Tampere, Finland, has succeeded in what most startup entrepreneurs dream of while racing from one technology event to another: signing collaboration agreements with several leading technology providers such as Nokia, Solita and Valmet.

Esko Mertsalmi, the chief executive of the four-year-old startup, uses an ice hockey analogy – a relatively common trait in that part of the country – to explain the success.

Founded in 2015, Unikie has swiftly grown into a major software development company.Founded in 2015, Unikie has swiftly grown into a major software development company.
UNIKIE
“Get the puck to the net and good things will happen,” he remarks self-deprecatingly.

“We’ve put in a lot of work,” he adds to expand on the analogy. “But maybe the logos of big international clients [in our references] have been a bit easier to come by than we dared to expect initially. I suppose we’ve just been doing the right things successfully.”

Unikie, he tells, was initially prepared to spend the first five years building a platform that enables it to run a profitable business domestically and pursue its internationalisation goals.

The startup is seemingly ahead of schedule. It announced recently it has signed agreements with a handful of major, but unnamed, automotive manufacturers in Central Europe. A growing number of its staff are consequently working on developing machine vision and pattern recognition solutions to enable cars and other machines to recognise and evade people and obstacles.

Its other main markets are telecommunications, smart industrial machines, and traditional internet and information technology.

“The solutions we’re increasingly providing for autonomous driving and smart industries, they require all of these four areas. It’s a smart strategy for allocating our competences. The cars are not driving themselves with all of the necessary stuff under the hood,” reminds Mertsalmi.

Steady, self-financed growth
Unikie has recorded a profit in each of its 48 months in operation and added to its headcount at an impressive pace. The company wrapped up last year with a staff of 250 spread across its subcontractor network and a turnover of 16 million euros, 80 per cent of which was derived from Finland.

“We’ve been growing on a cash flow-basis,” Mertsalmi tells.

The company was ranked one of Finland’s most promising startups earlier this year by Finnish financial publication Talouselämä.The company was ranked one of Finland’s most promising startups earlier this year by Finnish financial publication Talouselämä.
UNIKIE
“There haven’t been any big growth spurts. Four years, 250 people – that’s a pace of about 50 a year. The turnover target for this year is 30 million euros, but whether we’ll meet it still depends on a lot of things. We’ll break the 25 million-euro mark for sure, though.”

The startup, he views, has managed to carve out a share of the highly competitive market by taking advantage of the megatrend of digitalisation, aggressively pursuing international growth and – simply – offering unrivalled quality, regardless of cost.

“We aren’t competing with low prices; sometimes we’re more expensive than local alternatives. But we’re expensive because skilled labour costs money. We’re genuinely trying to identify the competences and find the best workers. And that’s a lot easier to justify when your prices are transparent,” he explains.

Although Unikie is involved in developing software for the machines and vehicles of tomorrow, it has one finger firmly on the pulse of the needs of existing technologies.

“We’re not only focusing on what’s coming 10 years from now, but we’re also developing solutions for today’s systems. I’m talking about things like how to update software remotely, how to enable motorists to add new features to their vehicles and how to connect phones to vehicles,” lists Mertsalmi.

Belief in Finnish engineering
Unikie’s belief in Finnish engineering expertise has proven another key differentiator, according to him.

“Internationalisation, for us at least, has always been about exporting Finnish engineering expertise. The prevalent thinking in the industry is that you should create a product and hope that it sells. But that’s a difficult way to generate traction for exports,” he says.

Mertsalmi points out that the digital transformation, which has been forced abruptly upon companies regardless of their industry, can be particularly challenging for well-established companies.

“The change needed to evolve into a technology firm in an agile fashion is massive for companies like automotive makers, which have grown into dominant brands on the strengths of their logistics systems,” he says.

Large corporations, he adds, may struggle to make the digital leap also due to the burden of their legacy solutions.

“Things don’t get simpler but more complicated, because many have a 20-year legacy of information technology solutions. You can’t just toss out the legacy, but it has to be taken into account in further development. That’s why the solution has to be more complicated,” explains Mertsalmi.

“Luckily Finland has plenty of capable people to solve these problems,” he adds.
"""


# plotting trianing curves and weights
from flair.visual.training_curves import Plotter
plotter = Plotter()
plotter.plot_training_curves('loss.tsv')
plotter.plot_weights('weights.txt')


# using the pretrained model from flair, 'ner' on above sentences
# looping through the sentences
for entity in sentence1.get_spans('ner'):
    print('type: ', type(entity))
    print('dir: ', dir(entity))
    print(entity.text())
    print(entity.tag_value())
    print(entity)

# sentence to dictionary:

sentence1_dict = sentence1.to_dict(tag_type='ner')

# find all names for organisation
# using stemmer to get "base name"
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

org_names = []
for entity in sentence1_dict['entities']:
    if entity['type'] == 'ORG':
        print(entity['text'])
        org_name_stemmed = stemmer.stem(entity['text'])
        org_names.append(org_name_stemmed)

# finding the most frequently occuring org from the list
import itertools
import operator
# helper function to find most common element in a list
def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    # print 'SL:', SL
    groups = itertools.groupby(SL, key=operator.itemgetter(0))
    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        # print 'item %r, count %r, minind %r' % (item, count, min_index)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


from flair.data import Sentence

# loading the NER tagger
tagger = SequenceTagger.load('ner')

# running NER over the sentence
tagger.predict(sentence)



def sentence_to_org(sentence):
    try:
        sentence_tokenized = Sentence(sentence)
        tagger.predict(sentence_tokenized)
        sentence_dict = sentence_tokenized.to_dict(tag_type='ner')

        org_names = []
        for entity in sentence_dict['entities']:
            if entity['type'] == 'ORG':
                org_names.append(entity['text'])
        
        predicted_org = most_common(org_names)
        return predicted_org
    except:
        print("Did not found any organisations from the text")