import numpy as np
print("running naive bayes...")

f = open("vocab.txt","a",encoding='utf-8')
vocab = {}
negVocab = {}
posVocab = {}
index = 0
negVocabTypeSize = 0
posVocabTypeSize = 0
negVocabTokenSize = 0
posVocabTokenSize = 0
mew = np.array([.5,.5])
lSmoothing = 5 #doesnt seem to matter  9 is best
k = 2
stoplist = {}
punc = '''!()-[];:'"\,<>./?@#$%^&*_~'''
with open("stoplist.txt",'r',encoding='utf-8') as file:
    for line in file:
        for word in line.split():
            stoplist[word] = 1

with open("train/allneg.txt",'r',encoding='utf-8') as file:
    for line in file:
        for word in line.split():
        #     if word in punc:
        #         word = word.replace(word,"")
        #     word = word.lower()
            if(word in stoplist):
                continue
            negVocab[word]=negVocab.get(word,0)+1
            negVocabTokenSize+=1
            if(vocab.get(word)==None):
                vocab[word]=index
                index=index+1

with open("train/allpos.txt",'r',encoding='utf-8') as file:
    for line in file:
        for word in line.split():
            # if word in punc:
            #     word = word.replace(word,"")
            # word = word.lower()
            if(word in stoplist):
                continue
            posVocabTokenSize+=1
            posVocab[word]=posVocab.get(word,0)+1
            if(vocab.get(word)==None):
                vocab[word]=index
                index=index+1
              
# f.truncate(0)
# for key, value in vocab.items(): 
#         f.write('%s:%s\n' % (key, value))
# f.close

def featureFunction(review,label):
    offset = index*(label)
    bow = np.zeros(index*k)
    for word in review.split():
        if word in punc:
           word = word.replace(word,"")
        word = word.lower()
        
        if(vocab.get(word)!=None): # if word in review is in vocab
            bow[vocab.get(word)+offset] +=1
    return bow

def rel_freq(word,label):
    if(label==0):
        return (negVocab.get(word,0)+lSmoothing)/((lSmoothing*index)+negVocabTokenSize)
    else:
        return (posVocab.get(word,0)+lSmoothing)/((lSmoothing*index)+posVocabTokenSize)

def computePhi():
    phi = np.zeros(index*k)
    for word in vocab:
        phi[vocab.get(word)]=rel_freq(word,0)
        phi[vocab.get(word)+index]=rel_freq(word,1)
    return phi

def fastScoringFunction(review,label):
    offset = index*(label)
    score = 0
    # bow = np.full(index*2,.1)
    for word in review.split():
        if word in punc:
           word = word.replace(word,"")
        word = word.lower()
        if(vocab.get(word)!=None): # if word in review is in vocab
            score += phi[vocab.get(word)+offset]
    return score

def fastMle(review):
    scores = np.array([
    fastScoringFunction(review,0),
    fastScoringFunction(review,1)
    ])
    return(np.argmax(scores))
    # Column vector θ of size 1 x K*|V|

print("setting weights...")
phi = np.log2(computePhi())

print("evaluating performance on training data...")
correctNegReview = 0
correctPosReview = 0
with open("train/allpos.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==1):
            correctNegReview+=1
    print("percent correctly guess as pos: ",correctNegReview/12500)

with open("train/allneg.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==0):
            correctPosReview+=1
    print("percent correctly guess as neg: ",correctPosReview/12500)
print("total training accuracy: ",((correctPosReview/12500)+(correctNegReview/12500))/2)

print("evaluating performance on testing data...")
correctNegReview = 0
correctPosReview = 0
with open("test/allpos.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==1):
            correctNegReview+=1
    print("percent correctly guess as pos: ",correctNegReview/12500)

with open("test/allneg.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==0):
            correctPosReview+=1
    print("percent correctly guess as neg: ",correctPosReview/12500)
print("total testing accuracy: ",((correctPosReview/12500)+(correctNegReview/12500))/2)