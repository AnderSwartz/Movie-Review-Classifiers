import numpy as np

print("running perceptron...")
f = open("vocab.txt","a",encoding='utf-8')
vocab = {}
index = 0
k = 2
stoplist = {}
punc = '''!()-[];:'"\,<>./?@#$%^&*_~'''
# with open("stoplist.txt",'r',encoding='utf-8') as file:
#     for line in file:
#         for word in line.split():
#             stoplist[word] = 1

with open("train/allneg.txt",'r',encoding='utf-8') as file:
    for line in file:
        for word in line.split():
            if word in punc:
                word = word.replace(word,"")
            word = word.lower()
            if(word in stoplist):
                continue
            if(vocab.get(word)==None):
                vocab[word]=index
                index=index+1

with open("train/allpos.txt",'r',encoding='utf-8') as file:
    for line in file:
        for word in line.split():
            if word in punc:
                word = word.replace(word,"")
            word = word.lower()
            if(word in stoplist):
                continue
            if(vocab.get(word)==None):
                vocab[word]=index
                index=index+1

f.truncate(0)
for key, value in vocab.items(): 
        f.write('%s:%s\n' % (key, value))
f.close

phi = np.zeros(index*k)

def featureFunction(review,label):
    offset = index*(label)
    bow = np.zeros(index*k)
    # bow = np.full(index*2,.1)
    for word in review.split():
        if word in punc:
           word = word.replace(word,"")
        word = word.lower()
        if(vocab.get(word)!=None): # if word in review is in vocab
            bow[vocab.get(word)+offset] +=1
    return bow

def scoringFunction(review,label,phi):
    featureOut = featureFunction(review,label)
    score = phi*featureOut #this is whats wrong
    score = np.sum(score)
    return score

def newScoringFunction(review,label,phi,bow):
    score = phi*bow #this is whats wrong
    score = np.sum(score)
    return score

def updatePhi(correctK):
    if(correctK==0):
        return (phi + (featureFunction(review,0)-featureFunction(review,1))) #when incorrectly guessed 1, add word counts to neg weights, sub from pos
    return ( phi +  (featureFunction(review,1)-featureFunction(review,0)))

def newUpdatePhi(bowExpected,bowPredicted):
    return phi + bowExpected - bowPredicted

def fastUpdatePhi(correctK,review):
    global phi
    for word in review.split():
        if word in punc:
           word = word.replace(word,"")
        word = word.lower()
        if(vocab.get(word)!=None): # if word in review is in vocab
            if(correctK==0):
                phi[vocab.get(word)] = phi[vocab.get(word)] + 1
                phi[vocab.get(word)+index] = phi[vocab.get(word)+index] - 1
            if(correctK==1):
                phi[vocab.get(word)] = phi[vocab.get(word)] -1
                phi[vocab.get(word)+index] = phi[vocab.get(word)+index] + 1

def fastScoringFunction(review,label):
    offset = index*(label)
    score = 0
    for word in review.split():
        if word in punc:
           word = word.replace(word,"")
        word = word.lower()
        if(vocab.get(word)!=None): # if word in review is in vocab
            score += phi[vocab.get(word)+offset]
    return score

def mle(review):
    scores = np.array([
    scoringFunction(review,0,phi),
    scoringFunction(review,1,phi)
    ])
    if(scores[0]==scores[1]):
        if(np.random.randint(2)==0): #tied, randomly guessing 0 or 1
            return 0
        return 1
    return(np.argmax(scores))

def fastMle(review):
    scores = np.array([
    fastScoringFunction(review,0),
    fastScoringFunction(review,1)
    ])
    if(scores[0]==scores[1]):
        if(np.random.randint(2)==0):
            # print("tied, randomly guessing 0")
            return 0
        # print("tied, randomly guessing 1")
        return 1
    return(np.argmax(scores))
print("training weights...    est. time to completion: 2 mins")

epochs = 1


#original slower functions V1:
for i in range(0,epochs):
    print("training epoch ", i," /", epochs," ...")
    with open("train/allneg.txt",'r',encoding='utf-8') as file:
        with open("train/allpos.txt",'r',encoding='utf-8') as file2:
            for review, review2 in zip(file,file2):
                # print("correct is 0")c
                nBow0 = featureFunction(review,0)
                nBow1 = featureFunction(review,1)
                nScore0 = newScoringFunction(review,0,phi,nBow0)
                nScore1 = newScoringFunction(review,1,phi,nBow1)
                if(nScore0<nScore1):     
                    phi = newUpdatePhi(nBow0,nBow1)
                elif(nScore0==nScore1):
                    if(np.random.randint(2)==0):
                        phi = newUpdatePhi(nBow0,nBow1)

                pBow0 = featureFunction(review2,0)
                pBow1 = featureFunction(review2,1)
                pScore0 = newScoringFunction(review2,0,phi,pBow0)
                pScore1 = newScoringFunction(review2,1,phi,pBow1)
                if(pScore0>pScore1):
                    phi = newUpdatePhi(pBow1,pBow0)
                elif(nScore0==nScore1):
                    if(np.random.randint(2)==0):
                        phi = newUpdatePhi(pBow1,pBow0)

#second modification using faster functions V2-V3:
# for i in range(0,epochs):
#     print("training epoch ", i," /", epochs," ...")
#     with open("train/allneg.txt",'r',encoding='utf-8') as file:
#         with open("train/allpos.txt",'r',encoding='utf-8') as file2:
#             for review, review2 in zip(file,file2):
#                 nScore0 = fastScoringFunction(review,0)
#                 nScore1 = fastScoringFunction(review,1)
#                 if(nScore0<nScore1):     
#                     fastUpdatePhi(0,review)
#                 elif(nScore0==nScore1):
#                     if(np.random.randint(2)==0):
#                         fastUpdatePhi(0,review)
#                 pScore0 = fastScoringFunction(review2,0)
#                 pScore1 = fastScoringFunction(review2,1)
#                 if(pScore0>pScore1):
#                     fastUpdatePhi(1,review2)
#                 elif(nScore0==nScore1):
#                     if(np.random.randint(2)==0):
#                         fastUpdatePhi(1,review2)

print("evaluating performance on training data...")
correctPosReview = 0
correctNegReview = 0
with open("train/allpos.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==1):
            correctPosReview+=1
    print("percent correctly guess as pos: ",correctPosReview/12500)

with open("train/allneg.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==0):
            correctNegReview+=1
    print("percent correctly guess as neg: ",correctNegReview/12500)
print("total training accuracy: ",((correctPosReview/12500)+(correctNegReview/12500))/2)

print("evaluating performance on testing data...")
correctNegReview = 0
correctPosReview = 0
with open("test/allpos.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==1):
            correctPosReview+=1
    print("percent correctly guess as pos: ",correctPosReview/12500)

with open("test/allneg.txt",'r',encoding='utf-8') as file:
    for review in file:
        if(fastMle(review)==0):
            correctNegReview+=1
    print("percent correctly guess as neg: ",correctNegReview/12500)
print("total testing accuracy: ",((correctPosReview/12500)+(correctNegReview/12500))/2)