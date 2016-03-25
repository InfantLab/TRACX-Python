"""
Created on Sat Feb  6 10:38:20 2016

@author: Caspar Addyman
"""


# module containing several version 1 & 2 of TRACX network
# * Implements the Truncated Recursive Atoassociative Chunk eXtractor
# * (TRACX, French, Addyman & Mareschal, Psych Rev, 2011) A neural network
# * that performs sequence segmentation and chunk extraction in artifical
# * grammar learning tasks and statistical learning tasks.
#
# Note: i've never programmed in Python before so this might get ugly.


import numpy as np
import sys, random
import time

class Tracx:
    """Truncated Autorecursive Chunk Extractor (TRACX) version 1 & 2.
    """


    def __init__(self):
        #default parameters
        self.TracxVersion = 2
        self.learningRate = 0.04
        self.recognitionCriterion = 0.4
        self.reinforcementProbability = 0.25
        self.momentum = 0.1
        self.temperature = 1.0
        self.fahlmanOffset = 0.1
        self.bias = -1
        self.sentenceRepetitions = 1
        self.trackingInterval  = 50 #how often do we test progress during traingin
        self.randomSeed = ""     #calculated from string value - leave blank for random
        self.numberSubjects = 1
        self.inputEncoding = "local"   # local,binary,user
        self.inputWidth = 8 #how many input nodes per encoded token
        self.sigmoidType = "tanh"
        self.deltaRule = "rms"  # "rms" or "max"
        self.coeff_tanh = 0.2197;
        # internal variables
        self.trackingFlag = False
        self.trackingSteps = []
        self.trackingResults = {}
        self.testErrorType = "conditional"
        self.testWords = []
        self.testNonWords = []
        self.testPartWords = []
        self.testData = []
        self.trainingData = []
        self.layer = [0,1] #for layer 0 & 1 weights
        self.deltas = [0,1] #for momentum
        
    # sigmoid functions
    def sigmoid(self,x):
        if self.sigmoidType == "logistic":
            return 1/(1+np.exp(-x))
        else:
            return np.tanh(x)   

    def d_sigmoid(self,x):
        if self.sigmoidType == "logistic":
            return x*(1-x)
        else:
            return self.temperature * (1 - x*x) + self.fahlmanOffset

    def set_training_data(self,data):
        '''what is the string of training data'''
        self.trainingData = data
        self.trainingLength = len(data)
    
    def set_tracking_words(self,testWords):
        '''test items we evaluate during training'''
        self.trackingFlag = True
        self.trackingWords = testWords
        self.trackingSteps = []

    def get_unique_items(self):
        '''all distinct tokens in training set'''
        output = set()
        for x in self.trainingData:
            output.add(x)
        return output

    def get_weights(self, layer=0):
        return self.layer[layer]

    def set_test_data(self,data):
        self.TestData = data

    def create_results_object(self):
        return {
            "trainSuccess": False,
            "elapsedTime": -1,
            "Words": {},
            "PartWords": {},
            "NonWords": {},
            "trackingSteps":    -1,
            "trackingOutputs": -1
        }
    def create_result_object(self):
        return {"inString" : "",
             "bigrams" : [],
             "tracxInputs":[],
             "tracxHidden":[],
             "tracxOutputs":[],
             "deltas": [],
             "totalDelta": 0,
             "meanDelta":  0,
             "testError": []
         }

    def set_input_encodings(self, newEncodings):
        #return the input encoding for this token
        self.inputEncodings = newEncodings
        #get inputwidth from one of the items        
        tempitem = list(newEncodings.values())[0]
        self.inputWidth = len(tempitem)

    def get_empty_hidden(self):
        #return the input encoding for this token
        return np.zeros(self.inputWidth + 1) 

    def get_input_encoding(self, token):
        #return the input encoding for this token
        return self.inputEncodings[token]
        
    def create_input_encodings(self,method="local"):
        #set up input vectors for each token
        self.inputEncoding = method
        tokens = self.get_unique_items() #find unique tokens
        #generate the input vectors
        self.inputEncodings = {}
        self.inputWidth = len(tokens)
        #include blank input as  a zero vector
        self.inputEncodings[" "] = np.zeros(self.inputWidth) 
        if self.inputEncoding == "binary":
            #binary encoding - each  letter numbered and
            #represented by corresponding 8bit binary array of -1 and 1.
            for idx in range(len(tokens)):
                #each input encoded as zeros everywhere
                ret = self.decimal_to_binary(idx+1)
                self.inputEncodings[tokens[idx]] = ret[0]
        elif  self.inputEncoding == "local":
            #local encoding - one column per letter.
            #i-th column +1, all others -1
            bipolarArray = -1. * np.ones(self.inputWidth)       
            idx = 0
            for x in tokens:
                bicopy = list(bipolarArray)
                bicopy[idx] = 1.
                idx += 1
                #each input encoded as zeros everywhere
                #except for i-th dimension
                self.inputEncodings[x] = list(bicopy)
        else:
            pass

    def get_current_step(self):
        #how many training steps have we taken?
        return self.currentStep

    def reset(self):
        #forget all training and reset weight matrices
        self.currentStep  = -1
        self.testResults = []
        if self.randomSeed:
            #if we have specified random seed use it
            self.randomSeed =  random.seed(self.randomSeed)
        else:
            #create one
            self.randomSeed = random.randint(0, sys.maxsize)
            random.Random(self.randomSeed)
        self.initialize_weights()

    def initialize_weights(self):
        N = self.inputWidth
        self.layer[0] = 2*np.random.random((2*N + 1, N)) - 1
        self.layer[1] = 2*np.random.random((N + 1, 2*N)) - 1
        #initialise momemtum weights too
        self.deltas[0] = 0 * self.layer[0]        
        self.deltas[1] = 0 * self.layer[1]        
        

    def get_last_training_tokens(self, n=2):
        #return the last n items seen in training
        if self.currentStep > n:
            return self.trainingData[:self.currentStep-n:self.currentStep]
        else:
            return None

    def decimal_to_binary(self,num1):
        binString = ""
        binArray = []
        bipolarArray = []
        if num1 >= 2**(self.inputWidth + 1):
            raise ValueError("Input value too large. Expecting value less than %d" % 2**(self.inputWidth +1))
        for pwr in range(self.inputWidth,1,-1):
            if num1 >= 2**(pwr-1):
                binString += "1"
                binArray += [1]
                bipolarArray += [1]
                num1 = num1 - 2**(pwr-1)
            else:
                binString += "0"
                binArray += [0]
                bipolarArray += [-1]
        return (binArray ,bipolarArray, binString)


    def network_output(self, token1,token2):
        if type(token1) is str:
            # build input & treat as vector so we can right-multiply
            input1 = self.inputEncodings[token1]
            input2 = self.inputEncodings[token2]
        else:
            input1 = token1
            input2 = token2
        inputfull = []
        inputfull.extend(input1)
        inputfull.extend(input2)
        inputfull.append(1.)  #1 for bias
        # multiply by first weight matrix
        # & pass through activation fn
        hidden =  self.sigmoid(np.dot(inputfull, self.layer[0])).tolist()
        hidden.append(1.) #1 for bias
        # multiply by second weight matrix
        # & pass through activation fn
        output  = self.sigmoid(np.dot(hidden,self.layer[1]))
        # calculate the delta between input and output
        # depending on which deltaRule we want to use
        # slice at end -1 to ignore bias
        deltaList = np.subtract(output,inputfull[:-1])
        if self.deltaRule == "max":
            delta = np.max(deltaList) 
        elif self.deltaRule == 'rms':
            delta = np.sqrt(    np.mean(np.square(deltaList)))
        return {"input": inputfull, "hidden": hidden, "output": output, "delta":delta}


    def back_propogate_error(self, net):
        # TODO
        # This code has not been optimised in any way.
        #
        # we want output to be same as input
        # so error to backProp is the difference between input and output
        layer_2_error = net["output"] - net["input"][:-1] #excluding bias 
        # so output errors is each diff multiplied by appropriate
        # derivative of output activation
        # 1st get deriv
        layer_2_delta = np.atleast_2d(layer_2_error * self.d_sigmoid(net["output"]))      
        # So change weights is this deriv times hidden activations
        dE_dw = np.atleast_2d(net["hidden"]) * layer_2_delta.T

        # multiplied by learning rate and with momentum added
        self.deltas[1] = dE_dw.T * self.learningRate + self.deltas[1] * self.momentum          

        # Errors on hidden layer are ouput errors back propogated
        layer_1_error = layer_2_delta.dot(self.layer[1].T)      

        layer_1_delta = layer_1_error * self.d_sigmoid(np.atleast_2d(net["hidden"]))
  
        #change in weights is this times inputs but not including bias  
        #hence the funny [:-1] slices at end of each array. Ugly huh?
        dE_dw = np.atleast_2d(net["input"]) * layer_1_delta[:,:-1].T
        
        self.deltas[0] = dE_dw.T * self.learningRate + self.deltas[0] * self.momentum
      
        # update weights 
        self.layer[0] -= self.deltas[0]
        self.layer[1] -= self.deltas[1]
        

    def step_forward(self, inString,start = 0, stop = -1,track_every = 1, trace = True, train = False):
        '''the main routine for stepping through inputs '''
        
        # how many steps do we train for on this call
        if stop <= 0: 
            untilStep = len(inString)-1
        else:
            untilStep = np.min(len(inString)-1,start+stop)

        stringResult = self.create_result_object()
        stringResult["InString"]= inString
        
        #initialise some variables
        token = [0, 1] 
        Input = [0, 1]
        net = {}
        net["delta"] = 99
        net["hidden"] = self.get_empty_hidden()
         
        # the main training loop
        for i in range(start,untilStep):
            token[0] = inString[i]
            token[1] = inString[i+1]
            Input[0] = self.inputEncodings[token[0]]
            Input[1] = self.inputEncodings[token[1]]            
            if self.TracxVersion == 1:        
                if i > start and net["delta"] < self.recognitionCriterion:
                    # new input is last hidden unit representation
                    Input[0] = net["hidden"][:-1]   # not including bias
                    token[0] = "#"
            else: #TRACX 2.0 French & Cotrell, Cogsci 2013
                d = np.tanh(self.coeff_tanh * net["delta"])   
                Input[0] = np.add(np.multiply(d,Input[0]),np.multiply(1-d,net["hidden"][:-1]))
                
            net = self.network_output(Input[0],Input[1])
            if train:
                if (self.TracxVersion == 2 or  #TRACX 2.0 always backprop 
                    net["delta"] > self.recognitionCriterion  or  #TRACX 1.0 train netowrk if error large
                    np.random.rand() <= self.reinforcementProbability): # or if probabilisticaly small error and below threshold
                    self.back_propogate_error(net)
                    net  = self.network_output(Input[0], Input[1])
            
            if trace:
                stringResult["bigrams"].append("".join(token))
                stringResult["deltas"].append(net["delta"])
                stringResult["tracxInputs"].append(Input)
                stringResult["tracxHidden"].append(net["hidden"])
                stringResult["tracxOutputs"].append(net["output"].tolist()) #simplfy type
                stringResult["totalDelta"] += net["delta"]

            if (track_every > 0 and i % track_every == 1):
                self.trackingSteps.append(i)
                # if tracking turned on we test the network
                # at fixed intervals with a set of test bigrams
                for x in self.trackingWords:
                    ret = self.test_string(x)
                    self.trackingResults[x].append([i, ret["totalDelta"]])

        stringResult["meanDelta"] = stringResult["totalDelta"]/(len(inString)-1)      
        return stringResult


    def train_network(self, steps = -1, track_every = 50,trace=False):
        return self.step_forward(self.trainingData,0,-1,track_every,trace = trace, train = True)        

    def test_string(self, inString):
        '''Get network output for a single word input.
        
        Passed a string  or list of arbitrary length, test_string passes along the string
        testing each bigram and return encodings and network activations.
        It also returns the average delta/error per bigram and the total delta.
        '''
        return self.step_forward(inString,0,-1,-1,trace = True, train = False)        
   
    def test_strings(self, inStrings):
        '''A function to test what the network has learned.

        We pass a list of test words ['ab','bc',...] or a comma separated
        string 'ab,bc,...' 
        It tests each one returning a dict object containing the
        test_string result for each word and overall mean delta per item.
        
        See also test_string
        '''
        stringResults = {}
        totalDelta = 0
        wordcount = 0
        if type(inStrings) is str:
            allwords = inStrings.split(",")
        else:
            allwords = ",".join(inStrings).split(",") #magic to combine list & csv
        for w in allwords:
            if len(w)>1:
                stringResults[w] = self.test_string(w)
                #keep running count of delta                
                wordcount += 1
                totalDelta += stringResults[w]["totalDelta"]
        if wordcount > 0:
            return {"items":allwords,"results":stringResults,"delta":totalDelta/wordcount }
        else:
            return {"items":allwords,"results":None,"delta":None}

    def test_categories(self, results_object):
        '''Tests how network performs on word, nonword, partword items.'''

        results_object["Words"] = self.test_strings(self.testWords)        
        results_object["PartWords"] =  self.test_strings(self.testPartWords)
        results_object["NonWords"] =  self.test_strings(self.testNonWords)
        
        return results_object
      
    def run_full_simulation(self, printProgress = False):
        self.reset()
        print("Random seed used: " + str(self.randomSeed))
        starttime =  time.time()
        if printProgress:
            print("Simulation started: " + time.strftime("%T",time.localtime()))

        self.currentStep = 0
        inputLength = len(self.trainingData) -1
        self.maxSteps = self.sentenceRepetitions * inputLength  
        testResults =self.create_results_object()
        print( 'Subjects: ')
        # loop round with a new network each time
        for  run_no in range(self.numberSubjects):
            if printProgress:
                print(str(1 + run_no) + ",")            
            if self.trackingFlag:
                # NB tracking data will be overwritten for each new participant                
                # initialise stacked array to store tracking data
                self.trackingSteps = []
                self.trackingResults = {}
                for x in self.trackingWords:
                    self.trackingResults[x] = []           
                                
            self.currentStep = 0
            self.initialize_weights()               
            if self.train_network(-1, printProgress):
                # training worked for this subject
                testResults = self.test_categories(testResults)

        if self.trackingFlag:
            testResults["trackingSteps"] = self.trackingSteps
            testResults["trackingOutputs"] = self.trackingResults
                            
        endtime =  time.time()
        elapsedTime = endtime - starttime
        testResults["elapsedTime"] = elapsedTime
        if printProgress:
            print("Finished. Duration: "+ "{:.3f}".format(elapsedTime) + " secs.")

        return testResults
    
    def step_through_training(self, stepSize, printProgress):
        '''
        The function which will step through the training process so user can
        see what is going on.
        '''
        if not self.currentStep or self.currentStep < 0:
            # initialize things
            starttime = time.time()
            if printProgress:
                print("Random seed used: " + self.randomSeed)
                print("Simulation started: " + time.strftime("%T",time.localtime()))
            self.maxSteps = self.sentenceRepetitions * len(self.trainingData)
            if printProgress:
                print("Stepping through once  ")
            if self.trackingFlag:
                # initialise stacked array to store tracking data
                self.trackingResults = []
                self.trackingSteps = []
                for x in self.trackingWords:
                    self.trackingResults[x] = []
            
        testResults =self.create_results_object()
                        
        if self.train_network(self, stepSize,  printProgress):
            testResults = self.test_categories(testResults)

#        get_test_stats(test)      
      
        endtime =  time.time()
        testResults["elapsedTime"] = endtime- starttime
        if printProgress:
            print("Finisehd. Duration: " + str(testResults["elapsedTime"]) + " secs. ")
       
        return testResults
        
#    def get_test_stats(self, results_object):
#        
#        testResults["Words"]["mean"] = np.mean(testResults["Words"]["all"])
#            testResults["Words"]["sd"] = np.std(testResults["Words"]["all"])
#        if len(testResults["PartWords"]["all"]) > 0:
#            testResults["PartWords"]["mean"] = np.mean(testResults["PartWords"]["all"])
#            testResults["PartWords"]["sd"] = np.std(testResults["PartWords"]["all"])
#        if len(testResults["NonWords"]["all"]) > 0:
#            testResults["NonWords"]["mean"] = np.mean(testResults["NonWords"]["all"])
#            testResults["NonWords"]["sd"] = np.std(testResults["NonWords"]["all"])
#        if self.trackingFlag:
#            testResults["trackingSteps"] = self.trackingSteps
#            testResults["trackingOutputs"] = self.trackingResults
