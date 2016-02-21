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
    """Truncated Autorecursive Chunk Extractor, version 1 (TRACX)."""


    def __init__(self):
        #default parameters
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
        self.deltaRule = "rms"  # "rms" or "max"
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
    def sigmoid(x):  
        return 1/(1+np.exp(-x))

    def d_sigmoid(x):
        return x*(1-x)
 
    def tanh(x):
        return np.tanh(x)
        
    def d_tanh(self, x):
        return  self.temperature * (1 - x*x) + self.fahlmanOffset

    def set_training_data(self,data):
        #what is the string of training data
        self.trainingData = data
        self.trainingLength = len(data)
    
    def set_tracking_words(self,testWords):
        self.trackingFlag = True
        self.trackingTestWords = testWords
        self.trackingSteps = []

    def get_unique_items(self):
        output = set()
        for x in self.trainingData:
            output.add(x)
        return output

    def get_weights(self, layer=0):
        return self.layer[layer]

    def set_test_data(self,data):
        self.TestData = data

    def get_input_encoding(self, token):
        #return the input encoding for this token
        return self.inputEncodings[token]

    def create_input_encodings(self,method="local"):
        #set up input vectors for each token
        self.inputEncoding = method
        tokens = self.get_unique_items() #find unique tokens
        if self.inputEncoding == "binary":
            #generate the input vectors
            self.inputEncodings = {}
            self.inputWidth = len(tokens)
            #binary encoding - each  letter numbered and
            #represented by corresponding 8bit binary array of -1 and 1.
            for idx in range(len(tokens)):
                #each input encoded as zeros everywhere
                ret = self.decimal_to_binary(idx+1)
                self.inputEncodings[tokens[idx]] = ret[0]
        elif  self.inputEncoding == "local":
            #local encoding - one column per letter.
            #i-th column +1, all others -1
            self.inputWidth = len(tokens)
            self.inputEncodings = {}
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
        Hid_net_in_acts =  np.dot(inputfull, self.layer[0])
        # pass through activation fn
        Hid_out_acts = np.tanh(Hid_net_in_acts).tolist()
        Hid_out_acts.append(1.) #1 for bias
        # multiply by second weight matrix
        Output_net_in_acts = np.dot(Hid_out_acts,self.layer[1])
        Output_out_acts = np.tanh(Output_net_in_acts)
        # calculate the delta between input and output
        # depending on which deltaRule we want to use
        # slice at end -1 to ignore bias
        deltaList = np.subtract(Output_out_acts,inputfull[:-1])
        if self.deltaRule == "max":
            delta = np.max(deltaList) 
        elif self.deltaRule == 'rms':
            delta = np.sqrt(    np.mean(np.square(deltaList)))
        return {"In": inputfull, "Hid": Hid_out_acts, "Out": Output_out_acts, "Delta":delta}


    def back_propogate_error(self, net):
        # TODO
        # This code has not been optimised in any way.
        #
        # we want output to be same as input
        # so error to backProp is the difference between input and output
        layer_2_error = net["Out"] - net["In"][:-1] #excluding bias 
        # so output errors is each diff multiplied by appropriate
        # derivative of output activation
        # 1st get deriv
        layer_2_delta = np.atleast_2d(layer_2_error * self.d_tanh(net["Out"]))
        
        # So change weights is this deriv times hidden activations
        dE_dw = np.atleast_2d(net["Hid"]) * layer_2_delta.T
        # multiplied by learning rate and with momentum added
        self.deltas[1] = dE_dw.T * self.learningRate + self.deltas[1] * self.momentum
            
        # Errors on hidden layer are ouput errors back propogated
        layer_1_error = layer_2_delta.dot(self.layer[1].T)
        
        layer_1_delta = layer_1_error * self.d_tanh(np.atleast_2d(net["Hid"]))
  
        #change in weights is this times inputs but not including bias  
        #hence the funny [:-1] slices at end of each array. Ugly huh?
        dE_dw = np.atleast_2d(net["In"]) * layer_1_delta[:,:-1].T
        
        self.deltas[0] = dE_dw.T * self.learningRate + self.deltas[0] * self.momentum
      
        # update weights (.T because these things always get tangled!!)
        self.layer[0] -= self.deltas[0]
        self.layer[1] -= self.deltas[1]

    def train_network(self, steps = -1, printProgress = False, batchMode = False):
        # TODO add try & catch code
        
        # how many steps do we train for on this call
        if steps <= 0:
            untilStep = self.maxSteps
        else:
            untilStep = np.min(self.maxSteps,self.currentStep+steps)

        lastDelta =99
        Input =[0,0]
        net = []
        
        # the main training loop
        while self.currentStep < untilStep:
            # read and encode the first bit of training data
            if lastDelta < self.recognitionCriterion:
                # new input is hidden unit representation
                Input[0] = net["Hid"][:-1] #not including the bias
            else:
                # input is next training item
                Input[0] = self.inputEncodings[self.trainingData[self.currentStep % self.trainingLength]]
            Input[1] = self.inputEncodings[self.trainingData[(self.currentStep + 1) % self.trainingLength]]

            net = self.network_output(Input[0],Input[1])

            # if on input the LHS comes from an internal representation then only
            # do a learning pass 25% of the time, since internal representations,
            # since internal representations are attentionally weaker than input
            # from the real, external world.
            if (lastDelta > self.recognitionCriterion  or  #train netowrk if error large
                np.random.rand() <= self.reinforcementProbability): # or if small error and below threshold
                self.back_propogate_error(net)
                net = self.network_output(Input[0], Input[1])
            lastDelta = net["Delta"]
            if not self.batchMode and printProgress and self.currentStep % self.trackingInterval == 1:
                self.trackingSteps.append(self.currentStep)
                # if tracking turned on we test the network
                # at fixed intervals with a set of test bigrams
                for x in self.trackingTestWords:
                    ret = self.test_string(x)
                    self.trackingResults[x].append([self.currentStep, ret["testError"]])
            self.currentStep += 1
        return True
#        catch(err){
#            console.log(err)
#            print("TRACX.trainNetwork Err: " + err.message + " ")
#            return False


#    /***
#     * a function to test what the network has learned.
#     * pass d list of test words, it
#     * tests each one returning deltas and mean delta
#     */
    def test_strings(self, testItems):
        deltas = []
        toterr = 0
        wordcount = 0
        for w in testItems:
            if len(w)>1:
                ret = self.test_string(w)
                toterr += ret["testError"]
                wordcount += 1
                deltas.append(ret["totalDelta"])
        if wordcount > 0:
            return {"delta":deltas,"testError":toterr/wordcount}
        else:
            return {"delta":[],"testError":[]}

#    /***
#     * a function to test what the network has learned.
#     * returns the total delta error on each letter pair in a string
#     * and a total delta for the word.
#     */
    def test_string(self, inString):
        '''
        a function to test what the network has learned.
        returns the total delta error on each letter pair in a string
        and a total delta for the word
        '''
        #TODO - try,except?
        net = {"Delta": 500}
        delta = []
        totDelta = 0
        input1Hidden = False
        Input = [0, 1]
        if self.testErrorType == "final":
            # used in the paper
            # always pass through hidden network activation
            CRITERION = 1000
        elif self.testErrorType == "conditional":
            # only use hidden activation if we have meet criterion
            CRITERION = self.recognitionCriterion
        else:
            # never pass the hidden activation
            CRITERION = -1
          
        for i in range(len(inString)-1):
            if i > 0 and net["Delta"] < CRITERION:
                # new input is hidden unit representation
                Input[0] = net["Hid"][:-1]                 # not including bias
                input1Hidden = True
            else:
                # input is next training item
                Input[0] = self.inputEncodings[inString[i]]
                input1Hidden = False
            Input[1] = self.inputEncodings[inString[i+1]]
            net = self.network_output(Input[0],Input[1])
            net["Input1Hidden"] = input1Hidden
            delta.append(net["Delta"])
            totDelta += net["Delta"]
     
        meanDelta = totDelta/(len(inString)-1)
        return {  "deltas":         delta,
                  "totalDelta":     totDelta,
                  "meanDelta":      meanDelta,
                  "finalDelta":     net["Delta"],
                  "testError": net["Delta"] if self.testErrorType == "final" else meanDelta,
                  "activations":net
              }


    def run_full_simulation(self, printProgress = True):
        self.reset()
        print("Random seed used: " + str(self.randomSeed))
        starttime =  time.clock()
        self.currentStep = 0
        inputLength = len(self.trainingData) -1
        self.maxSteps = self.sentenceRepetitions * inputLength
        if printProgress:
            print("Simulation started: " + str(starttime))
  
        # set up the object to store results
        testResults = {
            "trainSuccess": False,
            "elapsedTime": -1,
            "Words": {"mean":-1,"sd":-1,"all":[]},
            "PartWords": {"mean":-1,"sd":-1,"all":[]},
            "NonWords": {"mean":-1,"sd":-1,"all":[]},
            "trackingSteps":    -1,
            "trackingOutputs": -1
        }
        if self.trackingFlag:
            # initialise stacked array to store tracking data
            self.trackingSteps = []
            self.trackingResults = {}
            for x in self.trackingTestWords:
                self.trackingResults[x] = []

        print( 'Subjects: ')
        # loop round with a new network each time
        for  run_no in range(self.numberSubjects):
            # in batchmode we only track results of last participant
            self.batchMode = True if run_no < self.numberSubjects-1 else False
            self.currentStep = 0
            self.initialize_weights()
            if printProgress:
                print(str(1 + run_no) + ",")
                # print("Run: " + (1 + run_no) + " ")
                # print('Initial Weight Matrices Input to Hidden ')
                # print( weightsInputToHidden.inspect())
                # print(' Hidden to Output ')
                # print(weightsHiddenToOutput.inspect())

            if self.train_network(-1, printProgress):
                # training worked for this subject
                testResults["trainSuccess"] =True
                # TESTING THE NETWORK
                ret = self.test_strings(self.testWords )
                if ret["testError"]:
                    testResults["Words"]["all"].append(ret["testError"])
                ret =  self.test_strings(self.testPartWords )
                if ret["testError"]:
                    testResults["PartWords"]["all"].append(ret["testError"])
                ret =  self.test_strings(self.testNonWords )
                if ret["testError"]:
                    testResults["NonWords"]["all"].append(ret["testError"])

        self.batchMode = False

        endtime =  time.clock()
        testResults["elapsedTime"] = endtime- starttime
        if printProgress:
            print("Simulation finished: " + str(endtime))
            print("Duration: %.3f  secs. ", testResults["elapsedTime"])
        if self.trackingFlag:
            testResults["trackingSteps"] = self.trackingSteps
            testResults["trackingOutputs"] = self.trackingResults

        return testResults
    
    def step_through_training(self, stepSize, printProgress):
        '''
        The function which will step through the training process so user can
        see what is going on.
        '''
        self.batchMode = False
        if not self.currentStep or self.currentStep < 0:
            # initialize things
            startSimulation = datetime.datetime.now()
            if printProgress:
                print("Random seed used: " + self.randomSeed)
                print("Simulation started: " + startSimulation.toLocaleTimeString())
            lastDelta = 500  # some very big delta to start with
            currentStep = 0
            inputLength = trainingData.length -1
            self.maxSteps = self.sentenceRepetitions * inputLength
            if printProgress:
                print("Stepping through once  ")
            if self.trackingFlag:
                # initialise stacked array to store tracking data
                trackingResults = []
                trackingSteps = []
                for x in self.trackingTestWords:
                    trackingResults[trackingTestWords[x]] = []
            
        # set up the object to store results
        testResults = {"trainSuccess":    False,
                       "elapsedTime":    -1,
                       "Words":            {"mean":-1,"sd":-1,"all":[]},
                        "PartWords":       {"mean":-1,"sd":-1,"all":[]},
                        "NonWords":        {"mean":-1,"sd":-1,"all":[]},
                        "trackingSteps":   None,
                        "trackingOutputs": None}
                        
        # print( 'Subjects: ')
        # //loop round with a new network each time
        # for (var run_no=0 run_no<self.numberSubjectsrun_no++){
        if train_network(self, stepSize,  printProgress):
            #training worked for this subject
            testResults["trainSuccess"] =True
            # TESTING THE NETWORK
            # TESTING THE NETWORK
            ret = test_strings(testWords )
            if ret["testError"]:
                testResults["Words"]["all"].append(ret["testError"])
            ret =  test_strings(testPartWords )
            if ret["testError"]:
                testResults.PartWords["all"].append(ret["testError"])
            ret =  test_strings(testNonWords )
            if ret["testError"]:
                testResults["NonWords"]["all"].append(ret["testError"])
        if testResults["Words"]["all"].length > 0:
            testResults["Words"]["mean"] = np.mean(testResults["Words"]["all"])
            testResults["Words"]["sd"] = np.std(testResults["Words"]["all"])
        if testResults.PartWords["all"].length > 0:
            testResults.PartWords["mean"] = np.mean(testResults.PartWords["all"])
            testResults.PartWords["sd"] = np.std(testResults.PartWords["all"])
        if testResults["NonWords"]["all"].length > 0:
            testResults["NonWords"]["mean"] = np.mean(testResults["NonWords"]["all"])
            testResults["NonWords"]["sd"] = np.std(testResults["NonWords"]["all"])
        end = time ()
        testResults.elapsedTime = (end.getTime() - startSimulation.getTime())/1000
        if printProgress:
            print("Simulation finished: " + end.toLocaleTimeString() + " ")
            print("Duration: " + testResults.elapsedTime.toFixed(3) + " secs. ")
        if trackingFlag:
            testResults.trackingSteps = trackingSteps
            testResults.trackingOutputs = trackingResults
    
        return testResults